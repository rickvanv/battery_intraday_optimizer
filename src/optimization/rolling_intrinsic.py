import numpy as np
from optimization.base_strategy import BaseStrategy
import pyomo.environ as pyo


class RollingIntrinsicStrategy(BaseStrategy):
    """
    Rolling intrinsic strategy class to be used for optimization of battery trading.
    This model uses an Ornstein-Uhlenbeck process to model price paths and
    generate correlated paths for different delivery start times.
    """

    def __init__(self, battery, price_path_array, config):
        # TODO: add logging statements
        """
        Initialize the RollingIntrinsic strategy.

        Args:
            battery: An instance of your Battery class.
            price_path_array: numpy array of price paths.
            config: Dictionary of configuration parameters.
        """
        super().__init__(battery, price_path_array, config)
        self.price_path_array = price_path_array
        self.optimized_paths = None
        self.N_timesteps = config["N_timesteps"]
        self.results = np.zeros(config["N_trading_timesteps"])

    def run(self):
        """
        Execute the trading strategy over the simulation period.
        Must return a pandas DataFrame with at least: time, action, soc, price.
        """
        self.results, self.power_schedule, self.soc_schedule = (
            self.optimize_on_path()
        )
        return (
            self.power_schedule,
            self.soc_schedule,
            self.price_path_array,
            self.results,
        )

    def optimize_on_path(self):
        """
        Optimize battery operation on price path.
        """

        initial_price_path = self.price_path_array[0, :, :]

        # Initialize battery schedule and SOC
        self.battery.power_schedule = np.zeros(np.shape(initial_price_path)[1])
        self.battery.power_schedules = np.zeros(np.shape(initial_price_path))
        self.battery.soc_schedules = np.zeros(
            [
                np.shape(initial_price_path)[0],
                np.shape(initial_price_path)[1] + 1,
            ]
        )

        # Optimize on each timestep
        for t in range(np.size(initial_price_path, 0)):
            remaining_window_size_start = max(
                0,
                (
                    np.shape(initial_price_path)[1]
                    - np.shape(initial_price_path)[0]
                )
                + t,
            )
            remaining_price_path = initial_price_path[
                t, remaining_window_size_start:
            ]
            model = self.define_pyomo_model(
                price_path=remaining_price_path,
                optimization_window_timesteps=len(remaining_price_path),
                t=t,
            )

            # solve optimization problem
            results = pyo.SolverFactory("glpk").solve(model)

            # Check if the solve was successful before accessing results
            if (results.solver.status == pyo.SolverStatus.ok) and (
                results.solver.termination_condition
                == pyo.TerminationCondition.optimal
            ):
                # update battery SOC and power schedule
                # Access values directly from the model instance
                soc_values = np.array(list(model.soc.get_values().values()))
                power_flow_into_battery_values = np.array(
                    list(model.power_flow_into_battery.get_values().values())
                )
                power_flow_out_of_battery_values = np.array(
                    list(model.power_flow_out_of_battery.get_values().values())
                )

                self.battery.update_soc_schedule(
                    t,
                    soc_values,
                    initial_price_path,
                )
                self.battery.update_power_schedule(
                    t,
                    power_flow_into_battery_values,
                    power_flow_out_of_battery_values,
                    initial_price_path,
                )
                # if t == 46:
                #     print("we are here")

                if (self.N_timesteps - len(remaining_price_path)) > 0:
                    self.battery.N_cycles_till_now = (
                        (
                            sum(
                                self.battery.power_discharging_schedule[
                                    : (t - 36) + 1
                                ]
                            )
                            / self.battery.discharge_efficiency
                            + sum(
                                self.battery.power_charging_schedule[
                                    : (t - 36) + 1
                                ]
                            )
                            * self.battery.charge_efficiency
                        )
                        * self.battery.time_step
                        / 2.0
                    ) - 1e-3

                self.results[t] = self.results[t - 1] + pyo.value(
                    model.objective
                )

            else:
                print(
                    f"Solver did not find an optimal solution at timestep {t}"
                )
                # Handle the case where the solver fails
                break

            self.battery.power_schedules[t, :] = (
                -self.battery.power_charging_schedule
                + self.battery.power_discharging_schedule
            )
            self.battery.soc_schedules[t, :] = self.battery.soc_schedule
        return (
            self.results,
            self.battery.power_schedules,
            self.battery.soc_schedules,
        )

    def define_pyomo_model(
        self, price_path: np.array, optimization_window_timesteps, t
    ):
        """
        Define the pyomo model.
        Replaces np.maximum in SOC update with linear constraints.
        """
        model = pyo.ConcreteModel()

        # --- SETS ---
        # Define the time steps for the optimization window (e.g., 0, 1, ..., N-1 hours)
        model.T = pyo.RangeSet(0, optimization_window_timesteps - 1)
        model.T_plus_one = pyo.RangeSet(
            0, optimization_window_timesteps
        )  # For SOC update rule

        # --- PARAMETERS ---
        model.price = pyo.Param(
            model.T, within=pyo.Reals, initialize=price_path
        )  # Price forecast for the window
        model.initial_soc_schedule = pyo.Param(
            model.T_plus_one,
            within=pyo.NonNegativeReals,
            initialize=np.maximum(
                self.battery.soc_schedule[
                    (self.N_timesteps - optimization_window_timesteps) :
                ],
                0,
            ),
        )  # Initial SoC for this window
        model.time_step_duration = pyo.Param(
            within=pyo.NonNegativeReals, default=self.battery.time_step
        )  # e.g., 1 for 1 hour
        model.max_power_charge = pyo.Param(
            model.T,
            within=pyo.NonNegativeReals,
            initialize=np.maximum(
                self.battery.capacity_normalized * np.ones(len(model.T))
                - self.battery.power_schedule[
                    (self.N_timesteps - optimization_window_timesteps) :
                ],
                0,
            ),
        )
        model.max_power_discharge = pyo.Param(
            model.T,
            within=pyo.NonNegativeReals,
            initialize=np.maximum(
                self.battery.capacity_normalized * np.ones(len(model.T))
                + self.battery.power_schedule[
                    (self.N_timesteps - optimization_window_timesteps) :
                ],
                0,
            ),
        )

        model.max_soc = pyo.Param(within=pyo.NonNegativeReals, default=1.0)
        model.min_soc = pyo.Param(within=pyo.NonNegativeReals, default=0.0)
        model.charge_efficiency = pyo.Param(
            within=pyo.NonNegativeReals, default=self.battery.charge_efficiency
        )
        model.discharge_efficiency = pyo.Param(
            within=pyo.NonNegativeReals,
            default=self.battery.discharge_efficiency,
        )

        # Schedule for this window (ensure this slice is correct)
        model.power_charging_schedule = pyo.Param(
            model.T,
            within=pyo.Reals,
            initialize=self.battery.power_charging_schedule[
                (self.N_timesteps - optimization_window_timesteps) :
            ],
        )
        model.power_discharging_schedule = pyo.Param(
            model.T,
            within=pyo.Reals,
            initialize=self.battery.power_discharging_schedule[
                (self.N_timesteps - optimization_window_timesteps) :
            ],
        )
        # --- VARIABLES ---
        model.soc = pyo.Var(
            model.T_plus_one,
            domain=pyo.NonNegativeReals,
            bounds=(model.min_soc, model.max_soc),
        )
        model.power_charge = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
        )  # Raw charging power from external source
        model.power_discharge = pyo.Var(
            model.T, domain=pyo.NonNegativeReals
        )  # Raw discharging power to external sink
        model.is_charging = pyo.Var(model.T, domain=pyo.Binary)

        # Auxiliary variables for the effective power flow into/out of the battery terminals
        # These replace the np.maximum terms in the SOC update
        model.power_flow_into_battery = pyo.Var(
            model.T, domain=pyo.NonNegativeReals
        )
        model.power_flow_out_of_battery = pyo.Var(
            model.T, domain=pyo.NonNegativeReals
        )

        # Set bounds for raw power_charge and power_discharge variables
        for t_idx in model.T:
            model.power_charge[t_idx].bounds = (
                0,
                model.max_power_charge[t_idx],
            )
            model.power_discharge[t_idx].bounds = (
                0,
                model.max_power_discharge[t_idx],
            )

        # --- CONSTRAINTS ---
        # Initial SOC constraint
        def initial_soc_rule(model):
            return model.soc[0] == model.initial_soc_schedule[0]

        model.initial_soc_constraint = pyo.Constraint(rule=initial_soc_rule)

        # Constraints to link raw power, schedule, binary, and effective power flows
        # These linearize the logic of the original np.maximum terms

        # M is a number, greater than any possible power flow value.
        # The power flow at each stage is limited by 2 * self.battery.capacity_normalized.
        M = 2 * self.battery.capacity_normalized

        # If is_charging is 1, power_flow_into_battery should be power_charge + schedule,
        # and power_flow_out_of_battery should be 0.
        # If is_charging is 0, power_flow_into_battery should be 0,
        # and power_flow_out_of_battery should be power_discharge - schedule.

        # Link power_flow_into_battery and power_flow_out_of_battery to the net power (charge - discharge + schedule)
        # This ensures that the net effect on SOC is accounted for by the effective flows.
        def net_power_flow_link_rule(model, t):
            # The net power at the interaction point is power_charge - power_discharge + schedule
            # This net power must equal the net effective flow into/out of the battery (into - out)
            return (
                model.power_discharge[t]
                + model.power_discharging_schedule[t]
                - model.power_charge[t]
                - model.power_charging_schedule[t]
            ) == model.power_flow_out_of_battery[
                t
            ] - model.power_flow_into_battery[
                t
            ]

        model.net_power_flow_link_constraint = pyo.Constraint(
            model.T, rule=net_power_flow_link_rule
        )

        # Use the binary variable to enforce that only one of power_flow_into_battery
        # or power_flow_out_of_battery can be non-zero at any time step.
        # This is a standard big-M formulation for selecting between two non-negative variables.

        # If is_charging is 1, power_flow_out_of_battery must be 0.
        def effective_flow_binary_link_out(model, t):
            return model.power_flow_out_of_battery[t] <= M * (
                1 - model.is_charging[t]
            )

        model.effective_flow_binary_link_out_constraint = pyo.Constraint(
            model.T, rule=effective_flow_binary_link_out
        )

        # If is_charging is 0, power_flow_into_battery must be 0.
        def effective_flow_binary_link_into(model, t):
            return model.power_flow_into_battery[t] <= M * model.is_charging[t]

        model.effective_flow_binary_link_into_constraint = pyo.Constraint(
            model.T, rule=effective_flow_binary_link_into
        )

        # Directly link power_charge/discharge to binary for stricter mutual exclusion
        def link_charge_binary_rule(model, t):
            return (
                model.power_charge[t]
                <= model.is_charging[t] * model.max_power_charge[t]
            )

        model.link_charge_binary_constraint = pyo.Constraint(
            model.T, rule=link_charge_binary_rule
        )

        def link_discharge_binary_rule(model, t):
            return (
                model.power_discharge[t]
                <= (1 - model.is_charging[t]) * model.max_power_discharge[t]
            )

        model.link_discharge_binary_constraint = pyo.Constraint(
            model.T, rule=link_discharge_binary_rule
        )

        # SOC update constraint using the new effective power flow variables
        # This replaces the original rule with np.maximum
        def soc_update_rule(model, t):
            return (
                model.soc[t + 1]
                == model.soc[t]
                + (
                    model.power_flow_into_battery[t] * model.charge_efficiency
                    - model.power_flow_out_of_battery[t]
                    / model.discharge_efficiency
                )
                * model.time_step_duration
            )

        model.soc_update_constraint = pyo.Constraint(
            model.T, rule=soc_update_rule
        )

        # Define number of cycles in the new power schedule
        model.N_cycles = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(
                self.battery.N_cycles_till_now,
                self.battery.N_daily_cycles_max,
            ),
        )

        def N_cycles_rule(model):
            # total_normalized_throughput is sum of (normalized energy actually flowing into battery per step)
            # and (normalized energy actually flowing out of battery per step)
            total_normalized_throughput = sum(
                (
                    model.power_flow_into_battery[t_idx]
                    * model.charge_efficiency
                    + model.power_flow_out_of_battery[t_idx]
                    / model.discharge_efficiency
                )
                * model.time_step_duration
                for t_idx in model.T
            )
            # Divide by 2.0 because one cycle involves E_max charging AND E_max discharging
            return (
                model.N_cycles
                == self.battery.N_cycles_till_now
                + total_normalized_throughput / 2.0
            )

        model.N_cycles_calculation_constraint = pyo.Constraint(
            rule=N_cycles_rule
        )

        # model.N_cycles_limit_constraint = pyo.Constraint(
        #     rule=lambda model: model.N_cycles <= model.N_cycles_max
        # )

        # --- OBJECTIVE ---
        # Assuming revenue is from power exported to the grid and cost is from power imported from the grid.
        # Net power injected into grid = model.power_discharge[t] - model.power_charge[t] + model.schedule[t]
        # If positive, this is export. If negative, this is import.

        # Introduce auxiliary variables for grid export and import (non-negative)
        model.P_grid_export = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.P_grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        # Objective: Maximize (Revenue from discharging) - (Cost of charging)
        def objective_rule(model):
            return sum(
                (model.power_discharge[t] - model.power_charge[t])
                * model.price[t]
                for t in model.T
            )

        model.objective = pyo.Objective(
            rule=objective_rule, sense=pyo.maximize
        )

        return model
