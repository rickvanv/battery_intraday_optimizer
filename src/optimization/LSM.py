import numpy as np
from src.optimization.base_strategy import BaseStrategy
from typing import Union
import os
import time
import pandas as pd

eps = 1e-8


class LSMStrategy(BaseStrategy):
    """
    LSM strategy class to be used for optimization of battery trading.
    This model uses a Ornstein-Uhlenbeck (OU) process to model price paths and
    generate correlated paths for different delivery start times.
    """

    def __init__(self, battery, price_generator, config):
        # TODO: add logging statements
        """
        Initialize the LSM strategy.

        Args:
            battery: An instance of your Battery class.
            config: Dictionary of configuration parameters.
        """
        super().__init__(battery, config)
        self.price_generator = price_generator
        self.N_price_paths = config["N_price_paths"]
        self.N_timesteps = config["N_timesteps"]
        self.results = {}

        self.N_SOC_levels = config["N_SOC_levels"]
        self.N_SOC_levels_plus_one = self.N_SOC_levels + 1

        self.SOC_levels = np.linspace(0, 1, self.N_SOC_levels_plus_one)
        # Find index of SOC level that is closest to the initial SOC level.
        self.initial_SOC_level_index = np.argmin(
            np.abs(self.SOC_levels - self.battery.initial_soc)
        )

        # Set max SOC change for charging and discharging.
        self.max_SOC_change_charge = (
            self.config["time_step"]
            / self.battery.size_hours
            * self.battery.charge_efficiency
        )
        self.max_SOC_change_discharge = self.config["time_step"] / (
            self.battery.size_hours * self.battery.discharge_efficiency
        )

        # negative actions are charging, positive actions are discharging.
        self.actions = np.concatenate(
            (
                -np.flip(self.SOC_levels[1:]),
                self.SOC_levels,
            )
        )

        self.basis_functions_type = config["basis_functions_type"]
        self.N_basis_functions = config["N_basis_functions"]
        self.discount_factor = config.get("discount_factor", 1.0)
        self.results = {
            "continuation_values": None,
            "pi": None,
            "afc": None,
            "value_of_storage": None,
        }
        self.fitted_polynomials = None

    def get_poly_class(self):
        """
        Get basis functions class for the LSM strategy.
        """
        if self.basis_functions_type == "Polynomial":
            return np.polynomial.Polynomial
        if self.basis_functions_type == "Laguerre":
            return np.polynomial.Laguerre
        if self.basis_functions_type == "Hermite":
            return np.polynomial.Hermite
        if self.basis_functions_type == "Chebyshev":
            return np.polynomial.Chebyshev
        if self.basis_functions_type == "Legendre":
            return np.polynomial.Legendre
        if self.basis_functions_type == "Fourier":
            raise NotImplementedError(
                "Fourier basis functions are not supported."
            )

    def run(self, df_DA_prices: pd.DataFrame):
        """
        Execute the trading strategy over the simulation period.
        Must return a pandas DataFrame with at least: time, action, soc, price.
        """
        self.price_paths = self.generate_price_paths(df_DA_prices)
        self.results = self.LSM_algorithm()
        return self.results

    def pay_off_function(
        self,
        price: Union[float, np.ndarray],
        SOC_change: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate the pay-off function for the LSM strategy.
        Accounts for battery efficiencies.
        - If inputs are scalars, returns scalar.
        - If price and SOC_change are 1D arrays of same shape, returns element-wise product.
        - If price and SOC_change are 1D arrays of different shapes, computes the outer product.
        """
        price_arr = np.asanyarray(price)
        soc_arr = np.asanyarray(SOC_change)

        # Case for outer product
        if (
            price_arr.ndim == 1
            and soc_arr.ndim == 1
            and price_arr.shape != soc_arr.shape
        ):
            payoff = np.outer(price_arr, soc_arr) * self.battery.size_energy
            # Apply efficiencies based on action sign (on the columns of the result)
            payoff[:, soc_arr > 0] *= self.battery.discharge_efficiency
            payoff[:, soc_arr < 0] /= self.battery.charge_efficiency
            return payoff

        # Case for element-wise (scalar-scalar, scalar-array, array-array with same shape)
        else:
            payoff = np.multiply(price, SOC_change) * self.battery.size_energy

            # For array operations
            if isinstance(payoff, np.ndarray):
                payoff[soc_arr > 0] *= self.battery.discharge_efficiency
                payoff[soc_arr < 0] /= self.battery.charge_efficiency
            # For scalar operations
            elif np.isscalar(payoff):
                if SOC_change > 0:
                    payoff *= self.battery.discharge_efficiency
                elif SOC_change < 0:
                    payoff /= self.battery.charge_efficiency
            return payoff

    def generate_price_paths(self, df_DA_prices: pd.DataFrame):
        """
        Generate price paths using the price generator.
        """
        _, P = self.price_generator.generate_price_paths(
            season="spring",
            N_paths=self.N_price_paths,
            df_DA_prices=df_DA_prices,
        )
        return P

    def LSM_algorithm(self):
        """
        Implement the LSM algorithm.
        """
        results = {}

        # Initialize arrays``
        continuation_values = np.zeros(
            (self.N_price_paths, self.N_timesteps, self.N_SOC_levels_plus_one)
        )
        pi = np.zeros(
            (self.N_price_paths, self.N_timesteps, self.N_SOC_levels_plus_one)
        )
        afc = np.zeros(
            (self.N_price_paths, self.N_timesteps, self.N_SOC_levels_plus_one)
        )
        fitted_polynomials = np.empty(
            (self.N_timesteps, self.N_SOC_levels_plus_one), dtype=object
        )

        # Assign contract value at maturity (last timestep) - Vectorized
        last_prices = self.price_paths[:, -1]
        soc_changes = np.minimum(
            self.SOC_levels, self.max_SOC_change_discharge
        )
        afc[:, self.N_timesteps - 1, :] = self.pay_off_function(
            last_prices, soc_changes
        )

        # Iterate backwards in time
        for t in range(self.N_timesteps - 2, -1, -1):
            for soc_level_index, soc_level in enumerate(self.SOC_levels):
                # Fit the basis functions to the continuation values.
                poly_class = self.get_poly_class()
                basis_functions = poly_class.fit(
                    self.price_paths[:, t],
                    afc[:, t + 1, soc_level_index],
                    self.N_basis_functions,
                )
                fitted_polynomials[t, soc_level_index] = basis_functions

                # Calculate the continuation values
                continuation_values[:, t, soc_level_index] = (
                    self.discount_factor
                    * basis_functions(self.price_paths[:, t])
                )

            for soc_level_index, soc_level in enumerate(self.SOC_levels):
                # Return an array of possible actions, that satisfies the constraints.
                possible_actions = self.actions[
                    ((soc_level - self.actions) <= 1)
                    * ((soc_level - self.actions) >= 0)
                    * (self.actions >= -self.max_SOC_change_charge)
                    * (self.actions <= self.max_SOC_change_discharge)
                ]

                # Find the new SOC level indices for the possible actions (vectorized).
                new_socs_for_actions = soc_level - possible_actions
                new_soc_level_indices = np.argmin(
                    np.abs(
                        self.SOC_levels[:, np.newaxis] - new_socs_for_actions
                    ),
                    axis=0,
                )

                # Rewards for every possible action
                possible_rewards = (
                    self.pay_off_function(
                        self.price_paths[:, t],
                        possible_actions,
                    )
                    + continuation_values[:, t, new_soc_level_indices]
                )

                # Calculate the "best" decision in every path
                pi[:, t, soc_level_index] = possible_actions[
                    np.argmax(possible_rewards, axis=1)
                ]

                # Calculate the previous AFC (vectorized over price paths)
                pi_actions = pi[:, t, soc_level_index]
                payoffs = self.pay_off_function(
                    self.price_paths[:, t], pi_actions
                )

                new_socs = soc_level - pi_actions
                # Find the index of the closest SOC level for each new SOC.
                new_indices = np.argmin(
                    np.abs(self.SOC_levels[:, np.newaxis] - new_socs), axis=0
                )

                # Get continuation values for next timestep using advanced indexing
                continuation = (
                    self.discount_factor
                    * afc[np.arange(self.N_price_paths), t + 1, new_indices]
                )

                afc[:, t, soc_level_index] = payoffs + continuation

        value_of_storage = np.mean(afc[:, 0, self.initial_SOC_level_index])

        # Store results in the results dictionary.
        self.fitted_polynomials = fitted_polynomials
        results["continuation_values"] = continuation_values
        results["pi"] = pi
        results["afc"] = afc
        results["value_of_storage"] = value_of_storage
        return results

    def calculate_optimal_dispatch(self, price_path: np.ndarray):
        """
        Calculate the optimal dispatch for a given realized price path.
        """
        cycle_count = 0
        revenues = np.zeros(self.N_timesteps)
        pi = np.zeros(self.N_timesteps)

        initial_soc_level = self.SOC_levels[self.initial_SOC_level_index]
        soc_schedule = np.ones(self.N_timesteps + 1) * initial_soc_level

        threshold = 1e-6

        for t in range(self.N_timesteps - 1):
            current_soc = soc_schedule[t]
            current_price = price_path[t]

            possible_actions = self.actions[
                ((current_soc - self.actions) <= 1)
                * ((current_soc - self.actions) >= 0)
                * (self.actions >= -self.max_SOC_change_charge)
                * (self.actions <= self.max_SOC_change_discharge)
            ]

            new_socs_for_actions = current_soc - possible_actions
            new_soc_level_indices = np.argmin(
                np.abs(self.SOC_levels[:, np.newaxis] - new_socs_for_actions),
                axis=0,
            )
            new_soc_level_zero_index = np.argmin(np.abs(self.SOC_levels - 0))

            # use regression coefficients to calculate the continuation values
            fitted_polys = self.fitted_polynomials[t, new_soc_level_indices]
            continuation_values = np.array(
                [poly(current_price) for poly in fitted_polys]
            )
            continuation_values *= self.discount_factor

            # Calculate the "best" decision in every path
            current_payoffs = self.pay_off_function(
                current_price,
                possible_actions,
            )

            # penalty_per_unit_cycle = (
            #     cycle_count / self.battery.N_daily_cycles_max
            # ) * (current_payoffs + continuation_values)
            # cycle_penalty = penalty_per_unit_cycle * np.abs(possible_actions)

            if (
                np.abs(
                    np.max(current_payoffs + continuation_values)
                    - (current_payoffs + continuation_values)[
                        new_soc_level_zero_index
                    ]
                )
                < 500
            ):
                pi[t] = possible_actions[new_soc_level_zero_index]
            else:
                pi[t] = possible_actions[
                    np.argmax(current_payoffs + continuation_values)
                ]

            # Calculate the revenue for the current timestep
            revenues[t] = self.pay_off_function(current_price, pi[t])

            soc_schedule[t + 1] = current_soc - pi[t]
            cycle_count += abs(pi[t])

        # Calculate the revenue for the final timestep
        if cycle_count < self.battery.N_daily_cycles_max:
            revenues[self.N_timesteps - 1] = self.pay_off_function(
                price_path[-1], soc_schedule[self.N_timesteps - 1]
            )

        return revenues, pi, soc_schedule, cycle_count


if __name__ == "__main__":
    from src.price_modelling.price_path_generator import IDPriceSimulator
    from src.battery import Battery
    import pandas as pd
    import yaml
    import os
    import pickle

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # # Navigate to the data directory
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(current_dir)),
        "data",
        "all_VWAP_DA_data_spring.csv",
    )

    # load data
    df_all_VWAP_data_DA = pd.read_csv(data_path, sep=";")
    # set columns to datetime type
    df_all_VWAP_data_DA["delivery_start"] = pd.to_datetime(
        df_all_VWAP_data_DA["delivery_start"]
    )
    df_all_VWAP_data_DA["delivery_start_hour"] = pd.to_datetime(
        df_all_VWAP_data_DA["delivery_start_hour"]
    )
    df_all_VWAP_data_DA["traded"] = pd.to_datetime(
        df_all_VWAP_data_DA["traded"]
    )
    df_all_VWAP_data_DA = df_all_VWAP_data_DA.loc[
        np.abs(df_all_VWAP_data_DA["delta_VWAP_DA"]) < 300
    ]

    # DA_prices_test = df_all_VWAP_data_DA[["delivery_start", "DA_price"]].copy()
    # # Sample a random day from the DA_prices dataframe
    # random_day = DA_prices_test["delivery_start"].dt.date.sample(1).values[0]
    # DA_prices_day_test = DA_prices_test[
    #     DA_prices_test["delivery_start"].dt.date == random_day
    # ]
    # # Remove duplicate delivery start times
    # DA_prices_day_test = DA_prices_day_test.drop_duplicates(
    #     subset=["delivery_start"]
    # )

    # DA_prices_day_test = pd.read_csv("DA_prices_day_test.csv", sep=",")

    # read config.yml
    config_path = os.path.join(os.path.dirname(current_dir), "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    price_generator_DA = IDPriceSimulator(config)
    price_generator_DA.fit_model(df_all_VWAP_data_DA)

    # pickle price_generator_DA
    with open("price_generator_DA.pkl", "wb") as f:
        pickle.dump(price_generator_DA, f)

    # # unpickle price_generator_DA
    # with open("price_generator_DA.pkl", "rb") as f:
    #     price_generator_DA = pickle.load(f)

    # read config.yml
    config_path = os.path.join(os.path.dirname(current_dir), "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # battery = Battery(config)

    # # time LSM training runtime
    # start_time_training = time.time()
    # lsm_strategy = LSMStrategy(battery, price_generator_DA, config)
    # lsm_strategy.run(DA_prices_day_test)
    # end_time_training = time.time()

    # W_array, price_path = price_generator_DA.generate_price_paths(
    #     season="spring",
    #     N_paths=1,
    #     df_DA_prices=DA_prices_day_test,
    # )

    # # time LSM dispatch runtime
    # start_time_dispatch = time.time()
    # revenues, pi, soc_schedule = lsm_strategy.calculate_optimal_dispatch(
    #     price_path[0]
    # )
    # end_time_dispatch = time.time()

    # print(
    #     f"LSM training runtime: {end_time_training - start_time_training} seconds"
    # )
    # print(
    #     f"LSM dispatch runtime: {end_time_dispatch - start_time_dispatch} seconds"
    # )
    # print(f"Value of storage: {lsm_strategy.results["value_of_storage"]}")

    # print(f"Revenues from dispatch: {sum(revenues)}")
