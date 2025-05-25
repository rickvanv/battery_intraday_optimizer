import numpy as np


class Battery:
    def __init__(self, config):
        """
        Initialize the Battery class.

        Args:
            config: Dictionary of configuration parameters.
        """
        self.config = config
        self.capacity = config["capacity"]
        self.battery_size_hours = config["battery_size_hours"]
        self.battery_size_energy = self.battery_size_hours * self.capacity
        self.capacity_normalized = self.capacity / self.battery_size_energy
        self.charge_efficiency = config["charge_efficiency"]
        self.discharge_efficiency = config["discharge_efficiency"]
        self.time_step = config["time_step"]
        self.history = []
        self.N_cycles_till_now = 0
        self.N_daily_cycles_max = config["N_daily_cycles_max"]
        # Initialize the SOC and power schedule
        self.soc_schedule = config["initial_soc"] * np.ones(
            config["N_timesteps"] + 1
        )
        self.power_charging_schedule = np.zeros(config["N_timesteps"])
        self.power_discharging_schedule = np.zeros(config["N_timesteps"])

    def update_soc_schedule(self, t, new_soc_schedule, price_path):
        """
        Update the state of charge (SOC) of the battery.

        Args:
            new_soc_schedule: New SOC schedule.
            price_path: Price path.
        """
        remaining_window_size_start = max(
            0, (np.shape(price_path)[1] - np.shape(price_path)[0]) + t
        )
        self.soc_schedule[remaining_window_size_start:] = new_soc_schedule

    def update_power_schedule(
        self,
        t,
        power_flow_into_battery,
        power_flow_out_of_battery,
        price_path,
    ):
        """
        Update the battery power schedule.

        Args:
            power_flow_into_battery: Power charging schedule.
            power_flow_out_of_battery: Power discharging schedule.
            price_path: Price path.
        """
        remaining_window_size_start = max(
            0, (np.shape(price_path)[1] - np.shape(price_path)[0]) + t
        )
        self.power_charging_schedule[remaining_window_size_start:] = (
            power_flow_into_battery
        )
        self.power_discharging_schedule[remaining_window_size_start:] = (
            power_flow_out_of_battery
        )
