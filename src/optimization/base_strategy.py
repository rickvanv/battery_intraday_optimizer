from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, battery, config):
        """
        Abstract base for battery trading strategies.

        Args:
            battery: An instance of your Battery class.
            price_generator: Instance of IDPriceSimulator.
            config: Dictionary of configuration parameters.
        """
        self.battery = battery
        self.config = config

    @abstractmethod
    def run(self):
        """
        Execute the trading strategy over the simulation period.
        Must return a pandas DataFrame with at least: time, action, soc, price.
        """
        pass
