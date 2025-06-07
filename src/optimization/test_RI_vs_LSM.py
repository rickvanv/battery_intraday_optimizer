from src.optimization.LSM import LSMStrategy
from src.optimization.rolling_intrinsic import RollingIntrinsicStrategy
import time


if __name__ == "__main__":
    from src.battery import Battery
    import pandas as pd
    import yaml
    import os
    import pickle

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    DA_prices_day_test = pd.read_csv("DA_prices_day_test.csv", sep=",")

    # unpickle price_generator_DA
    with open("price_generator_DA.pkl", "rb") as f:
        price_generator_DA = pickle.load(f)

    # read config.yml
    config_path = os.path.join(os.path.dirname(current_dir), "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    battery_LSM = Battery(config)

    # time LSM training runtime
    start_time_training = time.time()
    lsm_strategy = LSMStrategy(battery_LSM, price_generator_DA, config)
    lsm_strategy.run(DA_prices_day_test)
    end_time_training = time.time()

    # Genrate truth price path
    W_array, price_path = price_generator_DA.generate_price_paths(
        season="spring",
        N_paths=1,
        df_DA_prices=DA_prices_day_test,
    )

    # time LSM dispatch runtime
    start_time_dispatch = time.time()
    revenues, pi, soc_schedule, cycle_count = (
        lsm_strategy.calculate_optimal_dispatch(price_path[0])
    )
    end_time_dispatch = time.time()

    print(
        f"LSM training runtime: {end_time_training - start_time_training} seconds"
    )
    print(
        f"LSM dispatch runtime: {end_time_dispatch - start_time_dispatch} seconds"
    )
    print(f"Value of storage: {lsm_strategy.results["value_of_storage"]}")

    print(f"Revenues from LSM dispatch: {sum(revenues)}")
    print(f"Cycle count from LSM dispatch: {cycle_count}")
    battery_RI = Battery(config)
    # time RI dispatch runtime
    start_time_dispatch = time.time()
    ri_strategy = RollingIntrinsicStrategy(battery_RI, W_array[0], config)
    (
        power_committed,
        power_incremental_schedules,
        soc_schedule,
        price_path,
        results,
        cycle_count,
    ) = ri_strategy.run()
    end_time_dispatch = time.time()

    print(
        f"RI dispatch runtime: {end_time_dispatch - start_time_dispatch} seconds"
    )
    print(f"Revenues from RI dispatch: {sum(results)}")
    print(f"Cycle count from RI dispatch: {cycle_count}")
