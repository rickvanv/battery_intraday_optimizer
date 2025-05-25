import io
import logging
from logging import config

import numpy as np
import pandas as pd
import requests
from mcp_data_loading import get_MCP_token
from mcpclient import mcp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from price_modelling.global_keys import *

config.fileConfig("src/logging.ini")
logger = logging.getLogger(__name__)


def read_trades_data(filename: str) -> pd.DataFrame:
    # Append base url with filename
    base_url = "https://wham-nginx-ui-container.mcp.bam.corp.vattenfall.com/public-trades-parsed/"
    url = base_url + filename

    # Configure retry strategy
    retry_strategy = Retry(
        total=5,  # Increased from default
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    # Create a session with retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    try:
        # Attempt to download the file
        response = session.get(url, verify="Vattenfall_Root_CA_2017.cer")
        response.raise_for_status()  # Raise an error for bad status codes

        # Try to read the Parquet file
        df_trades_data = pd.read_parquet(io.BytesIO(response.content))
        return df_trades_data

    except Exception as e:
        logger.info("[ERROR] Could not read data from {url}: {e}")
        return None


def read_DA_prices(
    ref_date_begin: pd.Timestamp, ref_date_end: pd.Timestamp, country: str
) -> pd.DataFrame:
    mcp_token = get_MCP_token("prd", "MCP_TOKEN")

    ts_name = DA_price_timeseries[country]["ts_name"]
    price_column = DA_price_timeseries[country]["price_column"]
    df_DA_prices = mcp.read(
        name=ts_name,
        ref_date_begin=ref_date_begin,
        ref_date_end=ref_date_end,
        timezone="UTC",
        token=mcp_token,
    )

    df_DA_prices = df_DA_prices[["delivery_begin", price_column]]
    df_DA_prices = df_DA_prices.rename(
        columns={
            "delivery_begin": "delivery_start_hour",
            price_column: "DA_price",
        }
    )

    return df_DA_prices


def process_trades_data(
    df_trades_data: pd.DataFrame, reference_date: pd.Timestamp, country: str
) -> pd.DataFrame:
    df_trades_data_processed = df_trades_data.copy()

    # only select trades from country
    country_eic_codes = price_zones_eic_dict[country].values()
    df_trades_data_processed = df_trades_data_processed.loc[
        df_trades_data_processed["eic"].isin(country_eic_codes)
    ]

    # Change ms datetime format to %Y-%M-%d %H:%m:%s
    for column in df_trades_data_processed.columns:
        if "_ms" in column:
            df_trades_data_processed[column] = pd.to_datetime(
                df_trades_data_processed[column], unit="ms", utc=True
            )
            df_trades_data_processed = df_trades_data_processed.rename(
                columns={column: column[:-3]}
            )

    # Round trades datetime stamp to 15 minutes.
    df_trades_data_processed["traded"] = df_trades_data_processed[
        "traded"
    ].dt.floor("15min")

    # Select only trades for 15-minute period contracts
    df_trades_data_processed["contract_length"] = (
        df_trades_data_processed["delivery_end"]
        - df_trades_data_processed["delivery_start"]
    )
    df_trades_data_processed = df_trades_data_processed.loc[
        df_trades_data_processed["contract_length"] == pd.Timedelta(minutes=15)
    ]

    # Select only trades in the last 15 minutes
    df_trades_data_processed["time_to_delivery"] = np.maximum(
        0,
        df_trades_data_processed["delivery_start"]
        - df_trades_data_processed["traded"],
    )
    df_trades_data_processed = df_trades_data_processed.loc[
        df_trades_data_processed["time_to_delivery"] >= pd.Timedelta(minutes=0)
    ]
    if not df_trades_data_processed.empty:
        # Calculate weighted average prices for every 15 min period.
        df_trades_data_processed = (
            calculate_weighted_avg_by_delivery_and_trade_time(
                df_trades_data_processed
            )
        )

        df_trades_data_processed = df_trades_data_processed.loc[
            df_trades_data_processed["delivery_start"].dt.date
            == reference_date
        ]

    df_trades_data_processed["delivery_start_hour"] = df_trades_data_processed[
        "delivery_start"
    ].dt.floor("h")

    return df_trades_data_processed


def calculate_weighted_avg_by_delivery_and_trade_time(
    df_trades_data: pd.DataFrame,
) -> pd.DataFrame:
    # Group by both delivery period and traded_ms
    df_trades_grouped = df_trades_data.groupby(["delivery_start", "traded"])

    # Calculate weighted average price for each group
    # TODO: cleanup
    def weighted_avg(group):
        return (
            1000
            * np.sum(group["price"] / (10**5) * group["quantity"])
            / np.sum(group["quantity"])
        )

    weighted_results = df_trades_grouped.apply(
        weighted_avg, include_groups=False
    )

    # Convert to DataFrame for better presentation
    result = weighted_results.reset_index()
    result.columns = ["delivery_start", "traded", "VWAP"]

    return result.sort_values(["delivery_start", "traded"])


def create_vwap_df(
    ref_date_begin: pd.Timestamp, ref_date_end: pd.Timestamp, country: str
) -> pd.DataFrame:
    dates = pd.date_range(start=ref_date_begin, end=ref_date_end)
    df_vwaps = pd.DataFrame()
    for datum in dates:
        # Consider dataframes of yesterday and today together to get all trades
        # for the delivery_periods on todate.
        yesterdate = (datum - pd.Timedelta(1, "day")).date()

        logger.info("Retrieving data for (%s)", datum.date())
        df_trades = pd.DataFrame()
        for date in pd.date_range(start=yesterdate, freq="1d", periods=3):
            df_trades_date = read_trades_data(str(date.date()) + ".parquet")
            df_trades = pd.concat([df_trades, df_trades_date])

        if not df_trades.empty:
            df_vwaps_date = process_trades_data(
                df_trades, datum.date(), country
            )
            df_vwaps = pd.concat([df_vwaps, df_vwaps_date])
        else:
            continue
    return df_vwaps


if __name__ == "__main__":
    ref_date_begin = "2023-01-01"
    yesterday = pd.Timestamp("today") - pd.Timedelta(1, "day")
    ref_date_end = yesterday
    # ref_date_end = "2025-06-08"
    country = "DE"

    # Calculate vwap and return results as dataframe
    df_vwaps = create_vwap_df(
        ref_date_begin=ref_date_begin,
        ref_date_end=ref_date_end,
        country=country,
    )

    # Retrieve DA prices for same period
    df_DA_prices = read_DA_prices(
        ref_date_begin=ref_date_begin,
        ref_date_end=ref_date_end,
        country=country,
    )

    # merge DA price df with VWAP df.
    df_all_vwap_data = pd.merge(
        left=df_vwaps, right=df_DA_prices, on="delivery_start_hour"
    )
    # calculate difference between VWAP and DA price
    df_all_vwap_data["delta_VWAP_DA"] = (
        df_all_vwap_data["VWAP"] - df_all_vwap_data["DA_price"]
    )

    # save data to csv
    df_all_vwap_data.to_csv(
        f"data/15min_vwap_{country}_2024.csv", sep=";", index=False
    )
