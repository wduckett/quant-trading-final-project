"""
transform_taq.py

Transform data pulled from the WRDS Trade and Quote (TAQ) database .

This module provides functions to transform National Best Bid & Offer (NBBO) and
WRDS Consolidated Trades (WCT) data from TAQ. 

For more information about WRDS TAQ, see:
https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/taq/general/wrds-overview-taq/
"""

from typing import Union

import pandas as pd
import polars as pl
from polars import col, when


# --------------------------------------------------------------------------
#  NBBO Transform
# --------------------------------------------------------------------------

def transform_nbbo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame from get_taq_nbbo().

    Returns
    -------
    pd.DataFrame
        The transformed data with new timestamp columns.
    """
    
    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # Convert time_m to timedelta (ensuring correct format)
    df["time_m"] = pd.to_timedelta(df["time_m"])

    # Combine date and time_m into a single datetime column
    df["time_quote"] = pd.to_datetime(df["date"]) + df["time_m"]

    # Convert to timezone-aware datetime
    df["time_quote"] = df["time_quote"].dt.tz_localize("America/New_York")

    # Handle nanoseconds if present
    if 'time_m_nano' in df.columns:
        df["time_quote"] += pd.to_timedelta(df["time_m_nano"], unit="ns")
        df.drop(columns=["time_m_nano"], inplace=True)

    # Rename columns based on `window_time` presence
    if "window_time" in df.columns:
        df.rename(columns={"time_quote": "time_of_last_quote"}, inplace=True)
        df["window_time"] = pd.to_datetime(df["window_time"]).dt.tz_localize("America/New_York")

    return df


# --------------------------------------------------------------------------
#  WCT Transform
# --------------------------------------------------------------------------

def transform_taq_wct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame from get_taq_wct().

    Returns
    -------
    pd.DataFrame
        The transformed data with new timestamp columns.
    """
    
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["time_m"] = pd.to_timedelta(df["time_m"])

    # Combine date and time_m into a single datetime column
    df["time_m"] = pd.to_datetime(df["date"]) + df["time_m"]

    # Convert to timezone-aware datetime
    df["time_m"] = df["time_m"].dt.tz_localize("America/New_York")

    # Drop unnecessary columns if needed
    df = df.rename(columns={"time_m": "time_trade"})  

    return df

# --------------------------------------------------------------------------
#  Feature Extraction
# --------------------------------------------------------------------------

def extract_features_taq(df: Union[pl.DataFrame, pd.DataFrame]                     
    ) -> Union[pl.DataFrame]:
    """
    Extract features from the WCT and NBBO DataFrames:
    - Duration since last trade: Time since the last trade.
    - Mid price: Average of the best bid and ask prices.
    - Spread: Difference between the best ask and bid prices.
    - Size imbalance: Difference between the best ask and bid sizes divided by their sum.
    - Trade direction (sign): +1 for buy, -1 for sell.
    - Next mid price: Mid price of the next quote.
    - Next mid-price change: Change in mid price from the current to the next quote.
    
    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        The merged dataframe of NBBO and WCT.

    Returns
    -------
    Union[pl.DataFrame, pd.DataFrame]
        The DataFrame with extracted features.
    """

    # Check wether to return a polars or pandas DataFrame
    if isinstance(df, pl.DataFrame):
        is_polars = True
    else:
        is_polars = False
        df = pl.from_pandas(df)

    # Compute some microstructure features
    df = df.with_columns([
        (col("time_trade").diff().dt.total_microseconds()).alias("duration_since_last_trade"),
        ((col("best_bid") + col("best_ask")) / 2).alias("mid_price"),
        (col("best_ask") - col("best_bid")).alias("spread"),
        # Simple size imbalance measure (avoid dividing by zero)
        when(
            (col("best_bidsizeshares") + col("best_asksizeshares")) > 0
        ).then(
            (col("best_bidsizeshares") - col("best_asksizeshares"))
            / (col("best_bidsizeshares") + col("best_asksizeshares"))
        ).otherwise(None).alias("size_imbalance")
    ])


    # Infer trade direction (sign): +1 if trade prints at or above ask, -1 if at or below bid, else 0
    df = df.with_columns(
        when(col("price") >= col("best_ask")).then(1)
        .when(col("price") <= col("best_bid")).then(-1)
        .otherwise(0)
        .alias("trade_direction")
    )

    # For simple modeling/backtesting, label the "next-step" mid-price change so we can do a one-step-ahead prediction or PnL simulation.
    df = df.with_columns([
        col("mid_price").shift(-1).alias("mid_price_next"),
        (col("mid_price").shift(-1) - col("mid_price")).alias("mid_price_change_next")
    ])

    # Convert back to pandas DataFrame if necessary
    if not is_polars:
        df = df.to_pandas()

    return df

    





