"""
transform_taq.py

Transform data pulled from the WRDS Trade and Quote (TAQ) database .

This module provides functions to transform National Best Bid & Offer (NBBO) and
WRDS Consolidated Trades (WCT) data from TAQ. 

For more information about WRDS TAQ, see:
https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/taq/general/wrds-overview-taq/
"""

import datetime

import pandas as pd
import polars as pl
import pytz


# --------------------------------------------------------------------------
#  NBBO Transform
# --------------------------------------------------------------------------
def transform_datetime(df: pd.DataFrame) -> pd.DataFrame:
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
    
    if isinstance(df, pl.DataFrame):
        is_polars = True
        df = df.to_pandas()
    else:
        is_polars = False
    
    if df.empty or 'time_m_nano' not in df.columns:
        return df
    
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["time_m"] = pd.to_datetime(df["time_m"]).dt.time

    def _make_timestamp(row):
        # Combine 'date' and the 'time_m' column (a time)
        t = datetime.datetime.combine(row["date"], row["time_m"])
        # Convert to tz-aware + add nanoseconds
        pdt = (
            pd.to_datetime(t)
            .tz_localize(pytz.timezone('America/New_York'))
            + pd.Timedelta(row["time_m_nano"], unit="ns")
        )
        return pdt

    df["time_m"] = df.apply(_make_timestamp, axis=1)
    del df["time_m_nano"]  # remove the old nanosec column

    if "window_time" in df.columns:
        df = df.rename(columns={"time_m": "time_of_last_quote"})
        df["window_time"] = (
            pd.to_datetime(df["window_time"])
            .dt.tz_localize(pytz.timezone('America/New_York'))
        )
    else:
        df = df.rename(columns={"time_m": "time_quote"})
    if is_polars:
        df = pl.from_pandas(df)
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
    if isinstance(df, pl.DataFrame):
        is_polars = True
        df = df.to_pandas()
    else:
        is_polars = False
    
    if df.empty or 'time_m_nano' not in df.columns:
        return df
    
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["time_m"] = pd.to_datetime(df["time_m"]).dt.time

    def _make_timestamp(row):
        # Combine 'date' and the 'time_m' column (a time)
        t = datetime.datetime.combine(row["date"], row["time_m"])
        # Convert to tz-aware + add nanoseconds
        pdt = (
            pd.to_datetime(t)
            .tz_localize(pytz.timezone('America/New_York'))
            + pd.Timedelta(row["time_m_nano"], unit="ns")
        )
        return pdt

    df["time_m"] = df.apply(_make_timestamp, axis=1)
    del df["time_m_nano"]  # remove the old nanosec column

    df = df.rename(columns={"time_m": "time_quote"})
    if is_polars:
        df = pl.from_pandas(df)
    return df