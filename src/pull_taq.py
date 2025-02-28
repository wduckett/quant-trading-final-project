"""
pull_taq.py

Pull data from the WRDS Trade and Quote (TAQ) database with caching support.

This module provides functions to download National Best Bid & Offer (NBBO) and
WRDS Consolidated Trades (WCT) data from TAQ. It follows the caching logic
used in pull_nasdaq.py (e.g., _cache_filename, _file_cached, etc.), and keeps
SQL queries separate from any data transformations. 

The Trade and Quote (TAQ) database contains intraday transactions data (trades and quotes)
for all securities listed on the NYSE, AMEX, Nasdaq NMS, and all other U.S. equity
exchanges. TAQ is often used for intraday analyses such as:
    - Daily volatility estimation
    - Probability of informed trading
    - Short-term impact of breaking news
    - Back-testing intraday trading strategies

For more information about WRDS TAQ, see:
https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/taq/general/wrds-overview-taq/

Examples
--------
You can fetch NBBO data for a particular date and ticker, returning a Polars DataFrame by default:
    df_nbbo = get_taq_nbbo(
        date='2023-01-03',
        ticker='AAPL',
        wrds_username='my_wrds_username'
    )
Then do any transformations in transform_taq_nbbo(df_nbbo).

Similarly for WCT data:
    df_wct = get_taq_wct(
        date='2023-01-03',
        ticker=['AAPL','MSFT'],
        wrds_username='my_wrds_username',
        use_polars=False
    )
Then transform in transform_taq_wct(df_wct).

-------------------------------------------------------------
"""

import datetime
import logging
from functools import lru_cache
from pathlib import Path
from typing import Union, List, Any, Dict

import pandas as pd
import polars as pl
import pytz
import wrds

# Local imports - adjust to your package structure
from ..settings import Config
from ..utils.utils import (
    _cache_filename,
    _hash_cache_filename,
    _file_cached,
    _read_cached_data,
    write_cache_data,
    _flatten_dict_to_str
)


RAW_DATA_DIR = Config.RAW_DATA_DIR
WRDS_USERNAME = Config.WRDS_USERNAME


# --------------------------------------------------------------------------
#  NBBO Transform
# --------------------------------------------------------------------------
def transform_taq_nbbo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example transform for NBBO bars. We create a proper 'time_of_last_quote'
    timestamp with nanoseconds, plus set 'window_time' as a localized
    datetime for the bar boundary.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame from get_taq_nbbo().

    Returns
    -------
    pd.DataFrame
        The transformed data with new timestamp columns.
    """
    if df.empty:
        return df

    def _make_timestamp(row):
        # Combine 'date' and the 'time_of_last_quote' column (a time)
        t = datetime.datetime.combine(row["date"], row["time_of_last_quote"])
        # Convert to tz-aware + add nanoseconds
        pdt = (
            pd.to_datetime(t)
            .tz_localize(pytz.timezone('America/New_York'))
            + pd.Timedelta(row["time_of_last_quote_ns"], unit="ns")
        )
        return pdt

    df["time_of_last_quote"] = df.apply(_make_timestamp, axis=1)
    del df["time_of_last_quote_ns"]  # remove the old nanosec column

    df["window_time"] = (
        pd.to_datetime(df["window_time"])
        .dt.tz_localize(pytz.timezone('America/New_York'))
    )

    return df


# --------------------------------------------------------------------------
#  WCT Transform (placeholder)
# --------------------------------------------------------------------------
def transform_taq_wct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example transform for WCT data. Adapt as needed; for now, it's a placeholder.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame from get_taq_wct().

    Returns
    -------
    pd.DataFrame
        Possibly cleaned/adjusted WCT data.
    """
    # Insert any transformation logic you need, e.g. timestamp creation:
    # ...
    return df


# --------------------------------------------------------------------------
#  NBBO Data Function
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_taq_nbbo(
    tickers: Union[str, List[str]],
    date: Union[str, datetime.date],
    bar_minutes: int = 30,
    use_polars: bool = True,
    wrds_username: str = WRDS_USERNAME,
    hash_file_name: bool = False,
    data_dir: Union[None, Path] = RAW_DATA_DIR,
    do_transform: bool = True
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Retrieve intraday NBBO data from TAQ (e.g., 30-min bars) for the given date
    and tickers. Uses local caching plus an LRU in-memory cache.

    This logic follows your existing code snippet, with:
        - A CTE that partitions quotes by hour & bar_minutes
        - ROW_NUMBER() selecting the last quote in each bar
        - Timestamps localized & combined with nanoseconds in transform_taq_nbbo()

    Parameters
    ----------
    tickers : str or list of str
        One or more ticker symbols. e.g. 'AAPL' or ['AAPL','MSFT'].
    date : str or datetime.date
        The trading date (e.g. '2023-01-03'). If string with '-' it is parsed.
    bar_minutes : int, optional
        The bar size in minutes (default 30). Must divide 60 or be 60 itself.
    use_polars : bool, optional
        If True, returns a Polars DataFrame; otherwise returns Pandas.
    wrds_username : str, optional
        Your WRDS username. Defaults to Config.WRDS_USERNAME.
    hash_file_name : bool, optional
        If True, uses a hashed filename for cache. Otherwise uses a verbose name.
    data_dir : Path or None, optional
        Local directory for caching. Defaults to Config.RAW_DATA_DIR.
    do_transform : bool, optional
        Whether to run transform_taq_nbbo() on the result. Default True.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        The final NBBO bars. By default, a Polars DataFrame (if use_polars=True).

    Notes
    -----
    - If you need monthly vs. daily logic, rename the table in the SQL as appropriate
      (e.g. taqm_{year}.complete_nbbo_{YYYYMMDD}, or wrds_taqs_nbbo.nbbo_YYYYMMDD, etc.).
    """
    # Parse date
    if isinstance(date, datetime.date):
        date_dt = date
    else:
        if "-" in date:
            date_dt = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        else:
            # e.g. '20230103'
            date_dt = datetime.datetime.strptime(date, "%Y%m%d").date()

    date_str = date_dt.strftime('%Y%m%d')
    year_str = date_dt.strftime('%Y')

    # Ensure bar_minutes is valid
    assert bar_minutes == 60 or (bar_minutes <= 30 and 30 % bar_minutes == 0), \
        "bar_minutes must be 60 or a divisor of 30"

    # Force tickers into a tuple
    if isinstance(tickers, str):
        tickers_tuple = (tickers,)
    else:
        tickers_tuple = tuple(tickers)

    # Build a filter string for caching
    filters: Dict[str, Any] = {
        "tickers": tickers_tuple,
        "date": date_str,
        "bar_minutes": bar_minutes
    }
    filter_str = _flatten_dict_to_str(filters)

    # Check file cache
    if hash_file_name:
        cache_paths = _hash_cache_filename("taq_nbbo_bars", filter_str, data_dir)
    else:
        cache_paths = _cache_filename("taq_nbbo_bars", filter_str, data_dir)

    cached_fp = _file_cached(cache_paths)
    if cached_fp:
        df_cached = _read_cached_data(cached_fp)
        if do_transform:
            df_cached = transform_taq_nbbo(df_cached)
        return pl.from_pandas(df_cached) if use_polars else df_cached

    # Build SQL
    # Example table: taqm_{year_str}.complete_nbbo_{date_str}
    # Adjust "db" usage if you have a global wrds.Connection or create a new one.
    sql = f"""
        WITH windowable_nbbo AS (
            SELECT
                sym_root AS ticker
                , date
                , time_m
                , time_m_nano
                , sym_root
                , qu_cond
                , best_bid
                , best_bidsizeshares
                , best_ask
                , best_asksizeshares
                , EXTRACT(HOUR FROM time_m) AS hour_of_day
                , {bar_minutes} * DIV(EXTRACT(MINUTE FROM time_m), {bar_minutes}) AS minute_of_hour
                , ROW_NUMBER() OVER (
                    PARTITION BY sym_root, EXTRACT(HOUR FROM time_m),
                                 DIV(EXTRACT(MINUTE FROM time_m), {bar_minutes})
                    ORDER BY time_m DESC
                  ) AS rownum
            FROM taqm_{year_str}.complete_nbbo_{date_str}
            WHERE 1=1
              AND sym_root IN {tickers_tuple}
              AND sym_suffix IS NULL
              AND time_m > '09:30:00' 
              AND time_m < '16:00:00'
        )
        SELECT DISTINCT ON (ticker, date, hour_of_day, minute_of_hour)
            ticker
            , date
            , date
              + (hour_of_day || ':' || minute_of_hour)::interval
              + ( '00:{bar_minutes}' )::interval AS window_time
            , best_bid
            , best_bidsizeshares
            , best_ask
            , best_asksizeshares
            , time_m AS time_of_last_quote
            , time_m_nano AS time_of_last_quote_ns
        FROM windowable_nbbo
        WHERE rownum = 1
    """
    logging.info(f"[get_taq_nbbo] Running SQL for date={date_str}, tickers={tickers_tuple} ...")

    # Query WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(sql)
    db.close()

    # Write to cache if not empty
    if not df.empty:
        csv_path = next((p for p in cache_paths if p.suffix == ".csv"), None)
        if csv_path:
            write_cache_data(df, csv_path, fmt="csv")

    # Transform if requested
    if do_transform and not df.empty:
        df = transform_taq_nbbo(df)

    return pl.from_pandas(df) if use_polars else df


# --------------------------------------------------------------------------
#  WCT Data Function
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_taq_wct(
    tickers: Union[str, List[str]],
    date: Union[str, datetime.date],
    use_polars: bool = True,
    wrds_username: str = WRDS_USERNAME,
    hash_file_name: bool = False,
    data_dir: Union[None, Path] = RAW_DATA_DIR,
    do_transform: bool = False
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Retrieve TAQ WRDS Consolidated Trades (WCT) data for the given date & tickers.
    Uses local-file caching plus an LRU in-memory cache.

    Parameters
    ----------
    tickers : str or list of str
        One or more ticker symbols. e.g. 'AAPL' or ['AAPL','MSFT'].
    date : str or datetime.date
        The trading date (e.g. '2023-01-03'). If string with '-' it is parsed.
    use_polars : bool, optional
        If True, returns a Polars DataFrame; otherwise Pandas.
    wrds_username : str, optional
        Your WRDS username. Defaults to Config.WRDS_USERNAME.
    hash_file_name : bool, optional
        If True, uses hashed filename for cache. Otherwise uses a verbose name.
    data_dir : Path or None, optional
        Local directory for caching. Defaults to Config.RAW_DATA_DIR.
    do_transform : bool, optional
        If True, run transform_taq_wct(). Default False.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        The WCT dataset. By default, returns Polars DataFrame if use_polars=True.

    Notes
    -----
    - The table naming here is purely an example (taqm_{year}.wct_{YYYYMMDD});
      adjust to your actual WCT location.
    """
    # Parse date
    if isinstance(date, datetime.date):
        date_dt = date
    else:
        if "-" in date:
            date_dt = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        else:
            date_dt = datetime.datetime.strptime(date, "%Y%m%d").date()

    date_str = date_dt.strftime('%Y%m%d')
    year_str = date_dt.strftime('%Y')

    # Force tickers into a tuple
    if isinstance(tickers, str):
        tickers_tuple = (tickers,)
    else:
        tickers_tuple = tuple(tickers)

    # Build filter/caching key
    filters = {
        "tickers": tickers_tuple,
        "date": date_str
    }
    filter_str = _flatten_dict_to_str(filters)

    # Check file cache
    if hash_file_name:
        cache_paths = _hash_cache_filename("taq_wct", filter_str, data_dir)
    else:
        cache_paths = _cache_filename("taq_wct", filter_str, data_dir)

    cached_fp = _file_cached(cache_paths)
    if cached_fp:
        df_cached = _read_cached_data(cached_fp)
        if do_transform:
            df_cached = transform_taq_wct(df_cached)
        return pl.from_pandas(df_cached) if use_polars else df_cached

    # Example WCT table name
    sql = f"""
        SELECT *
        FROM taqm_{year_str}.wct_{date_str}
        WHERE sym_root IN {tickers_tuple}
          AND sym_suffix IS NULL
    """
    logging.info(f"[get_taq_wct] Running SQL for date={date_str}, tickers={tickers_tuple} ...")

    # Query WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(sql)
    db.close()

    # Cache if not empty
    if not df.empty:
        csv_path = next((p for p in cache_paths if p.suffix == ".csv"), None)
        if csv_path:
            write_cache_data(df, csv_path, fmt="csv")

    # Transform if requested
    if do_transform and not df.empty:
        df = transform_taq_wct(df)

    return pl.from_pandas(df) if use_polars else df