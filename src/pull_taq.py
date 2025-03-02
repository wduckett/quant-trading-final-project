"""
pull_taq.py

Pull data from the WRDS Trade and Quote (TAQ) database with caching support.

This module provides functions to download National Best Bid & Offer (NBBO) and
WRDS Consolidated Trades (WCT) data from TAQ. 

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

Similarly for WCT data:
    df_wct = get_taq_wct(
        date='2023-01-03',
        ticker=['AAPL','MSFT'],
        wrds_username='my_wrds_username',
        use_polars=False
    )

-------------------------------------------------------------
"""

import datetime
import logging
from functools import lru_cache
from pathlib import Path
from typing import Union, List, Any, Dict

import pandas as pd
import polars as pl
import wrds

# Local imports - adjust to your package structure
from settings import config
from utils import (
    _cache_filename,
    _hash_cache_filename,
    _file_cached,
    _read_cached_data,
    write_cache_data,
    _flatten_dict_to_str
)


RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
WRDS_USERNAME = Path(config("WRDS_USERNAME"))


# --------------------------------------------------------------------------
#  Helper Functions
# --------------------------------------------------------------------------

def format_tickers_for_sql(tickers: tuple) -> str:
    """
    Convert a tuple of tickers into a properly formatted SQL list.
    """
    tickers_list = ", ".join(f"'{ticker}'" for ticker in tickers)
    return f"({tickers_list})"


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
#  NBBO Data Function
# --------------------------------------------------------------------------

def get_taq_nbbo(
    tickers: Union[str, List[str], tuple],
    date: Union[str, datetime.date],
    use_bars: bool = True,
    bar_minutes: int = 30,
    use_polars: bool = False,
    wrds_username: str = WRDS_USERNAME,
    hash_file_name: bool = False,
    data_dir: Union[None, Path] = RAW_DATA_DIR
) -> Union[pd.DataFrame, pl.DataFrame]:
    # Convert tickers to a tuple if not already one.
    if isinstance(tickers, str):
        tickers_tuple = (tickers,)
    elif isinstance(tickers, list):
        tickers_tuple = tuple(tickers)
    else:
        tickers_tuple = tickers

    # Call the cached function with the hashable tickers tuple.
    return _get_taq_nbbo_cached(
        tickers_tuple,
        date,
        use_bars,
        bar_minutes,
        use_polars,
        wrds_username,
        hash_file_name,
        data_dir
    )


@lru_cache(maxsize=None)
def _get_taq_nbbo_cached(
    tickers: tuple,
    date: Union[str, datetime.date],
    use_bars: bool = True,
    bar_minutes: int = 30,
    use_polars: bool = False,
    wrds_username: str = WRDS_USERNAME,
    hash_file_name: bool = False,
    data_dir: Union[None, Path] = RAW_DATA_DIR,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Retrieve intraday NBBO data from TAQ for the given date and tickers.
    Uses local caching plus an LRU in-memory cache.

    Parameters
    ----------
    tickers : tuple
        One or more ticker symbols. e.g. ('AAPL') or ('AAPL','MSFT').
    date : str or datetime.date
        The trading date (e.g. '2023-01-03'). If string with '-' it is parsed.
    use_bars: bool,
        If True, returns x-min bars; otherwise returns raw quotes.
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

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        The final NBBO bars. By default, a Polars DataFrame (if use_polars=True).

    Notes
    -----
    - If you need monthly vs. daily logic, rename the table in the SQL as appropriate
      (e.g. taqm_{year}.complete_nbbo_{YYYYMMDD}, or wrds_taqs_nbbo.nbbo_YYYYMMDD, etc.).
    - More information about WRDS TAQ NBBO can be found at:
        https://wrds-www.wharton.upenn.edu/pages/get-data/nyse-trade-and-quote/millisecond-trade-and-quote-daily-product-2003-present-updated-daily/consolidated-quotes/
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

    # Build a filter string for caching
    filters: Dict[str, Any] = {
        "tickers": tickers,
        "date": date_str
    }
    if use_bars:
        filters["bar_minutes"] = bar_minutes

    filter_str = _flatten_dict_to_str(filters)

    # Check file cache
    if hash_file_name:
        cache_paths = _hash_cache_filename("taq_nbbo_bars", filter_str, data_dir)
    else:
        cache_paths = _cache_filename("taq_nbbo_bars", filter_str, data_dir)

    cached_fp = _file_cached(cache_paths)
    if cached_fp:
        df_cached = _read_cached_data(cached_fp)
        return pl.from_pandas(df_cached) if use_polars else df_cached

    # Build SQL
    # Example table: taqm_{year_str}.complete_nbbo_{date_str}
    # Adjust "db" usage if you have a global wrds.Connection or create a new one.
    ticker_filter = format_tickers_for_sql(tickers)
    if use_bars:
        sql = f"""
            WITH windowable_nbbo AS (
                SELECT
                    sym_root AS ticker
                    , date
                    , time_m
                    , time_m_nano
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
                WHERE
                    sym_root IN {ticker_filter}
                    AND sym_suffix IS NULL
                    AND time_m > '09:30:00' 
                    AND time_m < '16:00:00'
            )
            SELECT DISTINCT ON (ticker, date, hour_of_day, minute_of_hour)
                ticker
                , CAST(date AS DATE) AS date
                , date
                + (hour_of_day || ':' || minute_of_hour)::interval
                + ( '00:{bar_minutes}' )::interval AS window_time
                , best_bid
                , best_bidsizeshares
                , best_ask
                , best_asksizeshares
                , time_m
                , time_m_nano
            FROM windowable_nbbo
            WHERE rownum = 1
        """
    else:
        sql = f"""
            SELECT
                sym_root AS ticker
                , date
                , time_m
                , time_m_nano
                , best_bid
                , best_bidsizeshares
                , best_ask
                , best_asksizeshares
            FROM taqm_{year_str}.complete_nbbo_{date_str}
            WHERE
                sym_root IN {ticker_filter}
                AND sym_suffix IS NULL
                AND time_m > '09:30:00' 
                AND time_m < '16:00:00'
        """
    logging.info(f"[get_taq_nbbo] Running SQL for date={date_str}, tickers={tickers} ...")

    # Query WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(sql)
    db.close()

    # Write to cache if not empty
    if not df.empty:
        csv_path = next((p for p in cache_paths if p.suffix == ".csv"), None)
        if csv_path:
            write_cache_data(df, csv_path, fmt="csv")

    df = transform_datetime(df)

    return pl.from_pandas(df) if use_polars else df


# --------------------------------------------------------------------------
#  WCT Data Function
# --------------------------------------------------------------------------

def get_taq_wct(
    tickers: Union[str, List[str], tuple],
    date: Union[str, datetime.date],
    use_polars: bool = False,
    wrds_username: str = WRDS_USERNAME,
    hash_file_name: bool = False,
    data_dir: Union[None, Path] = RAW_DATA_DIR
) -> Union[pd.DataFrame, pl.DataFrame]:
    # Convert tickers to a tuple if not already one.
    if isinstance(tickers, str):
        tickers_tuple = (tickers,)
    elif isinstance(tickers, list):
        tickers_tuple = tuple(tickers)
    else:
        tickers_tuple = tickers

    return _get_taq_wct_cached(
        tickers_tuple,
        date,
        use_polars,
        wrds_username,
        hash_file_name,
        data_dir
    )

def _get_taq_wct_cached(
    tickers: tuple,
    date: Union[str, datetime.date],
    use_polars: bool = False,
    wrds_username: str = WRDS_USERNAME,
    hash_file_name: bool = False,
    data_dir: Union[None, Path] = RAW_DATA_DIR,
    do_transform: bool = True
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Retrieve intraday WCT data from TAQ for the given date and tickers.
    Uses local caching plus an LRU in-memory cache.

    Parameters
    ----------
    tickers : tuple
        One or more ticker symbols. e.g. ('AAPL') or ('AAPL','MSFT').
    date : str or datetime.date
        The trading date (e.g. '2023-01-03'). If string with '-' it is parsed.
    use_polars : bool, optional
        If True, returns a Polars DataFrame; otherwise returns Pandas.
    wrds_username : str, optional
        Your WRDS username. Defaults to Config.WRDS_USERNAME.
    hash_file_name : bool, optional
        If True, uses a hashed filename for cache. Otherwise uses a verbose name.
    data_dir : Path or None, optional
        Local directory for caching. Defaults to Config.RAW_DATA_DIR.
    do_transform : bool, optional
        Whether to run transform_taq_wct() on the result. Defaults to True.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        The final WCT bars (if use_bars=True) or raw trades (if use_bars=False).  
        By default, a Polars DataFrame (if use_polars=True).

    Notes
    -----
    - If you need monthly vs. daily logic, rename the table in the SQL as appropriate
      (e.g. taqm_{year}.complete_wct_{YYYYMMDD}, etc.).
    - More information about WRDS TAQ WCT can be found here:
        https://wrds-www.wharton.upenn.edu/pages/get-data/nyse-trade-and-quote/millisecond-trade-and-quote-daily-product-2003-present-updated-daily/consolidated-trades/
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

    # Build a filter string for caching
    filters: Dict[str, Any] = {
        "tickers": tickers,
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
        return pl.from_pandas(df_cached) if use_polars else df_cached

    # Build SQL for WCT data. Adjust table and column names as needed.
    ticker_filter = format_tickers_for_sql(tickers)
    sql = f"""
        SELECT
            sym_root AS ticker,
            date,
            time_m,
            price,
            size,
            tr_corr AS trade_correction
        FROM taqm_{year_str}.ctm_{date_str}
        WHERE
            sym_root IN {ticker_filter}
            AND time_m > '09:30:00'
            AND time_m < '16:00:00'
    """
    logging.info(f"[get_taq_wct] Running SQL for date={date_str}, tickers={tickers} ...")

    # Query WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(sql)
    db.close()

    # Write to cache if not empty
    if not df.empty:
        csv_path = next((p for p in cache_paths if p.suffix == ".csv"), None)
        if csv_path:
            write_cache_data(df, csv_path, fmt="csv")

    df = transform_datetime(df)

    return pl.from_pandas(df) if use_polars else df
