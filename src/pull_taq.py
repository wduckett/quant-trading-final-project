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
from functools import lru_cache
from pathlib import Path
from typing import Union, List, Any, Dict

import pandas as pd
import polars as pl
import pytz
import wrds

# Local imports - adjust to your package structure
from settings import config
from utils import (
    _cache_filename,
    _hash_cache_filename,
    _file_cached,
    _read_cached_data,
    _flatten_dict_to_str,
    _write_cache_data,
    _tickers_to_tuple,
    _format_tuple_for_sql_list,
)
from transform_taq import transform_nbbo, transform_taq_wct

# =============================================================================
# Global Configuration
# =============================================================================

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
WRDS_USERNAME = Path(config("WRDS_USERNAME"))

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

    # Call the cached function with the hashable tickers tuple.
    return _get_taq_nbbo_cached(
        tickers,
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

    # Convert tickers to a tuple if not already one.
    tickers_tuple = _tickers_to_tuple(tickers)

    # Ensure bar_minutes is valid
    assert bar_minutes == 60 or (bar_minutes <= 30 and 30 % bar_minutes == 0), \
        "bar_minutes must be 60 or a divisor of 30"

    # Build a filter string for caching
    filters: Dict[str, Any] = {
        "tickers": tickers_tuple,
        "date": date_str
    }
    if use_bars:
        filters["bar_minutes"] = bar_minutes

    filter_str = _flatten_dict_to_str(filters)

    # Check file cache
    if hash_file_name:
        cache_paths = _hash_cache_filename("taq_nbbo", filter_str, data_dir)
    else:
        cache_paths = _cache_filename("taq_nbbo", filter_str, data_dir)

    cached_fp = _file_cached(cache_paths)
    if cached_fp:
        df_cached = _read_cached_data(cached_fp)
        return pl.from_pandas(df_cached) if use_polars else df_cached

    # Build SQL
    # Example table: taqm_{year_str}.complete_nbbo_{date_str}
    # Adjust "db" usage if you have a global wrds.Connection or create a new one.
    ticker_filter = _format_tuple_for_sql_list(tickers_tuple)
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

    # Query WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(sql)
    db.close()

    # Write to cache if not empty
    if not df.empty:
        df = transform_nbbo(df)
        file_path = next((p for p in cache_paths if p.suffix == ".parquet"), None)
        if file_path:
            _write_cache_data(df, file_path)

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
    
    return _get_taq_wct_cached(
        tickers,
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
    data_dir: Union[None, Path] = RAW_DATA_DIR
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

    # Convert tickers to a tuple if not already one.
    tickers_tuple = _tickers_to_tuple(tickers)

    # Build a filter string for caching
    filters: Dict[str, Any] = {
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
        return pl.from_pandas(df_cached) if use_polars else df_cached

    # Build SQL for WCT data. Adjust table and column names as needed.
    ticker_filter = _format_tuple_for_sql_list(tickers_tuple)
    sql = f"""
        SELECT
            sym_root AS ticker,
            date,
            time_m,
            price,
            size
        FROM taqm_{year_str}.ctm_{date_str}
        WHERE
            sym_root IN {ticker_filter}
            AND time_m > '09:30:00'
            AND time_m < '16:00:00'
            AND tr_corr IN ('00', '01', '02') -- Exclude non-regular trades
    """

    # Query WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(sql)
    db.close()

    # Write to cache if not empty
    if not df.empty:
        df = transform_taq_wct(df)
        file_path = next((p for p in cache_paths if p.suffix == ".parquet"), None)
        if file_path:
            _write_cache_data(df, file_path)

    return pl.from_pandas(df) if use_polars else df