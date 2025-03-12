"""
transform_taq.py

Transform data pulled from the WRDS Trade and Quote (TAQ) database .

This module provides functions to transform National Best Bid & Offer (NBBO) and
WRDS Consolidated Trades (WCT) data from TAQ. 

For more information about WRDS TAQ, see:
https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/taq/general/wrds-overview-taq/
"""

import pandas as pd
import numpy as np
import math


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

def calc_duration_since_last_trade(df: pd.DataFrame,
                                   time_col: str = "time_trade") -> pd.DataFrame:
    """
    Calculate the time duration since the last trade (in seconds).
    """
    df = df.sort_values(time_col).copy()
    df["duration_since_last_trade"] = df[time_col].diff().dt.total_seconds()
    return df


def calc_mid_price(df: pd.DataFrame,
                   bid_col: str = "best_bid",
                   ask_col: str = "best_ask") -> pd.DataFrame:
    """
    Calculate the mid price = (bid + ask)/2.
    """
    df = df.copy()
    df["mid_price"] = (df[bid_col] + df[ask_col]) / 2
    return df


def calc_order_weighted_price(df: pd.DataFrame,
                              bid_col: str = "best_bid",
                              ask_col: str = "best_ask",
                              bid_size_col: str = "best_bidsizeshares",
                              ask_size_col: str = "best_asksizeshares",
                              out_col: str = "owa_price") -> pd.DataFrame:
    """
    Weighted average of best bid and ask using sqrt of sizes:
    OWA = ( sqrt(ask_size)/(sqrt(ask_size) + sqrt(bid_size))*bid
            + sqrt(bid_size)/(sqrt(ask_size) + sqrt(bid_size))*ask )
    """
    df = df.copy()
    # Avoid dividing by zero:
    denom = np.sqrt(df[ask_size_col]) + np.sqrt(df[bid_size_col])
    mask = denom > 0
    df[out_col] = np.nan
    df.loc[mask, out_col] = (
          (np.sqrt(df.loc[mask, ask_size_col]) / denom[mask]) * df.loc[mask, bid_col]
        + (np.sqrt(df.loc[mask, bid_size_col]) / denom[mask]) * df.loc[mask, ask_col]
    )
    return df


def calc_vwap(
    df: pd.DataFrame,
    time_col: str = "time_trade",
    price_col: str = "price",
    size_col: str = "size",
    past_window: str = "1s",
    future_window: str = "1s",
    out_past_col: str = "past_vwap",
    out_future_col: str = "future_vwap"
) -> pd.DataFrame:
    """
    For each trade row, computes:
      1) past_vwap over the preceding `past_window`
      2) future_vwap over the next `future_window`

    Uses time-based rolling:
      - Sorts data by `time_col`, sets as index
      - For "past_vwap": sums (price*size) and size over the *past_window*, 
        ratio => VWAP.
      - For "future_vwap": reverses the DF in time, applies the same logic 
        over *future_window*, then reverses back to align results.

    Parameters
    ----------
    df : pd.DataFrame
        Must include time_col, price_col, size_col
    time_col : str
        Name of the timestamp column (pandas datetime type recommended).
    price_col : str
        Name of the trade price column.
    size_col : str
        Name of the trade size (volume) column.
    past_window : str
        Pandas time offset (e.g. '1s', '5T' for 5 minutes, etc.) for look-back window.
    future_window : str
        Pandas time offset for look-ahead window.
    out_past_col : str
        Name of the output column for past-window VWAP.
    out_future_col : str
        Name of the output column for future-window VWAP.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with two extra columns: 
          [out_past_col, out_future_col].
    """
    # 1) Make a copy and sort by time ascending
    df = df.copy()
    df.sort_values(time_col, inplace=True)

    # 2) Set index to time_col for rolling
    df.set_index(time_col, inplace=True)

    # 3) Past VWAP: rolling over `past_window` looking *back* in time
    #    sum(price * size) / sum(size)
    # We'll create temporary columns for the sums, then take ratio.
    df["past_notional"] = (df[price_col] * df[size_col]).rolling(past_window).sum()
    df["past_volume"]   = df[size_col].rolling(past_window).sum()
    df[out_past_col]    = df["past_notional"] / df["past_volume"]

    # 4) Future VWAP: we reverse in time, do the same rolling, and reverse back.
    df_rev = df.iloc[::-1].copy()  # reverse
    # Now from the reversed perspective, a rolling window of `future_window`
    # is effectively the next future_window in forward time.
    df_rev["future_notional"] = (df_rev[price_col] * df_rev[size_col]).rolling(future_window).sum()
    df_rev["future_volume"] = df_rev[size_col].rolling(future_window).sum()
    df_rev[out_future_col] = df_rev["future_notional"] / df_rev["future_volume"]

    # Reverse back
    df_rev = df_rev.iloc[::-1]

    # Merge future vwap from df_rev into df (they share the same index):
    df[out_future_col] = df_rev[out_future_col]

    # 5) Clean up
    df.drop(columns=["past_notional", "past_volume"], inplace=True)
    df_rev.drop(columns=["future_notional", "future_volume", out_future_col], inplace=True)

    # 6) Reset index so that `time_col` is again a normal column
    df.reset_index(inplace=True)

    return df


def calc_ewma_price_returns(
    df: pd.DataFrame,
    time_col: str = "time_trade",
    price_col: str = "price",
    mid_price_col: str = "mid_price",
    owa_price_col: str = "owa_price",
    half_life_s: float = 20.0
) -> pd.DataFrame:
    """
    Computes a time-based EWMA (half-life = half_life_s seconds) separately for:
      1) price_col,
      2) mid_price_col,
      3) owa_price_col

    Then for each, stores the log-return of actual value vs. EWMA in columns:
      - ewma_price_return
      - ewma_mid_price_return
      - ewma_owa_price_return

    The formula for each series s[i] is:
      EWMA[i] = EWMA[i-1]*exp(-alpha * dt) + s[i]*(1 - exp(-alpha*dt))
    where alpha = ln(2) / half_life_s, 
          dt = (time[i] - time[i-1]) in seconds (via np.timedelta64).

    Then log-return = log( s[i] / EWMA[i] ).

    Returns:
        A new DataFrame with columns:
            ["ewma_price_return", "ewma_mid_price_return", "ewma_owa_price_return"]
    """
    df = df.copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    alpha = math.log(2) / half_life_s
    times = df[time_col].values

    # Prepare arrays for each input series
    arr_price    = df[price_col].values
    arr_mid      = df[mid_price_col].values
    arr_owa      = df[owa_price_col].values

    # Initialize EWMA arrays
    ewma_price_vals = np.zeros(len(df), dtype=float)
    ewma_mid_vals   = np.zeros(len(df), dtype=float)
    ewma_owa_vals   = np.zeros(len(df), dtype=float)

    # Start them off with the first row's value
    ewma_price_vals[0] = arr_price[0]
    ewma_mid_vals[0]   = arr_mid[0]
    ewma_owa_vals[0]   = arr_owa[0]

    for i in range(1, len(df)):
        # dt in seconds
        dt_seconds = (times[i] - times[i-1]) / np.timedelta64(1, 's')
        if dt_seconds < 0:
            dt_seconds = 0.0

        decay = math.exp(-alpha * dt_seconds)

        # Update each EWMA
        ewma_price_vals[i] = ewma_price_vals[i-1] * decay + arr_price[i] * (1 - decay)
        ewma_mid_vals[i]   = ewma_mid_vals[i-1]   * decay + arr_mid[i]   * (1 - decay)
        ewma_owa_vals[i]   = ewma_owa_vals[i-1]   * decay + arr_owa[i]   * (1 - decay)

    # Now define the log-return columns
    df["ewma_price_return"]      = np.log(arr_price / ewma_price_vals)
    df["ewma_mid_price_return"]  = np.log(arr_mid   / ewma_mid_vals)
    df["ewma_owa_price_return"]  = np.log(arr_owa   / ewma_owa_vals)

    return df


def calc_spread(df: pd.DataFrame,
                bid_col: str = "best_bid",
                ask_col: str = "best_ask",
                out_col: str = "spread") -> pd.DataFrame:
    """
    Spread = best_ask - best_bid
    """
    df = df.copy()
    df[out_col] = df[ask_col] - df[bid_col]
    return df


def calc_spread_features(
    df: pd.DataFrame,
    time_col: str = "time_trade",
    spread_col: str = "spread",
    window: str = "1s",
    out_spread_mean: str = "spread_mean",
    out_spread_cv: str = "spread_CV",
    out_spread_zscore: str = "spread_Zscore"
) -> pd.DataFrame:
    """
    Computes a time-based rolling mean & std dev of `spread_col` 
    over the past 1s (or another time-based window).
    
    1. Sort by `time_col`.
    2. Set as index.
    3. `.rolling(window=window).mean()` and `.std()`.
    4. Reset index. 
    5. Return new columns: out_spread_mean, out_spread_std.
    
    The result is that for each trade i, the new columns represent the
    average and stdev of the spread for trades in the preceding 'window' interval.
    """
    df = df.copy()
    df = df.sort_values(time_col).set_index(time_col)

    df[out_spread_mean] = df[spread_col].rolling(window).mean()
    df["out_spread_std"]  = df[spread_col].rolling(window).std()
    df[out_spread_cv] = df["out_spread_std"] / df["spread_mean"]
    df[out_spread_zscore] = (df[spread_col] - df["spread_mean"]) / df["out_spread_std"]

    # Drop the temporary columns
    df.drop(columns=["out_spread_std"], inplace=True)

    # Reset index so we keep time as a column
    df = df.reset_index()
    return df


def calc_trade_direction_sign(df: pd.DataFrame,
                              price_col: str = "price",
                              bid_col: str = "best_bid",
                              ask_col: str = "best_ask",
                              mid_col: str = "mid_price",
                              out_col: str = "trade_sign") -> pd.DataFrame:
    """
    +1 for buyer-initiated, -1 for seller-initiated, else 0 if exactly mid (or no info).
    """
    df = df.copy()
    conditions = [
        (df[price_col] >= df[ask_col]),
        (df[price_col] <= df[bid_col])
    ]
    choices = [1, -1]
    df[out_col] = np.select(conditions, choices, default=0)

    # For trades strictly between bid and ask, compare to mid:
    between_mask = (df[out_col] == 0)
    df.loc[between_mask, out_col] = np.where(
        df.loc[between_mask, price_col] > df.loc[between_mask, mid_col], 
        1, 
        np.where(df.loc[between_mask, price_col] < df.loc[between_mask, mid_col], -1, 0)
    )
    return df


def calc_size_imbalance(df: pd.DataFrame,
                        bid_size_col: str = "best_bidsizeshares",
                        ask_size_col: str = "best_asksizeshares",
                        out_col: str = "size_imbalance") -> pd.DataFrame:
    """
    SI = (ask_size - bid_size) / (ask_size + bid_size)
    """
    df = df.copy()
    denom = df[bid_size_col] + df[ask_size_col]
    df[out_col] = np.where(denom > 0, (df[ask_size_col] - df[bid_size_col]) / denom, np.nan)
    return df


def calc_market_pressure(df: pd.DataFrame,
                        size_col: str = "size",
                        bid_size_col: str = "best_bidsizeshares",
                        ask_size_col: str = "best_asksizeshares",
                        out_col: str = "market_pressure") -> pd.DataFrame:
    """
    MP_t = Trade Volume / (BidSize + AskSize)
    """
    df = df.copy()
    denom = df[bid_size_col] + df[ask_size_col]
    df[out_col] = np.where(denom > 0, df[size_col] / denom, np.nan)
    return df


def calc_ofi(df: pd.DataFrame,
             bid_size_col: str = "best_bidsizeshares",
             ask_size_col: str = "best_asksizeshares",
             out_col: str = "OFI") -> pd.DataFrame:
    """
    Order Flow Imbalance:
    OFI = (BidSize_t - BidSize_{t-1}) - (AskSize_t - AskSize_{t-1})
    """
    df = df.copy().sort_values("time_trade")
    df[out_col] = (
        (df[bid_size_col] - df[bid_size_col].shift(1))
        - (df[ask_size_col] - df[ask_size_col].shift(1))
    )
    return df


def calc_trade_flow(df: pd.DataFrame,
                    sign_col: str = "trade_sign",
                    size_col: str = "size",
                    time_step: str = "100ms",  # e.g. 1 millisecond
                    window: str = "1s",  # e.g. 1 second rolling
                    out_col: str = "trade_flow") -> pd.DataFrame:
    """
    Signed volume (trade_sign * size) aggregated over a time-based rolling window.
    This uses a resample or rolling on time, then merges back.
    """
    df = df.copy().sort_values("time_trade")
    df["signed_volume"] = df[sign_col] * df[size_col]

    # A straightforward approach: resample trades in time, sum signed_volume, and asof merge back
    # 1) set index to time, 2) resample, 3) rolling sum, 4) merge_asof
    temp = df.set_index("time_trade")
    # A rolling approach in Pandas that is strictly forward in time requires
    # some trick.  We can do an expanding sum minus a shifted sum, or any method.
    # For simplicity, here is a resample approach:
    resampled = temp["signed_volume"].resample(time_step).sum().fillna(0.0).to_frame()  
    # now do a rolling sum over 'window'
    resampled[out_col] = resampled.rolling(window=window).sum()

    # merge back on second-by-second index, then asof to trades
    resampled = resampled.drop(columns=["signed_volume"]).reset_index()
    df = pd.merge_asof(df.reset_index(drop=True).sort_values("time_trade"),
                       resampled.sort_values("time_trade"),
                       on="time_trade", direction="backward")
    return df


def calc_order_imbalance_and_vpin(
    df: pd.DataFrame,
    sign_col: str = "trade_sign",
    size_col: str = "size",
    time_col: str = "time_trade",
    bucket_size: float = 1000,
    rolling_buckets_for_vpin: int = 20
) -> pd.DataFrame:
    """
    Implements the fixed-volume bucket approach to compute:
       Iₙ = |Vₙᵇ - Vₙˢ| / (Vₙᵇ + Vₙˢ)
    and then VPIN = rolling average of Iₙ over the last N buckets.
    
    Returns the original DF with columns:
        [bucket_id, I_n, VPIN, ...]
    assigned to each trade row via an asof merge.
    
    Steps:
      1) Sort trades by time.
      2) Accumulate trades into volume buckets of size `bucket_size`.
      3) For each bucket, compute Iₙ.  Keep track of (start_time, end_time).
      4) Compute a rolling mean of Iₙ => VPIN across last `rolling_buckets_for_vpin` buckets.
      5) Merge back onto each trade row so we know which bucket it fell into, 
         and the corresponding Iₙ and VPIN.
    """
    df = df.sort_values(time_col).copy()
    
    buyer_vol = 0.0
    seller_vol = 0.0
    total_vol = 0.0
    bucket_id = 0
    start_time = None

    # We'll store a row in bucket_list once each bucket is complete
    bucket_list = []  # each entry = [bucket_id, start_time, end_time, I_n]

    for idx, row in df.iterrows():
        if start_time is None:
            start_time = row[time_col]

        # Accumulate volume
        if row[sign_col] > 0:
            buyer_vol += row[size_col]
        elif row[sign_col] < 0:
            seller_vol += row[size_col]
        else:
            # If trade_sign=0, treat as neither or split volume, etc.
            # For simplicity, ignore or treat as seller. Here we do nothing
            pass

        total_vol = buyer_vol + seller_vol

        # Once we meet or exceed the bucket_size threshold => finalize bucket
        if total_vol >= bucket_size:
            end_time = row[time_col]
            I_n = abs(buyer_vol - seller_vol) / total_vol if total_vol > 0 else np.nan
            bucket_list.append([bucket_id, start_time, end_time, I_n])
            # reset
            bucket_id += 1
            buyer_vol = 0.0
            seller_vol = 0.0
            total_vol = 0.0
            start_time = None

    # Convert the bucket_list to a DataFrame
    buckets_df = pd.DataFrame(bucket_list, columns=["bucket_id","start_time","end_time","I_n"])

    # Now compute VPIN = rolling mean of I_n over last N buckets
    buckets_df["VPIN"] = buckets_df["I_n"].rolling(rolling_buckets_for_vpin).mean()

    # We want to attach each trade row to the bucket that just ended at or before that row's time.
    # But because we only know "end_time" for each bucket, we can do a merge_asof
    # matching each trade's time to the "end_time" of the last-completed bucket.
    # That means each trade that came after the bucket closed will get that bucket's I_n, VPIN, etc.
    # If you prefer the trade *within* the bucket to get the bucket_id, you might do a different approach
    # that merges on the (start_time, end_time) interval.  Here is a typical approach:
    df = pd.merge_asof(
        df.sort_values(time_col),
        buckets_df.sort_values("end_time"),
        left_on=time_col,
        right_on="end_time",
        direction="backward"
    )

    # Now each trade has columns: [bucket_id, I_n, VPIN, ...].
    return df


# ------------------------------------------------------------------------------
# Finally, our main “driver” function that calls each subfunction in sequence:
# ------------------------------------------------------------------------------
def extract_features(
    original_df: pd.DataFrame,
    half_life_s: float = 20.0,
    realized_vol_window: str = "1s",
    spread_window: str = "1s",
    flow_time_step: str = "100ms",
    flow_window: str = "1s",
    bucket_size: float = 1000,
    vpin_buckets: int = 20,
    vwap_past_window: str = "3s",
    vwap_future_window: str = "1s",
) -> pd.DataFrame:
    """
    Master function that applies all microstructure feature calculations.
    Returns a NEW DataFrame with the original columns plus the features.
    """
    # Start with a copy so we don’t mutate original_df
    df = original_df.copy()
    # 1) Sort by time to ensure correct diffs, rolling, etc.
    df = df.sort_values("time_trade")

    # 2) Duration since last trade
    df = calc_duration_since_last_trade(df, time_col="time_trade")

    # 3) Mid Price
    df = calc_mid_price(df)

    # 4) Order-Weighted Average Price
    df = calc_order_weighted_price(df)

    # 5) VWAP
    df = calc_vwap(df,
                   past_window=vwap_past_window,     # look-back 5 seconds
                   future_window=vwap_future_window,   # look-ahead 3 seconds
                   out_past_col="past_vwap",
                   out_future_col="future_vwap"
    )

    # 6) EWMA Price Returns
    df = calc_ewma_price_returns(df,
                                 half_life_s=half_life_s,
                                 price_col="price",
                                 mid_price_col="mid_price",
                                 owa_price_col="owa_price")
    
    # 7) Spread
    df = calc_spread(df)

    # 8) Rolling Spread Mean & Volatility
    df = calc_spread_features(df, window=spread_window)

    # 9) Trade Direction (sign)
    df = calc_trade_direction_sign(df)

    # 10) Size Imbalance
    df = calc_size_imbalance(df)

    # 11) Market Pressure
    df = calc_market_pressure(df)

    # 12) OFI
    df = calc_ofi(df)

    # 13) Trade Flow (time-based rolling or resample)
    df = calc_trade_flow(df, time_step=flow_time_step, window=flow_window)

    # 14) Order Imbalance & VPIN (fixed-volume bucket)
    df = calc_order_imbalance_and_vpin(df,
                                       bucket_size=bucket_size,
                                       rolling_buckets_for_vpin=vpin_buckets)

    df = df[[
        "time_trade", "price", "mid_price", "owa_price", 
        "ewma_price_return", "ewma_mid_price_return", "ewma_owa_price_return",
        "past_vwap", "future_vwap","trade_sign", "duration_since_last_trade", 
        "size_imbalance", "market_pressure", "OFI", "trade_flow", 
        "spread_mean", "spread_CV", "spread_Zscore",
        ]]
    
    return df


    





