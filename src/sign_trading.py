import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from settings import config
from pull_taq import get_taq_nbbo, get_taq_wct
from transform_taq import extract_features_taq
from strategy import (
    create_labels,
    train_ml_model_pipeline,
    signal_to_returns
)

def parse_wrds_time(s):
    # Remove any extra whitespace
    s = s.strip()
    try:
        # Try parsing with microseconds
        return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f%z")
    except ValueError:
        # Fall back to parsing without microseconds
        return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S%z")

def run_intraday_hft_pipeline_with_plots(
    ticker: str,
    trade_date: str,
    use_bars: bool = False,
    model_name: str = "random_forest_hft_signal",
    buy_threshold_std: float = 1.0,
    sell_threshold_std: float = 1.0,
    transaction_cost: float = 0.0001,
    use_file: bool = True,                      # New parameter: use file-based data if True
    file_path: str = "data/merged_trades.csv"   # Path to the pre-merged CSV file
):
    """
    End-to-end intraday pipeline for a single ticker & date, with plots saved
    to data/results/<model_name>.

    Depending on the use_file flag, the pipeline either:
      - Loads pre-merged trades data from a CSV file (no WRDS authentication required), or
      - Pulls NBBO quotes & trades from WRDS, merging them as-of.
    
    1) Pulls/Loads NBBO quotes & trades
    2) Merges them (as-of)
    3) Extracts features (microstructure)
    4) Creates classification labels
    5) Trains or loads a random forest model to predict signals
    6) Generates signals & computes strategy returns
    7) Saves basic plots to data/results/<model_name>/

    Parameters
    ----------
    ticker : str
        Ticker symbol to process (e.g. 'SPY', 'AAPL', etc.)
    trade_date : str
        Date in 'YYYY-MM-DD' or 'YYYYMMDD' format (e.g. '2024-03-07')
    use_bars : bool
        Whether to request bar-aggregated NBBO data or raw quotes (default raw)
    model_name : str
        Name used for saving/loading the model and results folder
    buy_threshold_std : float
        Threshold above mean future return for +1 signal
    sell_threshold_std : float
        Threshold below mean future return for -1 signal
    transaction_cost : float
        Per-trade cost, e.g. 0.0001 = 1 basis point
    use_file : bool
        If True, load merged trades data from a CSV file instead of pulling from WRDS.
    file_path : str
        Path to the CSV file containing merged trades data.

    Returns
    -------
    feats_df, signals_df, strategy_ret, final_cum_return
    """

    # ----------------------------------------------------------------------------
    # Create the output directory for plots
    # ----------------------------------------------------------------------------
    results_dir = Path("data") / "results" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if use_file:
        # ----------------------------------------------------------------------------
        # Load merged trades data from file
        # ----------------------------------------------------------------------------
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File {file_path} not found.")
        print(f"Loading merged trades data from {file_path}.")
        merged_trades = pd.read_csv(file_path_obj, parse_dates=["time_trade", "date"])
        # Convert the 'time_trade' column without specifying an exact format.

        merged_trades["time_trade"] = merged_trades["time_trade"].apply(parse_wrds_time)

        # Remove timezone info to match WRDS's datetime64[ns] type
        merged_trades["time_trade"] = merged_trades["time_trade"].dt.tz_localize(None)
        print(merged_trades.head())

    else:
        # ----------------------------------------------------------------------------
        # 1) Pull quotes & trades from WRDS
        # ----------------------------------------------------------------------------
        quotes = get_taq_nbbo((ticker,), date=trade_date, use_bars=use_bars)
        trades = get_taq_wct((ticker,), date=trade_date)
        if quotes.empty or trades.empty:
            raise ValueError(f"No TAQ data returned for {ticker} on {trade_date}.")

        # ----------------------------------------------------------------------------
        # 2) Merge trades onto quotes (as-of)
        # ----------------------------------------------------------------------------
        merged_trades = pd.merge_asof(
            trades.sort_values("time_trade"),
            quotes.sort_values("time_quote")[["time_quote",
                                              "best_bid",
                                              "best_bidsizeshares",
                                              "best_ask",
                                              "best_asksizeshares"]],
            left_on="time_trade",
            right_on="time_quote",
            direction="backward"
        )
        merged_trades.drop(columns="time_quote", inplace=True)

    # ----------------------------------------------------------------------------
    # 3) Extract features
    # ----------------------------------------------------------------------------
    feats_df = extract_features_taq(
        merged_trades,
        half_life_s=60,
        spread_window="60s",
        flow_time_step="100ms",
        flow_window="5s",
        bucket_size=1000,
        vpin_buckets=30,
        vwap_past_window="60s",
        vwap_future_window="20s"
    )
    feats_df.dropna(inplace=True)

    best_price_df = feats_df.set_index("time_trade")[["best_bid", "best_ask"]]

    # ----------------------------------------------------------------------------
    # 4) Create classification labels
    # ----------------------------------------------------------------------------
    LABEL_COL = "ml_signal"
    feats_df = create_labels(
        feats_df,
        price_col="price",
        future_vwap_col="future_vwap",
        out_return_col="future_return",
        out_label_col=LABEL_COL,
        buy_threshold_std=buy_threshold_std,
        sell_threshold_std=sell_threshold_std,
    )

    # Subset to relevant features + label
    feats_df = feats_df[[
        "ewma_mid_price_return",
        "trade_sign",
        "duration_since_last_trade",
        "size_imbalance",
        "market_pressure",
        "OFI",
        "trade_flow",
        "spread_mean",
        "spread_CV",
        "spread_Zscore",
        LABEL_COL
    ]]


    # ----------------------------------------------------------------------------
    # 5) Train model or load existing one
    # ----------------------------------------------------------------------------
    OUTPUT_DIR = Path(config("OUTPUT_DIR"))
    model_path = OUTPUT_DIR / "models" / f"{model_name}.pkl"

    if model_path.exists():
        print(f"[{model_name}] Found existing model at {model_path}. Loading it.")
        ml_model = joblib.load(model_path)
    else:
        print(f"[{model_name}] Model not found at {model_path}. Training a new one...")
        ml_model = train_ml_model_pipeline(
            data=feats_df,
            label_col=LABEL_COL,
            use_pca=True,
            pca_components=0.90,
            train_size=0.8,
            model_name=model_name,
            model_path=str(OUTPUT_DIR / "models")
        )

    # ----------------------------------------------------------------------------
    # 6) Generate signals
    # ----------------------------------------------------------------------------
    X_for_prediction = feats_df.drop(columns=[LABEL_COL], errors="ignore").dropna()
    signal_array = ml_model.predict(X_for_prediction)
    signals_df = pd.DataFrame(signal_array, index=X_for_prediction.index, columns=["signals"])

    # ----------------------------------------------------------------------------
    # 7) Convert signals to realized returns
    # ----------------------------------------------------------------------------

    signals_df.index = best_price_df.index

    signals_df = signals_df.reindex(best_price_df.index, method='ffill').fillna(0)

    filtered_signals_df = signals_df[signals_df["signals"] != signals_df["signals"].shift()]


    strategy_ret = signal_to_returns(
        signals=filtered_signals_df,
        best_prices=best_price_df,
        strategy_name="ML_strategy",
        lookahead_timedelta=pd.Timedelta("1ms"),
        up_weight=1.0,
        down_weight=1.0,
        transaction_cost=transaction_cost
    )

    cum_ret = (1.0 + strategy_ret["ML_strategy_returns"]).cumprod() - 1.0
    final_ret = cum_ret.iloc[-1] if not cum_ret.empty else 0.0

    # ----------------------------------------------------------------------------
    # 8) Print & Plot results
    # ----------------------------------------------------------------------------
    print("-------------------------------------------")
    print(f"Ticker: {ticker}")
    print(f"Date:   {trade_date}")
    print(f"Model:  {model_name}")
    print(f"Final strategy cumulative return =  {final_ret:.4f}")
    mean_trade_ret = strategy_ret["ML_strategy_returns"].mean()
    std_trade_ret  = strategy_ret["ML_strategy_returns"].std()
    print(f"Average trade return           =  {mean_trade_ret:.6f}")
    print(f"Std of trade returns           =  {std_trade_ret:.6f}")
    print("-------------------------------------------")

    # ========== PLOT 1: Cumulative Return Time Series ==========
    plt.figure()
    plt.plot(cum_ret.index, cum_ret.values)
    plt.title(f"{model_name}: Cumulative Return")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_cumulative_return.png")
    plt.close()

    # ========== PLOT 2: Distribution of Strategy Returns ==========
    plt.figure()
    plt.hist(strategy_ret["ML_strategy_returns"], bins=30)
    plt.title(f"{model_name}: Distribution of Strategy Returns")
    plt.xlabel("Return per Trade")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_returns_histogram.png")
    plt.close()

    # ========== PLOT 3: Distribution of Predicted Signals ==========
    signals_count = signals_df["signals"].value_counts()
    plt.figure()
    plt.bar(signals_count.index.astype(str), signals_count.values)
    plt.title(f"{model_name}: Distribution of Signals")
    plt.xlabel("Signal Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_signals_distribution.png")
    plt.close()

    print(f"Plots saved in: {results_dir.resolve()}")

    return feats_df, signals_df, strategy_ret, final_ret

# Example usage:
if __name__ == "__main__":
    # Set use_file=True to load data from a CSV file (no WRDS authentication needed)
    run_intraday_hft_pipeline_with_plots(
        ticker="SPY",
        trade_date="2024-03-07",
        use_file=True,
        file_path="./data/processed/merged_trades.csv"
    )
