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
    use_file: bool = True,
    file_path: str = "data/merged_trades.csv",

    # -----------------------------
    # Additional microstructure parameters
    # -----------------------------
    half_life_s: float = 60,
    spread_window: str = "60s",
    flow_time_step: str = "100ms",
    flow_window: str = "5s",
    bucket_size: int = 1000,
    vpin_buckets: int = 30,
    vwap_past_window: str = "60s",
    vwap_future_window: str = "20s",

    # -----------------------------
    # ML training / model parameters
    # -----------------------------
    use_pca: bool = False,
    pca_components: float = 0.90,
    train_size: float = 0.8,

    # -----------------------------
    # Strategy execution parameters
    # -----------------------------
    lookahead_timedelta: str = "100ms",
    up_weight: float = 1.0,
    down_weight: float = 1.0
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
    half_life_s : float
        Used for exponential weighting in microstructure features (e.g., flow).
    spread_window : str
        How far back in time to compute average spread features (e.g. '60s').
    flow_time_step : str
        Time step for trade flow calculations (e.g. '100ms').
    flow_window : str
        Lookback window for aggregated trade flow (e.g. '5s').
    bucket_size : int
        Bucket size used for volume-based calculations (e.g., 1000 shares).
    vpin_buckets : int
        Number of buckets for VPIN calculation.
    vwap_past_window : str
        Lookback window for historical VWAP (e.g. '60s').
    vwap_future_window : str
        Forward window for future VWAP (e.g. '20s').
    use_pca : bool
        Whether to apply PCA for dimensionality reduction.
    pca_components : float
        Number of PCA components or fraction of variance to keep (if float < 1).
    train_size : float
        Fraction of data used for training (e.g. 0.8).
    lookahead_timedelta : str
        Time offset for realizing returns (e.g. '100ms').
    up_weight : float
        Scaling factor for long signals.
    down_weight : float
        Scaling factor for short signals.

    Returns
    -------
    feats_df, signals_df, strategy_ret, final_cum_return
    """

    # ----------------------------------------------------------------------------
    # Create the output directory for plots
    # ----------------------------------------------------------------------------
    results_dir = Path("data") / "results" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------
    # 1) Pull or load data
    # ----------------------------------------------------------------------------
    if use_file:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File {file_path} not found.")
        print(f"Loading merged trades data from {file_path}.")

        merged_trades = pd.read_csv(file_path_obj, parse_dates=["time_trade", "date"])
        # Convert the 'time_trade' column without specifying an exact format
        merged_trades["time_trade"] = merged_trades["time_trade"].apply(parse_wrds_time)
        # Remove timezone info to match WRDS's datetime64[ns] type
        merged_trades["time_trade"] = merged_trades["time_trade"].dt.tz_localize(None)

    else:
        quotes = get_taq_nbbo((ticker,), date=trade_date, use_bars=use_bars)
        trades = get_taq_wct((ticker,), date=trade_date)
        if quotes.empty or trades.empty:
            raise ValueError(f"No TAQ data returned for {ticker} on {trade_date}.")

        merged_trades = pd.merge_asof(
            trades.sort_values("time_trade"),
            quotes.sort_values("time_quote")[[
                "time_quote",
                "best_bid",
                "best_bidsizeshares",
                "best_ask",
                "best_asksizeshares"
            ]],
            left_on="time_trade",
            right_on="time_quote",
            direction="backward"
        )
        merged_trades.drop(columns="time_quote", inplace=True)

    # ----------------------------------------------------------------------------
    # 2) Extract features
    # ----------------------------------------------------------------------------
    feats_df = extract_features_taq(
        merged_trades,
        half_life_s=half_life_s,
        spread_window=spread_window,
        flow_time_step=flow_time_step,
        flow_window=flow_window,
        bucket_size=bucket_size,
        vpin_buckets=vpin_buckets,
        vwap_past_window=vwap_past_window,
        vwap_future_window=vwap_future_window
    )
    # Drop rows with any missing values in extracted features
    feats_df.dropna(inplace=True)

    # For later usage in signal generation & shading
    best_price_df = feats_df.set_index("time_trade")[["best_bid", "best_ask"]]

    # ----------------------------------------------------------------------------
    # 3) Create classification labels
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
    # 4) Train / load model
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
            use_pca=use_pca,
            pca_components=pca_components,
            train_size=train_size,
            model_name=model_name,
            model_path=str(OUTPUT_DIR / "models")
        )

    # ----------------------------------------------------------------------------
    # 5) Generate signals
    # ----------------------------------------------------------------------------
    X_for_prediction = feats_df.drop(columns=[LABEL_COL], errors="ignore").fillna(0)
    signal_array = ml_model.predict(X_for_prediction)
    signals_df = pd.DataFrame(signal_array, index=X_for_prediction.index, columns=["signals"])

    # ----------------------------------------------------------------------------
    # 6) Convert signals to realized returns
    # ----------------------------------------------------------------------------
    # Make sure signals align with best_price_df's index
    signals_df.index = best_price_df.index
    signals_df = signals_df.reindex(best_price_df.index, method='ffill').fillna(0)

    # Filter out repeated signals to only keep the changes
    filtered_signals_df = signals_df[signals_df["signals"] != signals_df["signals"].shift()]


    strategy_ret = signal_to_returns(
        signals=filtered_signals_df,
        best_prices=best_price_df,
        strategy_name="ML_strategy",
        lookahead_timedelta=pd.Timedelta(lookahead_timedelta),
        up_weight=up_weight,
        down_weight=down_weight,
        transaction_cost=transaction_cost
    )

    cum_ret = (1.0 + strategy_ret["ML_strategy_returns"]).cumprod() - 1.0
    final_ret = cum_ret.iloc[-1] if not cum_ret.empty else 0.0

    # ----------------------------------------------------------------------------
    # 7) Print & Plot results
    # ----------------------------------------------------------------------------
    # Initialize a list to store plot objects
    plots = []

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
    fig1, ax1 = plt.subplots()
    ax1.plot(cum_ret.index, cum_ret.values)
    ax1.set_title(f"{model_name}: Cumulative Return")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cumulative Return")
    fig1.tight_layout()
    fig1.savefig(results_dir / f"{model_name}_cumulative_return.png")
    plots.append(fig1)

    # ========== PLOT 2: Distribution of Strategy Returns ==========
    fig2, ax2 = plt.subplots()
    ax2.hist(strategy_ret["ML_strategy_returns"], bins=30)
    ax2.set_title(f"{model_name}: Distribution of Strategy Returns")
    ax2.set_xlabel("Return per Trade")
    ax2.set_ylabel("Frequency")
    fig2.tight_layout()
    fig2.savefig(results_dir / f"{model_name}_returns_histogram.png")
    plots.append(fig2)

    # ========== PLOT 3: Distribution of Predicted Signals ==========
    signals_count = signals_df["signals"].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.bar(signals_count.index.astype(str), signals_count.values)
    ax3.set_title(f"{model_name}: Distribution of Signals")
    ax3.set_xlabel("Signal Value")
    ax3.set_ylabel("Count")
    fig3.tight_layout()
    fig3.savefig(results_dir / f"{model_name}_signals_distribution.png")
    plots.append(fig3)

    # ========== PLOT 4: Mid Price with Signal Background Shading ==========
    mid_price = (best_price_df['best_bid'] + best_price_df['best_ask']) / 2

    fig4, ax4 = plt.subplots()
    ax4.plot(best_price_df.index, mid_price, label='Mid Price', color='blue')

    signals_aligned = signals_df["signals"].reindex(mid_price.index, method="ffill")
    change_points = signals_aligned != signals_aligned.shift()
    change_times = signals_aligned.index[change_points].to_list()

    if len(change_times) > 0 and change_times[-1] != mid_price.index[-1]:
        change_times.append(mid_price.index[-1])

    for i in range(len(change_times) - 1):
        start_time = change_times[i]
        end_time = change_times[i + 1]
        sig_val = signals_aligned.loc[start_time]
        if isinstance(sig_val, pd.Series):
            sig_val = sig_val.iloc[0]
        shade_color = 'green' if sig_val > 0 else 'red' if sig_val < 0 else None
        if shade_color:
            ax4.axvspan(start_time, end_time, color=shade_color, alpha=0.3)

    ax4.set_title(f"{model_name}: Mid Price with Signal Background Shading")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Mid Price")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(results_dir / f"{model_name}_midprice_shaded.png")
    plots.append(fig4)

    # ========== PLOT 5: Drawdown Chart ==========
    cummax = cum_ret.cummax()
    drawdown = cum_ret - cummax
    fig5, ax5 = plt.subplots()
    ax5.plot(drawdown.index, drawdown.values)
    ax5.set_title(f"{model_name}: Drawdown")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Drawdown")
    fig5.tight_layout()
    fig5.savefig(results_dir / f"{model_name}_drawdown.png")
    plots.append(fig5)

    # ========== PLOT 6: Correlation Heatmap of Features ==========
    features_for_corr = feats_df.drop(columns=[LABEL_COL])
    corr_matrix = features_for_corr.corr()

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    cax = ax6.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    fig6.colorbar(cax)
    ax6.set_xticks(range(len(corr_matrix.columns)))
    ax6.set_xticklabels(corr_matrix.columns, rotation=90)
    ax6.set_yticks(range(len(corr_matrix.index)))
    ax6.set_yticklabels(corr_matrix.index)
    ax6.set_title(f"{model_name}: Feature Correlation Heatmap")
    fig6.tight_layout()
    fig6.savefig(results_dir / f"{model_name}_features_corr_heatmap.png")
    plots.append(fig6)

    plt.close()

    print(f"Plots saved in: {results_dir.resolve()}")

    # Explicitly return the list of plot objects


    return feats_df, signals_df, strategy_ret, final_ret, plots


def single_param_sensitivity(
    ticker: str,
    trade_date: str,
    use_file: bool,
    file_path: str,
    base_params: dict,
    param_values_dict: dict,
    output_dir: str = "data/results/param_sensitivity",
    base_model_name: str = "random_forest_hft_signal"
):
    """
    For each parameter in param_values_dict, iterate over its values (keeping all
    other parameters constant as defined in base_params), and plot 
    parameter_value vs. final_return.

    Parameters
    ----------
    ticker : str
        Ticker to analyze.
    trade_date : str
        Date of the data.
    use_file : bool
        If True, load data from a file. Otherwise, pull from WRDS.
    file_path : str
        Path to the file (if use_file=True).
    base_params : dict
        A dictionary of default parameter values. These stay fixed unless
        overridden by the parameter we're varying.
        For example:
            {
                "buy_threshold_std": 1.0,
                "sell_threshold_std": 1.0,
                "transaction_cost": 0.0001,
                "half_life_s": 60,
                ...
            }
    param_values_dict : dict
        Keys are parameter names, values are lists of parameter settings to test.
        For example:
            {
                "half_life_s": [30, 60, 90],
                "spread_window": ["30s", "60s"]
            }
    output_dir : str
        Where to save the resulting plots.
    base_model_name : str
        Base name for the model; individual runs will be suffixed.

    Returns
    -------
    results_df : pd.DataFrame
        A summary table of all runs, containing columns:
        [param_name, param_value, final_return]
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # This will hold all results across parameters
    all_runs = []

    # Loop over each parameter in param_values_dict
    for param_name, values in param_values_dict.items():
        # We'll store the final returns for this parameter in a list
        param_runs = []

        for val in values:
            # Copy the base_params and override the current parameter
            run_params = dict(base_params)
            run_params[param_name] = val

            # Create a custom model name suffix so we don't overwrite results
            model_name = f"{base_model_name}_{param_name}={val}"

            print(f"\n=== Varying '{param_name}' => {val} ===")

            # Run the pipeline
            feats_df, signals_df, strategy_ret, final_ret, _ = run_intraday_hft_pipeline_with_plots(
                ticker=ticker,
                trade_date=trade_date,
                use_file=use_file,
                file_path=file_path,
                model_name=model_name,
                **run_params  # Expand all the parameters
            )

            # Store the result
            param_runs.append((val, final_ret))

        # Convert to a DataFrame for plotting
        param_df = pd.DataFrame(param_runs, columns=[param_name, "final_return"]).sort_values(by=param_name)

        # Make a line plot of param value vs. final_return
        plt.figure()
        plt.plot(param_df[param_name], param_df["final_return"], marker='o')
        plt.title(f"Sensitivity of {param_name} vs. Strategy Final Return")
        plt.xlabel(param_name)
        plt.ylabel("Final Cumulative Return")
        plt.tight_layout()

        # Save the figure
        out_name = output_path / f"sensitivity_{param_name}.png"
        plt.savefig(out_name)
        plt.close()
        print(f"Saved plot: {out_name}")

        # Also store param_df in the all_runs list, marking the parameter
        param_df["parameter"] = param_name
        all_runs.append(param_df)

    # Combine all parameter data
    results_df = pd.concat(all_runs, ignore_index=True)
    return results_df

# Example usage:
if __name__ == "__main__":

    # Define the base parameters
    base_params = {
        "buy_threshold_std": 1.0,
        "sell_threshold_std": 1.0,
        "transaction_cost": 0.0001,
        "half_life_s": 60,
        "use_pca": False,
        "train_size": 0.8,
        "lookahead_timedelta": pd.Timedelta("1ms")
    }

    trade_date = "2024-03-07"
    ticker = 'SPY'

    # Output directory
    #output_dir = "./data/results/param_sensitivity"

    # Sensitivity to Trading Costs
    param_values_transaction_cost = {"transaction_cost": [0, 0.00005, 0.0001, 0.0005, 0.001]}
    results_tc = single_param_sensitivity(
        ticker='SPY',
        trade_date='2024-03-07',
        use_file=True,
        file_path="./data/processed/merged_trades.csv",
        base_params=base_params,
        param_values_dict=param_values_transaction_cost,
        base_model_name='sensitivity_transaction_cost'
    )
    results_tc.plot(x='transaction_cost', y='final_return', marker='o',
                    title="Sensitivity to Trading Costs", xlabel="Transaction Costs", ylabel="Final Returns")

    # Sensitivity to Lookahead Timedelta
    param_values_lookahead = {
        "lookahead_timedelta": [pd.Timedelta("1ms"), pd.Timedelta("10ms"), pd.Timedelta("50ms"), pd.Timedelta("100ms"), pd.Timedelta("500ms")]
    }
    results_lookahead = single_param_sensitivity(
        ticker='SPY',
        trade_date='2024-03-07',
        use_file=True,
        file_path="./data/processed/merged_trades.csv",
        base_params=base_params,
        param_values_dict=param_values_lookahead,
        base_model_name='sensitivity_lookahead'
    )
    results_lookahead.plot(x='lookahead_timedelta', y='final_return', marker='o', title="Sensitivity to Lookahead Timedelta", xlabel="Lookahead Timedelta", ylabel="Final Return")

    # Sensitivity to Training Size
    param_values_train_size = {
        "train_size": [0.6, 0.7, 0.8, 0.9]
    }
    results_train_size = single_param_sensitivity(
        ticker='SPY',
        trade_date='2024-03-07',
        use_file=True,
        file_path="./data/processed/merged_trades.csv",
        base_params=base_params,
        param_values_dict=param_values_train_size,
        base_model_name='sensitivity_train_size'
    )
    results_train_size.plot(x='train_size', y='final_return', marker='o', title="Sensitivity to Training Size", xlabel="Training Size", ylabel="Final Return")

    # Sensitivity to Buy and Sell Thresholds
    param_values_thresholds = {
        "sell_threshold_std": [0.5, 1.0, 2.0]
    }
    results_thresholds = single_param_sensitivity(
        ticker='SPY',
        trade_date='2024-03-07',
        use_file=True,
        file_path="./data/processed/merged_trades.csv",
        base_params=base_params,
        param_values_dict=param_values_thresholds,
        base_model_name='sensitivity_thresholds'
    )
    results_thresholds.groupby("parameter").plot(x='parameter', y='final_return', marker='o', title="Sensitivity to Buy/Sell Thresholds", xlabel="Threshold (std dev)", ylabel="Final Return")

    # Sensitivity to PCA usage
    param_values_pca = {
        "use_pca": [False, True]
    }
    results_pca = single_param_sensitivity(
        ticker='SPY',
        trade_date='2024-03-07',
        use_file=True,
        file_path="./data/processed/merged_trades.csv",
        base_params=base_params,
        param_values_dict=param_values_pca,
        base_model_name='sensitivity_pca'
    )
    results_pca.plot(x='use_pca', y='final_return', kind='bar', title="Sensitivity to PCA Usage", xlabel="PCA Usage", ylabel="Final Return", rot=0)

    # Display plots
    plt.show()