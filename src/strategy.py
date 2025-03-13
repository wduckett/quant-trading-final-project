import os
import joblib
import numpy as np
import pandas as pd

from typing import Union
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import Memory

from pathlib import Path

from settings import config

# =============================================================================
# Global Configuration
# =============================================================================

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

memory = Memory(location=f"{OUTPUT_DIR}/cache/", verbose=0)

################################################################################
# 1) Prepare the dataset for ML: define future return, create classification labels
################################################################################

def create_labels(
    df: pd.DataFrame,
    price_col: str = "price",
    future_vwap_col: str = "future_vwap",
    out_return_col: str = "future_return",
    out_label_col: str = "ml_signal",
    buy_threshold_std: float = 1.0,
    sell_threshold_std: float = 1.0
) -> pd.DataFrame:
    """
    From the current price and a 'future_vwap', compute a future return:
        future_return_t = log( future_vwap_t / price_t )

    Then create a 3-class label: +1 (buy), 0 (hold), -1 (sell),
    based on whether the future return is above or below mean +/- X std dev.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns [price_col, future_vwap_col].
    price_col : str
        Current trade price column name.
    future_vwap_col : str
        Future VWAP column name (the target "future" price).
    out_return_col : str
        Name for the output future return column.
    out_label_col : str
        Name for the output discrete signal/label column.
    buy_threshold_std : float
        # of std dev above mean for labeling a 'buy' signal.
    sell_threshold_std : float
        # of std dev below mean for labeling a 'sell' signal.

    Returns:
    --------
    df : pd.DataFrame
        The same DataFrame with two extra columns:
          - out_return_col (float)
          - out_label_col (integer in {-1,0,1})
    """
    df = df.copy()
    df.dropna(subset=[price_col, future_vwap_col], inplace=True)

    # 1) Compute log future return
    df[out_return_col] = np.log(df[future_vwap_col] / df[price_col])

    # 2) Label as +1, 0, -1 based on thresholds (mean +/- std dev)
    mean_val = df[out_return_col].mean()
    std_val = df[out_return_col].std()

    buy_threshold = mean_val + buy_threshold_std * std_val
    sell_threshold = mean_val - sell_threshold_std * std_val

    def get_signal(x):
        if x > buy_threshold:
            return 1
        elif x < sell_threshold:
            return -1
        else:
            return 0

    df[out_label_col] = df[out_return_col].apply(get_signal)

    df = df.dropna(subset=[out_return_col])

    # Remove Na
    return df


##############################################################################################
# 2) Train a random forest classification model (TimeSeriesSplit + Pipeline) with optional PCA
##############################################################################################

def train_ml_model_pipeline(
    data: pd.DataFrame,
    label_col: str = "ml_signal",
    non_scaling_cols: list = None,
    use_pca: bool = True,
    pca_components: float = 0.90,
    train_size: float = 0.8,
    random_state: int = 42,
    model_name: str = "random_forest_future_return",
    model_path: str = f"{OUTPUT_DIR}/models/"
):
    """
    Trains a RandomForestClassifier (classification of future returns).
    Performs time-series splits, random hyperparam search, then final evaluation 
    on a holdout test set.

    Parameters
    ----------
    data : pd.DataFrame
        The feature DataFrame (already containing the future_return column and a discrete label column).
    label_col : str
        Name of the classification label column. Default = "ml_signal".
    non_scaling_cols : list
        Columns that we do NOT want to scale (e.g. trade_sign, spread_Zscore, etc.).
    use_pca : bool
        Whether to apply PCA for dimensionality reduction after scaling.
    pca_components : float or int
        If float (0 < float < 1), select the number of components such that 
        the amount of variance that needs to be explained is greater than the float value.
        If int, the number of components to keep.
    train_size : float
        Fraction of data to be used for training. The remainder is the final test set.
    random_state : int
        Seed for reproducibility.
    model_name : str
        Name of the saved model.
    model_path : str
        Path to save the trained model.

    Returns
    -------
    model : Pipeline
        The best estimator pipeline (including preprocessing and RF) after RandomizedSearchCV.
    """
    print("state 0")
    df = data.copy().dropna()
    # Separate target from features
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    print("state 1")
    # List out columns that need or do not need scaling
    if non_scaling_cols is None:
        # discrete sign signals or z-scores
        non_scaling_cols = ["trade_sign", "spread_Zscore"]
    print("state 2")
    if "spread_Zscore" in X.columns:
        X["spread_Zscore"] = X["spread_Zscore"].astype("float")
    print("state 3")
    # Basic train/test split by time
    total_size = len(X)
    print(total_size, len(y))
    train_cutoff = int(train_size * total_size)

    X_train_full, X_test = X.iloc[:train_cutoff], X.iloc[train_cutoff:]
    y_train_full, y_test = y.iloc[:train_cutoff], y.iloc[train_cutoff:]
    print("state 4")
    # Identify numeric columns
    numeric_cols = [c for c in X_train_full.columns 
                    if pd.api.types.is_numeric_dtype(X_train_full[c])]

    # The columns to scale are numeric but not in the non_scaling_cols
    scaling_cols = [c for c in numeric_cols if c not in non_scaling_cols]
    print("state 6")
    # Build column transformer
    transformers = []
    if len(scaling_cols) > 0:
        transformers.append(("scale", StandardScaler(), scaling_cols))

    # Pass through the non-scaling columns
    transformers.append(("passthrough", "passthrough", non_scaling_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    print("state 7")
    # If want PCA, Chain PCA in the pipeline after scaling
    pipeline_steps = [("preprocessor", preprocessor)]
    if use_pca:
        pipeline_steps.append(("pca", PCA(n_components=pca_components, svd_solver='full', random_state=random_state)))
    print("state 8")
    # Add Random Forest as final step, uses parallel processing
    pipeline_steps.append(("rf", RandomForestClassifier(n_jobs=-1, random_state=random_state, class_weight="balanced")))
    print("state 9")
    # Uses joblib.Memory to cache the transformers and the classifier to avoid redundant computation
    pipeline = Pipeline(steps=pipeline_steps, memory=memory)
    print("state 10")
    # TimeSeriesSplit for cross-validation to test on different time periods, avoiding overfitting.
    validation_set_size = len(X_test)  # size of the test/validation set
    fold_size = max(1, validation_set_size // 5) # Choose a reasonable validation fold size (~20% of validation set)
    n_splits = max(2, (train_cutoff // fold_size)) # Determine the number of splits dynamically based on the training set size
    print("state 11")
    tscv = TimeSeriesSplit(n_splits=n_splits) # Create TimeSeriesSplit object with calculated splits

    # Hyperparameter search space
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [4, 8, 16],
        'rf__min_samples_split': [10, 50, 100]
    }
    print("state 12")
    grid_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=2,  # Adjust later, use maybe 5
        cv=3, # tscv, instead of "3"
        scoring='accuracy',
        random_state=random_state,
        verbose=1
    )
    print("state 13")
    print(len(X_train_full))
    print(len(y_train_full))
    print(":3 0")
    na_indices = set()
    for idx, row in X_train_full.iterrows():
        if any(row.isna()):
            na_indices.add(idx)
    print(":3 1")
    for idx, val in enumerate(y_train_full):
        if np.isnan(val):
            na_indices.add(idx)
    print(":3 2")
    na_indices = list(na_indices)
    print(len(na_indices))
    print(sum(X_train_full.isna().values.ravel()))
    print(y_train_full.isna().sum())
    X_train_full.drop(na_indices, axis=0, inplace=True)
    y_train_full.drop(na_indices, axis=0, inplace=True)
    print(":3 3")
    print(sum(X_train_full.isna().values.ravel()))
    print(y_train_full.isna().sum())
    print(":3 4")
    print(len(X_train_full))
    print(len(y_train_full))
    grid_search.fit(X_train_full, y_train_full)
    model = grid_search.best_estimator_
    print("state 14")
    # Final evaluation on the holdout test set
    y_train_pred = model.predict(X_train_full)
    y_test_pred = model.predict(X_test)

    print("\n--- Final Model Performance ---")
    print("Best hyperparameters:", grid_search.best_params_)
    print(f"Accuracy:\n  Train: {accuracy_score(y_train_full, y_train_pred):.2%}\n  Test:  {accuracy_score(y_test, y_test_pred):.2%}")
    print(f"F1-Score:\n  Train: {f1_score(y_train_full, y_train_pred, average='weighted'):.2%}\n  Test:  {f1_score(y_test, y_test_pred, average='weighted'):.2%}")

    # Save the model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_path = os.path.join(model_path, f"{model_name}.pkl")
    joblib.dump(model, file_path)
    print(f"Model saved to: {file_path}")

    # Clear cached computations
    memory.clear()
    
    return model


###############################################################################################
# 3) Get the return from the trade signal, considering that the traded price is the next
#    best ask price (for buys) or best bid price (for sells) after 1ms from the current decision.
###############################################################################################

def signal_to_returns(
    signals: Union[pd.DataFrame, pd.Series],
    best_prices: Union[pd.DataFrame],
    ask_col: str = "best_ask",
    bid_col: str = "best_bid",
    strategy_name: str = None,
    lookahead_timedelta: pd.Timedelta = pd.Timedelta("1ms"),
    up_weight: float = 1.0,
    down_weight: float = 1.0,
    trade_style: str = "swing",
    transaction_cost: float = 0.0001 # 1bp per trade
) -> pd.DataFrame:
    """
    Translate trading signals into realized returns by applying the next available
    ask price (after a small 1ms lookahead) as the execution price if the signal=+1
    (and similarly, the next bid price if we had it).
    For simplicity here, we demonstrate using `ask_prices` only.

    Parameters:
    -----------
    signals : DataFrame/Series
        Trading signals for each timestamp, e.g. -1, 0, +1, with times as indexes. Must have a DateTimeIndex or similar.
    best_prices : DataFrame
        The best ask and bid price series, with times as indexes. Must have a DateTimeIndex or similar.
    ask_col : str
        Column name for the best ask price.
    bid_col : str
        Column name for the best bid price.
    time_col : str
        Column name for the timestamp (if not the index).
    strategy_name : str
        Strategy name label.
    lookahead_timedelta : pd.Timedelta
        The time offset to fetch the next ask price for execution to account for delay in order routing.
    up_weight : float
        Multiplier for the +1 signals
    down_weight : float
        Multiplier for the -1 signals
    trade_style : str
        When to trade. "swing" is for swinging between long and short
    transaction_cost : float
        Transaction cost per trade (e.g. 1bp = 0.0001)

    Returns:
    --------
    strategy_returns : pd.DataFrame
        Series (or DataFrame) of strategy returns.
    """
    best_prices = best_prices.copy()
    sig = signals.copy()

    if trade_style not in ["swing"]:
        print("Unrecognized trade style. Using default behavior (swing).")
        trade_style = "swing"

    # Ensure signals is a DataFrame
    if isinstance(sig, pd.Series):
        sig = sig.to_frame()
    # We'll just rename the first column to 'signal'
    elif sig.shape[1] > 1:
        print("Warning: multiple signal columns. Using the first column.")
        sig = sig.iloc[:, 0]

    col_signal = sig.columns[0]

    # Shift the signals by the lookahead, merging on time
    #    so that if signal at time t is +1, we find the ask price at t + 1ms or next available.
    sig = sig.sort_index()
    best_prices = best_prices.sort_index()

    # Create a shifted index for signals
    execution_times = sig.index + lookahead_timedelta
    # Reindex best_prices to these execution_times (asof or forward-fill as needed)
    # Better approach:
    sig["execution_time"] = execution_times
    merged = pd.merge_asof(
        sig.reset_index().rename(columns={"time_trade": "orig_time"}),
        best_prices.reset_index(),
        left_on="execution_time",
        right_on="time_trade",
        direction="forward"  # or 'backward'
    )

    # Sort by original time and set it as index
    merged = merged.sort_values("orig_time").set_index("orig_time")

    # Map raw signals to weighted positions:
    #   1  -> up_weight
    #   0  -> 0
    #  -1  -> -down_weight
    merged["weighted_signal"] = merged[col_signal].map({1: up_weight, 0: 0, -1: -down_weight})

    if trade_style == "swing":
        if up_weight == -down_weight:
            print("Strategy does the same thing on up and down signals. Results will be meaningless. Terminating.")
            assert(False)
        elif up_weight == 0:
            print("Warning: up weight being changed to 1e-8")
            up_weight = 1e-8
        elif down_weight == 0:
            print("Warning: down weight being changed to 1e-8")
            down_weight = 1e-8
        
        drop_rows = []
        if merged["weighted_signal"].iloc[0] == up_weight:
            last = up_weight
        elif merged["weighted_signal"].iloc[0] == -down_weight:
            last = -down_weight
        else:
            drop_rows.append(signals.index[0])
            if merged["weighted_signal"].iloc[1] == up_weight:
                last = up_weight
            elif merged["weighted_signal"].iloc[1] == -down_weight:
                last = -down_weight
            else:
                print("Ensure signals df is filtered to alternate signals.")
                assert(False)
        
        for i in range(2, len(merged)):
            if last == up_weight:
                if merged["weighted_signal"].iloc[i] == -down_weight:
                    last = -down_weight
                else:
                    drop_rows.append(signals.index[i])
            else:
                if merged["weighted_signal"].iloc[i] == up_weight:
                    last = up_weight
                else:
                    drop_rows.append(signals.index[i])

        merged.drop(index=drop_rows, inplace=True)

    # Get previous weighted position (defaulting to 0 for the first observation)
    merged["prev_weighted_signal"] = merged["weighted_signal"].shift(1).fillna(0)

    # Trade size is the change in weighted position
    merged["trade_size"] = merged["weighted_signal"] - merged["prev_weighted_signal"]

    # Determine executed trade price:
    # If trade_size > 0, it's a buy executed at the ask; if trade_size < 0, it's a sell executed at the bid.
    merged["executed_price"] = np.where(merged["trade_size"] > 0, merged[ask_col], merged[bid_col])

    # **Add this line to compute the price change**
    merged["price_change"] = merged["executed_price"].pct_change()

    # Compute strategy return:
    # Use the previous trade size (i.e. the position held) multiplied by the subsequent price change, adjusted for transaction costs
    merged["strategy_return"] = (
        merged["trade_size"].shift(1) * merged["price_change"]
        - np.abs(merged["trade_size"]) * transaction_cost
    )

    # Drop any rows with missing returns (e.g., the first row)
    merged.dropna(subset=["strategy_return"], inplace=True)

    # Optionally, rename the strategy return column
    merged.rename(columns={"strategy_return": f"{strategy_name}_returns"}, inplace=True)

    return merged

