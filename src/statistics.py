import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, List, Dict
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import time_series_to_df, fix_dates_index, filter_columns_and_indexes

# ================================================================================================
# Regressions
# ================================================================================================

def calc_regression_rolling(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    factors: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    intercept: bool = True,
    moving_window: bool = False,
    exp_decaying_window: bool = False,
    window_size: int = 121,
    decay_alpha: float = 0.94, 
    betas_only: bool = True,
    fitted_values: bool = False,
    residuals: bool = False,
    ) -> Dict[datetime.datetime, pd.DataFrame]:

    """
    Performs a multiple OLS regression of a "one-to-many" returns time series with optional intercept on a rolling window. 
    Allows for different methods of windowing: expanding window (default), moving window, and exponential decay.

    This is the first stage of a Fama-MacBeth model, used to estimate the betas for every asset for every window.

    Parameters:
        returns (pd.DataFrame, pd.Series or List of pd.Series): Dependent variable for the regression.
        factors (pd.DataFrame, pd.Series or List of pd.Series): Independent variable(s) for the regression.
        intercept (bool, default=True): If True, includes an intercept in the regression.
        window_size (int): Number of observations to include in the moving window.
        betas_only (bool): If True, returns only the betas for each asset for each window.
        fitted_values (bool): If True, returns the fitted values.
        residuals (bool): If True, returns the residuals.
        moving_window (bool): If True, uses moving windows of size `window_size`.
        expanding_window (bool): If True, uses expanding windows of minimum size `window_size`.
        exp_decaying_window (bool): If True, uses expanding windows of minimum size `window_size` and exponential decaying weights.
        decay_alpha (float): Decay factor for exponential weighting.
        betas_only (bool, default=True): If True, returns only the betas for each asset for each window.
        fitted_values (bool, default=False): If True, also returns the fitted values of the regression.
        residuals (bool, default=False): If True, also returns the residuals of the regression.

    Returns: a dictionary of dataframes with the regression statistics for each rolling window.
    Returns the intercept (optional) and betas for each asset for each window.
    """
    
    factors = time_series_to_df(factors) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(factors) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float
    
    returns = time_series_to_df(returns) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float
    
    y_names = list(returns.columns) if isinstance(returns, pd.DataFrame) else [returns.name]
    factor_names = list(factors.columns)
    factor_names = ['Intercept'] + factor_names if (intercept == True and betas_only == False) else factor_names

    # Add the intercept
    if intercept:
        factors = sm.add_constant(factors)
    
    # Check if y and X have the same length
    if len(factors.index) != len(returns.index):
        print(f'y has lenght {len(returns.index)} and X has lenght {len(factors.index)}. Joining y and X by y.index...')
        df = returns.join(factors, how='left')
        df = df.dropna()
        returns = df[y_names]
        factors = df.drop(columns=y_names)
        if len(factors.index) < len(factors.columns) + 1:
            raise Exception('Indexes of y and X do not match and there are less observations than degrees of freedom. Cannot calculate regression')

    regres_columns = ['Beta (' + factor + ')' for factor in factor_names]
    regression_statistics = pd.DataFrame(index=returns.index, columns=regres_columns)

    # Loop through the windows
    for i in range(window_size, len(returns.index), 1):
        if exp_decaying_window:
            y_i = returns.iloc[:i]
            X_i = factors.iloc[:i]
            n_obs = i
            weights = np.array([decay_alpha ** (n_obs - j) for j in range(n_obs)])
            
            try:
                ols_model = sm.WLS(y_i, X_i, missing="drop", weights=weights)
            except ValueError:
                y_i = y_i.reset_index(drop=True)
                X_i = X_i.reset_index(drop=True)
                ols_model = sm.WLS(y_i, X_i, missing="drop", weights=weights)
                print('Required to reset indexes to make regression work. Try passing "y" and "X" as pd.DataFrame')
        else:
            if moving_window:
                y_i = returns.iloc[i-window_size:i]
                X_i = factors.iloc[i-window_size:i]
            else: # Expanding Window
                y_i = returns.iloc[:i]
                X_i = factors.iloc[:i]

            # Fit the regression model: 
            try:
                ols_model = sm.OLS(y_i, X_i, missing="drop")
            except ValueError:
                y_i = y_i.reset_index(drop=True)
                X_i = X_i.reset_index(drop=True)

                ols_model = sm.OLS(y_i, X_i, missing="drop")

                print('Required to reset indexes to make regression work. Try passing "y" and "X" as pd.DataFrame')
            
        ols_results = ols_model.fit() 

        # Process betas for explanatory variables
        coeff = ols_results.params[1:] if (intercept and betas_only) else ols_results.params
        regression_statistics.loc[returns.index[i], regres_columns] = coeff.values # Betas
        
        current_X = factors.loc[returns.index[i], :]
        current_y = returns.iloc[i][0]
        if fitted_values:
            regression_statistics.loc[returns.index[i], 'Fitted Values'] = current_X @ coeff # Fitted Value
        if residuals:
            regression_statistics.loc[returns.index[i], 'Residuals'] = current_y - current_X @ coeff # Residuals
    regression_statistics = regression_statistics.dropna(how='all')  

    return regression_statistics


# ===============================================================================================
# Statistics and Analysis
# ===============================================================================================


def calc_returns_statistics(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    annual_factor: int = None,
    provided_excess_returns: bool = None,
    rf_returns: Union[pd.Series, pd.DataFrame] = None,
    var_quantile: Union[float , List] = .05,
    timeframes: Union[None, dict] = None,
    return_tangency_weights: bool = False,
    correlations: Union[bool, List] = False,
    tail_risks: bool = True,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
    _timeframe_name: str = None,
) -> pd.DataFrame:
    """
    Calculates summary statistics for a time series of returns.   

    Parameters:
        returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
        annual_factor (int, default=None): Factor for annualizing returns.
        provided_excess_returns (bool, default=None): Whether excess returns are already provided.
        rf (pd.Series or pd.DataFrame, default=None): Risk-free rate data.
        var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
        timeframes (dict or None, default=None): Dictionary of timeframes [start, finish] to calculate statistics for each period.
        return_tangency_weights (bool, default=True): If True, returns tangency portfolio weights.
        correlations (bool or list, default=True): If True, returns correlations, or specify columns for correlations.
        tail_risks (bool, default=True): If True, include tail risk statistics.
        keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
        drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
        keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
        drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: Summary statistics of the returns.
    """

    returns = time_series_to_df(returns) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float

    if rf_returns is not None:
        rf_returns = time_series_to_df(rf_returns) # Convert returns to DataFrame if it is a Series
        fix_dates_index(rf_returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float
        rf_returns = rf_returns.reindex(returns.index).dropna()
        
        if len(rf_returns.index) != len(returns.index):
            raise Exception('"rf_returns" has missing data to match "returns" index')
        if type(rf_returns) == pd.DataFrame:
            rf = rf_returns.iloc[:, 0].to_list()
        elif type(rf_returns) == pd.Series:
            rf = rf_returns.to_list()
    
    if keep_columns is None:
        keep_columns = ['Accumulated Return', 'Annualized Mean', 'Annualized Vol', 'Annualized Sharpe', 'Annualized Sortino', 'Min', 'Mean', 'Max', 'Correlation']
        if tail_risks == True:
            keep_columns += ['Skewness', 'Excess Kurtosis', f'Historical VaR', f'Annualized Historical VaR', 
                                f'Historical CVaR', f'Annualized Historical CVaR', 'Max Drawdown', 
                                'Peak Date', 'Bottom Date', 'Recovery', 'DD Duration (days)']
    if return_tangency_weights == True:
        keep_columns += ['Tangency Portfolio']
    if correlations != False:
        keep_columns += ['Correlation']

    # Iterate to calculate statistics for multiple timeframes
    if isinstance(timeframes, dict):
        all_timeframes_summary_statistics = pd.DataFrame({})
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_returns = returns.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_returns = returns.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_returns = returns.loc[:timeframe[1]]
            else:
                timeframe_returns = returns.copy()
            if len(timeframe_returns.index) == 0:
                raise Exception(f'No returns data for {name} timeframe')
            
            timeframe_returns = timeframe_returns.rename(columns=lambda col: col + f' ({name})')
            timeframe_summary_statistics = calc_returns_statistics(
                returns=timeframe_returns,
                annual_factor=annual_factor,
                provided_excess_returns=provided_excess_returns,
                rf_returns=rf_returns,
                var_quantile=var_quantile,
                timeframes=None,
                return_tangency_weights=return_tangency_weights,
                correlations=correlations,
                tail_risks=tail_risks,
                _timeframe_name=name,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes
            )
            all_timeframes_summary_statistics = pd.concat(
                [all_timeframes_summary_statistics, timeframe_summary_statistics],
                axis=0
            )
        return all_timeframes_summary_statistics

    # Calculate summary statistics for a single timeframe
    summary_statistics = pd.DataFrame(index=returns.columns)
    summary_statistics['Mean'] = returns.mean()
    summary_statistics['Annualized Mean'] = returns.mean() * annual_factor
    summary_statistics['Vol'] = returns.std()
    summary_statistics['Annualized Vol'] = returns.std() * np.sqrt(annual_factor)
    if provided_excess_returns is True:
        if rf_returns is not None:
            print('Excess returns and risk-free were both provided.'
                ' Excess returns will be consider as is, and risk-free rate given will be ignored.\n')
        summary_statistics['Sharpe'] = returns.mean() / returns.std()
        summary_statistics['Sortino'] = returns.mean() / returns[returns < 0].std()
    else:
        try:
            if rf_returns is None:
                warnings.warn('No risk-free rate provided. Interpret "Sharpe" as "Mean/Volatility" and "Sortino Ratio as "Mean/DownVolatility.\n')
                summary_statistics['Sharpe'] = returns.mean() / returns.std()
                summary_statistics['Sortino'] = returns.mean() / returns[returns < 0].std()
            else:
                excess_returns = returns.subtract(rf_returns.iloc[:, 0], axis=0)
                summary_statistics['Sharpe'] = excess_returns.mean() / returns.std()
                summary_statistics['Sortino'] = excess_returns.mean() / excess_returns[excess_returns < 0].std()
        except Exception as e:
            print(f'Could not calculate Sharpe: {e}')

    summary_statistics['Annualized Sharpe'] = summary_statistics['Sharpe'] * np.sqrt(annual_factor)
    summary_statistics['Annualized Sortino'] = summary_statistics['Sortino'] * np.sqrt(annual_factor)
    summary_statistics['Min'] = returns.min()
    summary_statistics['Max'] = returns.max()

    summary_statistics['Win Rate'] = (returns > 0).mean()
    
    if tail_risks == True:
        tail_risk_stats = stats_tail_risk(returns,
                                        annual_factor=annual_factor,
                                        var_quantile=var_quantile,
                                        keep_indexes=keep_indexes,
                                        drop_indexes=drop_indexes)
        
        summary_statistics = summary_statistics.join(tail_risk_stats)

    if correlations is True or isinstance(correlations, list):
               
        returns_corr = returns.corr()
        if _timeframe_name:
            returns_corr = returns_corr.rename(columns=lambda col: col.replace(f' ({_timeframe_name})', ''))

        if isinstance(correlations, list):
            # Check if all selected columns exist in returns_corr
            corr_not_in_returns_corr = [col for col in correlations if col not in returns_corr.columns]
            if len(corr_not_in_returns_corr) > 0:
                not_in_returns_corr = ", ".join([c for c in corr_not_in_returns_corr])
                raise Exception(f'{not_in_returns_corr} not in returns columns')
            
            returns_corr = returns_corr[[col for col in correlations]]
            
        returns_corr = returns_corr.rename(columns=lambda col: col + ' Correlation')
        
        # Specify a suffix to be added to overlapping columns
        summary_statistics = summary_statistics.join(returns_corr)
    
    return filter_columns_and_indexes(
        summary_statistics,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )



def stats_tail_risk(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    annual_factor: int = None,
    var_quantile: Union[float , List] = 0.05,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None,
) -> pd.DataFrame:
    """
    Calculates tail risk summary statistics for a time series of returns.   

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
    annual_factor (int, default=None): Factor for annualizing returns.
    var_quantile (float or list, default=0.05): Quantile for Value at Risk (VaR) calculation.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
    pd.DataFrame: tail risk summary statistics of the returns.
    """

    returns = time_series_to_df(returns) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float

    tail_risk_stats = pd.DataFrame(index=returns.columns)

    tail_risk_stats['Skewness'] = returns.skew()
    tail_risk_stats['Excess Kurtosis'] = returns.kurtosis()
    var_quantile = [var_quantile] if isinstance(var_quantile, (float, int)) else var_quantile
    for var_q in var_quantile:
        tail_risk_stats[f'Historical VaR ({var_q:.1%})'] = returns.quantile(var_q, axis = 0)
        tail_risk_stats[f'Historical CVaR ({var_q:.1%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean()
        if annual_factor:
            tail_risk_stats[f'Annualized Historical VaR ({var_q:.1%})'] = returns.quantile(var_q, axis = 0) * np.sqrt(annual_factor)
            tail_risk_stats[f'Annualized Historical CVaR ({var_q:.1%})'] = returns[returns <= returns.quantile(var_q, axis = 0)].mean() * np.sqrt(annual_factor)
    
    cum_returns = (1 + returns).cumprod()
    maximum = cum_returns.cummax()
    drawdown = cum_returns / maximum - 1

    tail_risk_stats['Accumulated Return'] = cum_returns.iloc[-1] - 1
    tail_risk_stats['Max Drawdown'] = drawdown.min()
    tail_risk_stats['Peak Date'] = [maximum[col][:drawdown[col].idxmin()].idxmax() for col in maximum.columns]
    tail_risk_stats['Bottom Date'] = drawdown.idxmin()
    
    recovery_date = []
    for col in cum_returns.columns:
        prev_max = maximum[col][:drawdown[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([cum_returns[col][drawdown[col].idxmin():]]).T
        recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
    tail_risk_stats['Recovery'] = recovery_date

    tail_risk_stats["Recovery"] = pd.to_datetime(tail_risk_stats["Recovery"], errors='coerce')
    tail_risk_stats["Bottom Date"] = pd.to_datetime(tail_risk_stats["Bottom Date"], errors='coerce')

    # Perform subtraction safely
    tail_risk_stats["DD Duration (days)"] = [
        (i - j).days if pd.notna(i) and pd.notna(j) else "-" 
        for i, j in zip(tail_risk_stats["Recovery"], tail_risk_stats["Bottom Date"])
    ]

    return filter_columns_and_indexes(
        tail_risk_stats,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )


def calc_correlations(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    print_highest_lowest: bool = True,
    show_heatmap: bool = True,
    return_matrix: bool = False,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
) -> Union[sns.heatmap, pd.DataFrame]:
    """
    Calculates the correlation matrix of the provided returns and optionally prints or visualizes it.

    Parameters:
        returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.
        print_highest_lowest (bool, default=True): If True, prints the highest and lowest correlations.
        show_heatmap (bool, default=False): If True, returns a heatmap of the correlation matrix.
        keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
        drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
        keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
        drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
        fig_size (tuple, default=(PLOT_WIDTH, PLOT_HEIGHT)): Size of the figure for the heatmap.
        save_plot (bool, default=False): If True and show_heatmap is also True, saves the heatmap as a PNG.
        output_dir (Path, optional): Directory to save the heatmap if save_plot is True.
        plot_name (str, optional): Custom filename prefix. If None, defaults to 'heatmap_correlations'.

    Returns:
    sns.heatmap or pd.DataFrame: Heatmap of the correlation matrix or the correlation matrix itself.
    """

    returns = time_series_to_df(returns)  # convert to DataFrame if needed
    fix_dates_index(returns)  # ensure datetime index and float dtype

    returns = filter_columns_and_indexes(
        returns,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )

    correlation_matrix = returns.corr()

    if print_highest_lowest:
        highest_lowest_corr = (
            correlation_matrix
            .unstack()
            .sort_values()
            .reset_index()
            .set_axis(['asset_1', 'asset_2', 'corr'], axis=1)
            .loc[lambda df: df.asset_1 != df.asset_2]
        )
        highest_corr = highest_lowest_corr.iloc[-1, :]
        lowest_corr = highest_lowest_corr.iloc[0, :]
        print(f'The highest correlation ({highest_corr["corr"]:.4f}) is between {highest_corr.asset_1} and {highest_corr.asset_2}')
        print(f'The lowest correlation ({lowest_corr["corr"]:.4f}) is between {lowest_corr.asset_1} and {lowest_corr.asset_2}')

    if show_heatmap:
        num_assets = len(correlation_matrix.columns)
        fig_width = max(6, num_assets * 0.6)  # Scale width dynamically
        fig_height = max(5, num_assets * 0.5)  # Scale height dynamically

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Adjust figure size
        heatmap = sns.heatmap(
            correlation_matrix, 
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        plt.xticks(rotation=45)  # Rotate labels for better visibility if needed
        plt.yticks(rotation=0)
        plt.title("Correlation Matrix of Returns", fontsize=14)
        plt.show()


        # Return the heatmap object if needed (though typically you don't "return" a heatmap)
        if return_matrix:
            return correlation_matrix
        else:
            return heatmap

    if return_matrix:
        return correlation_matrix
    else:
        return None
    
def calc_downside_correlations(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    print_highest_lowest: bool = True,
    show_heatmap: bool = True,
    return_matrix: bool = False,
    is_int_rate: bool = False,
    num_std_dev: int = 1,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
) -> Union[sns.heatmap, pd.DataFrame, None]:
    """
    Calculates the correlation matrix of the provided returns, but only for those dates
    when the daily mean (across all assets) is more than 1 standard deviation below the
    grand mean across all dates. Optionally prints or visualizes the correlation matrix.

    Parameters:
        returns (pd.DataFrame, pd.Series or List[pd.Series]): Time series of returns.
        print_highest_lowest (bool, default=True): If True, prints the highest and lowest correlations.
        show_heatmap (bool, default=True): If True, returns a heatmap of the correlation matrix.
        return_matrix (bool, default=False): If True, returns the correlation matrix as a pd.DataFrame.
        is_int_rate (bool, default=False): If True, indicates that the returns are interest rates. In that case, the sign of "downside" is flipped.
        num_std_dev (int, default=1): Number of standard deviations to use for filtering "downside" days.
        keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
        drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
        keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
        drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.

    Returns:
        sns.heatmap or pd.DataFrame or None:
            - sns.heatmap if show_heatmap=True and return_matrix=False
            - pd.DataFrame if return_matrix=True
            - None if show_heatmap=False and return_matrix=False
    """

    # Convert returns to a DataFrame if needed and ensure proper formatting
    returns = time_series_to_df(returns)  
    fix_dates_index(returns)

    # Filter columns and indexes as desired
    returns = filter_columns_and_indexes(
        returns,
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        keep_indexes=keep_indexes,
        drop_indexes=drop_indexes
    )

    # Calculate the daily mean across assets
    daily_mean = returns.mean(axis=1)
    
    # Calculate the grand mean and standard deviation of the daily means
    grand_mean = daily_mean.mean()
    std_dev = daily_mean.std()

    # Filter for "downside" days (daily mean < grand_mean - std_dev)
    if is_int_rate == False:
        downside_mask = daily_mean < (grand_mean - std_dev * num_std_dev)
        returns_downside = returns.loc[downside_mask]
    elif is_int_rate == True:
        downside_mask = daily_mean > (grand_mean + std_dev * num_std_dev)
        returns_downside = returns.loc[downside_mask]

    # If there are no downside days, we can return early or raise a warning
    if returns_downside.empty:
        print("No downside days found based on the chosen criteria.")
        return None

    correlation_matrix = returns_downside.corr()

    if print_highest_lowest:
        # Flatten and sort correlation matrix
        highest_lowest_corr = (
            correlation_matrix
            .unstack()
            .sort_values()
            .reset_index()
            .set_axis(['asset_1', 'asset_2', 'corr'], axis=1)
            .loc[lambda df: df.asset_1 != df.asset_2]
        )
        highest_corr = highest_lowest_corr.iloc[-1, :]
        lowest_corr = highest_lowest_corr.iloc[0, :]

        print(f"[Downside] Highest correlation ({highest_corr['corr']:.4f}) "
              f"is between {highest_corr.asset_1} and {highest_corr.asset_2}")
        print(f"[Downside] Lowest correlation ({lowest_corr['corr']:.4f}) "
              f"is between {lowest_corr.asset_1} and {lowest_corr.asset_2}")

    # Step 8 (optional): Show heatmap
    if show_heatmap:
        num_assets = len(correlation_matrix.columns)
        fig_width = max(6, num_assets * 0.6)
        fig_height = max(5, num_assets * 0.5)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        heatmap = sns.heatmap(
            correlation_matrix,
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.title("Downside Correlation Matrix of Returns", fontsize=14)
        plt.show()

        if return_matrix:
            return correlation_matrix
        return heatmap

    if return_matrix:
        return correlation_matrix
    return None


def calc_strategy_correlations(strategy_returns: pd.Series,
                               factors_returns: pd.DataFrame,
                               strategy_returns_name: str = 'Strategy') -> pd.Series:
    """
    Compare the strategy returns to factor returns (e.g., Fama-French factors) by computing correlations.

    Parameters
    ----------
    strategy_returns : pd.Series
        A pandas Series of strategy daily returns indexed by date.
    ff_returns : pd.DataFrame
        A pandas DataFrame of factor returns indexed by date.
    etf_pair : tuple, optional
        Tuple containing names of the assets in the strategy, e.g., ('ETF1', 'ETF2').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the correlation of the strategy with each factor.
    """
    # Ensure date indexes are in datetime format
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    factors_returns.index = pd.to_datetime(factors_returns.index)

    # Merge the strategy returns with factor returns on the date index
    combined_data = pd.merge(strategy_returns, factors_returns, left_index=True, right_index=True, how='left')

    # Compute the correlation matrix
    correlation_matrix = combined_data.corr()

    # Extract the correlation of the strategy with the factor returns
    strategy_corr_FF = correlation_matrix.loc[correlation_matrix.columns != strategy_returns.name, strategy_returns.name]
    strategy_corr_FF.name = strategy_returns_name
    
    return strategy_corr_FF

def plot_histogram_grid(df):
    # Create a 4x4 grid of histograms for each column in the DataFrame
    fig = make_subplots(rows=4, cols=4, subplot_titles=list(df.columns))

    row, col = 1, 1
    for column in df.columns:
        fig.add_trace(
            go.Histogram(x=df[column], name=column, showlegend=False),
            row=row, col=col
        )
        col += 1
        if col > 4:
            col = 1
            row += 1

    fig.update_layout(height=800, width=800, title_text="4x4 Grid of Histograms")
    fig.show()
    return
