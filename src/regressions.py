import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
from typing import Union, List, Dict
import plotly.graph_objects as go

from utils import time_series_to_df, fix_dates_index

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

