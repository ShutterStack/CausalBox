# utils/time_series_causal.py
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def perform_granger_causality(data_list, timestamp_col, variables_to_analyze, max_lags=1):
    """
    Performs pairwise Granger Causality tests on the given time-series data.

    Args:
        data_list (list of dict): List of dictionaries representing the dataset.
        timestamp_col (str): Name of the timestamp column.
        variables_to_analyze (list): List of names of variables to test for causality.
        max_lags (int): The maximum number of lags to use for the Granger causality test.

    Returns:
        list: A list of dictionaries, each describing a causal relationship found.
    """
    df = pd.DataFrame(data_list)

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in data.")

    # Ensure timestamp column is datetime and set as index
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()
    except Exception as e:
        raise ValueError(f"Could not convert timestamp column '{timestamp_col}' to datetime: {e}")

    # Ensure all variables to analyze are numeric
    for col in variables_to_analyze:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Variable '{col}' is not numeric. Granger Causality requires numeric variables.")
        if df[col].isnull().any():
            # Handle NaNs: Granger Causality tests require no NaN values.
            # You might choose to drop rows with NaNs or impute.
            # For simplicity, here we'll raise an error or drop them.
            # print(f"Warning: Variable '{col}' contains NaN values. Rows with NaNs will be dropped.")
            df = df.dropna(subset=[col])


    # Select only the relevant columns
    df_selected = df[variables_to_analyze]

    # Granger Causality requires stationarity in theory.
    # While statsmodels can run on non-stationary data, results should be interpreted cautiously.
    # You might want to add differencing logic here (e.g., df.diff().dropna())
    # or a warning for the user.
    # For now, we proceed directly.

    causal_results = []
    
    # Iterate through all unique pairs of variables
    for i in range(len(variables_to_analyze)):
        for j in range(len(variables_to_analyze)):
            if i == j:
                continue # Skip self-causation tests

            cause_var = variables_to_analyze[i]
            effect_var = variables_to_analyze[j]

            # Prepare data for grangercausalitytests: [effect_var, cause_var]
            # grangercausalitytests takes a DataFrame where the first column is the dependent variable (effect)
            # and the second column is the independent variable (cause)
            data_for_test = df_selected[[effect_var, cause_var]]

            if data_for_test.empty or len(data_for_test) <= max_lags:
                # Not enough data points to perform test with specified lags
                # This can happen if NaNs were dropped or dataset is too small
                continue

            try:
                # Perform Granger Causality test
                # The output is a dictionary. The key 'ssr_ftest' (or 'params_ftest')
                # usually contains the p-value.
                test_result = grangercausalitytests(data_for_test, max_lags, verbose=False)
                
                # Extract p-value for the optimal lag or the test that interests you
                # Commonly, F-test p-value for the last lag tested is used
                # test_result is a dictionary where keys are lag numbers
                # Each lag has a tuple of (test_statistics, p_values).
                # (F-test, Chi2-test, LR-test, SSR-test) -> [statistic, p-value, df_denom, df_num]
                
                # Let's consider the F-test for the last lag as a general indicator
                last_lag_p_value = test_result[max_lags][0]['ssr_ftest'][1] # F-test p-value

                causal_results.append({
                    "cause": cause_var,
                    "effect": effect_var,
                    "p_value": last_lag_p_value,
                    "test_type": "Granger Causality (F-test)",
                    "max_lags": max_lags
                })
            except ValueError as ve:
                # Handle cases where the test cannot be performed (e.g., singular matrix)
                print(f"Could not perform Granger Causality for {cause_var} -> {effect_var} with max_lags={max_lags}: {ve}")
                continue # Skip this pair
            except Exception as e:
                print(f"An unexpected error occurred for {cause_var} -> {effect_var}: {e}")
                continue

    return causal_results