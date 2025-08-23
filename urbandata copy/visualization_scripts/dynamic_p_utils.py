import numpy as np
import warnings
import pandas as pd

def calculate_dynamic_p(cbsa_df):
    """Calculates the longitudinal p_t series for a given CBSA DataFrame.
    
    This function computes population-weighted p_t values where the weight
    for each observation is the end-year population (corresponding to the
    growth rate observation period).
    
    Args:
        cbsa_df: DataFrame with columns ['block_group_fips', 'year', 'mean_income', 'population']
        
    Returns:
        tuple: (p_t_series, sorted_years) where p_t_series is clipped to (1e-5, 1-1e-5)
    """
    yearly_wins = {}
    yearly_totals = {}
    
    for _, group in cbsa_df.groupby('block_group_fips'):
        group = group.sort_values('year')
        if len(group) > 1 and not (group['mean_income'] <= 0).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                log_income = np.log(group['mean_income'].values)
            y_values = np.diff(log_income)
            if np.all(np.isfinite(y_values)):
                years = group['year'].values[1:]  # Years corresponding to y_values
                pop_end = group['population'].values[1:]  # End-year population weights
                for i, year in enumerate(years):
                    weight = float(pop_end[i]) if np.isfinite(pop_end[i]) and pop_end[i] > 0 else 0.0
                    if year not in yearly_totals:
                        yearly_totals[year] = 0.0
                        yearly_wins[year] = 0.0
                    yearly_totals[year] += weight
                    if y_values[i] > 0:
                        yearly_wins[year] += weight
    
    # Use only years with positive total weight
    sorted_years = [yr for yr in sorted(yearly_totals.keys()) if yearly_totals[yr] > 0]
    p_t_series = np.array([yearly_wins[year] / yearly_totals[year] for year in sorted_years])
    
    # Ensure p_t is within (0, 1) for logit transform
    return np.clip(p_t_series, 1e-5, 1 - 1e-5), sorted_years

def sanitize_filename(filename):
    """Removes or replaces characters from a string to make it a valid filename."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

def compute_weighted_mean_trajectory(cbsa_results):
    """
    Computes the population-weighted mean x_trajectory for a single CBSA.

    Args:
        cbsa_results: A list of result dictionaries for a single CBSA.

    Returns:
        pandas.Series: A time series representing the weighted mean x_trajectory,
                       with years as the index.
    """
    if not cbsa_results:
        return pd.Series(dtype=float)

    cbsa_x_series_list = []
    cbsa_weight_series_list = []
    for res in cbsa_results:
        if res.get("status") == "success":
            x_traj = res["x_mean_trajectory"]
            bg_years = res["year"][1:]
            # Population at the end of the growth period acts as the weight
            weights = res["population_ts"][1:]

            cbsa_x_series_list.append(pd.Series(x_traj, index=bg_years))
            cbsa_weight_series_list.append(pd.Series(weights, index=bg_years))

    if not cbsa_x_series_list:
        return pd.Series(dtype=float)

    # Combine lists of series into dataframes, aligning on the year index
    df_x = pd.concat(cbsa_x_series_list, axis=1)
    df_weights = pd.concat(cbsa_weight_series_list, axis=1)

    # Calculate the numerator (sum of x * weight for each year)
    weighted_x_sum = (df_x * df_weights).sum(axis=1)
    
    # Calculate the denominator (sum of weights for each year)
    total_weight = df_weights.sum(axis=1)
    
    # Avoid division by zero by replacing 0 with NaN
    total_weight_sanitized = total_weight.replace(0, np.nan)

    mean_x_cbsa = weighted_x_sum / total_weight_sanitized
    
    return mean_x_cbsa.dropna()

def pad_to_equal_length(seqs, align="right"):
    """Pads a list of 1D arrays/lists with NaNs to equal length.
    
    Args:
        seqs: List of 1D arrays/lists to pad
        align: "right" keeps the tail aligned (useful to align by latest year),
               "left" keeps the head aligned
               
    Returns:
        2D numpy array with padded sequences
    """
    if not seqs:
        return np.empty((0, 0))
    max_len = max(len(s) for s in seqs)
    padded = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=float)
        L = len(s)
        if L == 0:
            continue
        if align == "left":
            padded[i, :L] = s
        else:  # align right
            padded[i, -L:] = s
    return padded

def transform_x_to_gamma_dynamic(x_traj, p_t, l=2):
    """Transforms x_t trajectory into a growth rate (gamma) trajectory using a dynamic p_t.
    
    Args:
        x_traj: Agent belief trajectory
        p_t: Dynamic environmental predictability series
        l: Model parameter (default=2)
        
    Returns:
        numpy array: Growth rate trajectory
    """
    x_traj = np.asarray(x_traj)
    
    # Handle cases where p_t might be None, empty, or not an array
    if p_t is None or (hasattr(p_t, '__len__') and len(p_t) == 0):
        raise ValueError("p_t is None or empty")
    
    p_t = np.asarray(p_t)
    
    # Ensure p_t and x_traj have the same length for element-wise operations
    if len(p_t) != len(x_traj):
        raise ValueError(f"Shape mismatch: p_t has length {len(p_t)} but x_traj has length {len(x_traj)}")

    x_clipped = np.clip(x_traj, 1e-9, 1 - 1e-9)
    return np.log(l) + p_t * np.log(x_clipped) + (1 - p_t) * np.log(1 - x_clipped)

def compute_trajectory_statistics(trajectories, align="right"):
    """Compute mean and confidence intervals for a list of trajectories.
    
    Args:
        trajectories: List of 1D arrays/lists 
        align: Alignment for padding ("right" or "left")
        
    Returns:
        tuple: (mean_traj, std_traj, ci_low, ci_high) all as numpy arrays
    """
    if not trajectories:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    padded = pad_to_equal_length(trajectories, align=align)
    mean_traj = np.nanmean(padded, axis=0)
    std_traj = np.nanstd(padded, axis=0)
    ci_low = np.nanpercentile(padded, 2.5, axis=0)
    ci_high = np.nanpercentile(padded, 97.5, axis=0)
    
    return mean_traj, std_traj, ci_low, ci_high

def calculate_divergence(p, x):
    """
    Computes the KL divergence between two Bernoulli distributions parameterized by p and x.
    Formula: D_KL(P || Q) = p*log(p/x) + (1-p)*log((1-p)/(1-x))
    """
    # Handle cases where p or x might be None, empty, or not arrays
    if p is None or (hasattr(p, '__len__') and len(p) == 0):
        raise ValueError("p is None or empty")
    if x is None or (hasattr(x, '__len__') and len(x) == 0):
        raise ValueError("x is None or empty")
    
    p = np.asarray(p)
    x = np.asarray(x)

    # Clip values to avoid log(0) or division by zero.
    p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
    x_clipped = np.clip(x, 1e-9, 1 - 1e-9)

    term1 = p_clipped * np.log(p_clipped / x_clipped)
    term2 = (1 - p_clipped) * np.log((1 - p_clipped) / (1 - x_clipped))
    
    return term1 + term2 