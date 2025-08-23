import os
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
from ssm_model import fit_ssm_dynamic_p_model

def sanitize_filename(filename):
    """Removes or replaces characters from a string to make it a valid filename."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

def process_block_group(bg_fips, y_values, year, population_ts, p_t_series, p_t_years, method='vi'):
    """
    Fits the dynamic p SSM model for a single block group.
    """
    warnings.simplefilter("ignore", UserWarning)
    
    try:
        bg_p_t_years = year[1:]
        p_t_map = pd.Series(p_t_series, index=p_t_years)
        aligned_p_t = p_t_map.reindex(bg_p_t_years).values
        
        if np.isnan(aligned_p_t).any():
            return {"block_group_fips": bg_fips, "status": "failure", "error": "Could not align p_t series."}

        idata, loss = fit_ssm_dynamic_p_model(y_values, aligned_p_t, method=method, vi_steps=30000)
        x_mean_trajectory = idata.posterior["x"].mean(axis=(0, 1)).to_numpy()

        return {
            "block_group_fips": bg_fips,
            "x_mean_trajectory": x_mean_trajectory,
            "p_hat_frequentist": aligned_p_t,
            "population_ts": population_ts,
            "loss": loss,
            "year": year,
            "status": "success"
        }
    except Exception as e:
        return {"block_group_fips": bg_fips, "status": "failure", "error": str(e)}

def process_one_cbsa(cbsa_name, cbsa_df, output_dir, method='vi'):
    """
    Processes all block groups for a single CBSA.
    """
    sanitized_cbsa_name = sanitize_filename(cbsa_name)
    output_path = os.path.join(output_dir, f"{sanitized_cbsa_name}.pkl")

    if os.path.exists(output_path):
        # print(f"Skipping already processed CBSA: {cbsa_name}")
        return f"Skipped: {cbsa_name} (already processed)"

    # --- Pre-calculate the dynamic p_t series for the entire CBSA ---
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
                years = group['year'].values[1:]
                pop_end = group['population'].values[1:]
                for i, year in enumerate(years):
                    weight = float(pop_end[i]) if np.isfinite(pop_end[i]) and pop_end[i] > 0 else 0.0
                    if year not in yearly_totals:
                        yearly_totals[year] = 0.0
                        yearly_wins[year] = 0.0
                    yearly_totals[year] += weight
                    if y_values[i] > 0:
                        yearly_wins[year] += weight
    
    sorted_years = [yr for yr in sorted(yearly_totals.keys()) if yearly_totals[yr] > 0]
    if not sorted_years:
        return f"Skipped: {cbsa_name} (no valid years for p_t)"
        
    p_t_series = np.array([yearly_wins[year] / yearly_totals[year] for year in sorted_years])
    p_t_series = np.clip(p_t_series, 1e-5, 1 - 1e-5)

    # --- Prepare tasks for joblib ---
    tasks = []
    for bg_fips, group in cbsa_df.groupby('block_group_fips'):
        group = group.sort_values('year')
        if len(group) < 2 or (group['mean_income'] <= 0).any():
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            log_income = np.log(group['mean_income'].values)
        y_values = np.diff(log_income)

        if not np.all(np.isfinite(y_values)):
            continue

        population_ts = group['population'].values.tolist()
        year = group['year'].values.tolist()
        tasks.append(delayed(process_block_group)(bg_fips, y_values, year, population_ts, p_t_series, sorted_years, method))
    
    if not tasks:
        return f"Skipped: {cbsa_name} (no valid block groups)"
        
    # --- Run block group processing in parallel ---
    results = Parallel(n_jobs=-1)(tasks)

    successful_results = [res for res in results if res and res["status"] == "success"]

    with open(output_path, 'wb') as f:
        pickle.dump(successful_results, f)
    
    return f"Success: {cbsa_name} ({len(successful_results)} results)" 