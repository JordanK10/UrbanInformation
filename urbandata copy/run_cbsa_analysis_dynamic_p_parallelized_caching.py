import os
# Configure pytensor to use the clang++ compiler
# This is crucial for stability on macOS, especially with Conda environments.
os.environ.setdefault('CXX', '/opt/anaconda3/envs/priceinequality/bin/clang++')
import pickle
import numpy as np
import arviz as az
import pandas as pd
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
from ssm_model import fit_ssm_dynamic_p_model # Import the new model function
import traceback

INITIAL_YEAR = 2014
FINAL_YEAR = 2023

def sanitize_filename(filename):
    """Removes or replaces characters from a string to make it a valid filename."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

def process_block_group(bg_fips, y_values, year, population_ts, p_t_series, p_t_years, method='vi'):
    """
    Fits the dynamic p SSM model for a single block group.
    """
    warnings.simplefilter("ignore", UserWarning)
    
    try:
        # Align the CBSA-level p_t_series with the specific years of this block group.
        # The years for y_values start from the second year of the raw data.
        bg_p_t_years = year[1:]
        
        # Create a pandas Series for easy alignment
        p_t_map = pd.Series(p_t_series, index=p_t_years)
        
        # Reindex to match the block group's years. This handles any subset of years.
        aligned_p_t = p_t_map.reindex(bg_p_t_years).values
        
        # Check if alignment was successful
        if np.isnan(aligned_p_t).any():
            return {"block_group_fips": bg_fips, "status": "failure", "error": "Could not align p_t series."}

        #BUG Attempted to use lagged p_t_series, but it didn't work. - results in awful fits/crash
        # p_t_lag = np.concatenate([[0],p_t_series[:-1]])
        # idata_lag, loss_lag = fit_ssm_dynamic_p_model(y_values, p_t_lag,    method=method, vi_steps=30000)
        # x_mean_trajectory_lag = idata_lag.posterior["x"].mean(axis=(0, 1)).to_numpy()

        idata,     loss     = fit_ssm_dynamic_p_model(y_values, aligned_p_t, method=method, vi_steps=30000)
        
        # We no longer infer p, so we don't extract it.
        x_mean_trajectory     = idata.posterior["x"].mean(axis=(0, 1)).to_numpy()

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
        print(traceback.format_exc())
        return {"block_group_fips": bg_fips, "status": "failure", "error": str(e)}

def process_cbsa(cbsa_name, cbsa_df, output_dir, method):
    """
    Processes all block groups for a single CBSA, including p_t calculation
    and parallel inference.
    """
    sanitized_cbsa_name = sanitize_filename(cbsa_name)
    output_path = os.path.join(output_dir, f"{sanitized_cbsa_name}.pkl")

    if os.path.exists(output_path):
        # This will be printed by the worker process, so it's useful for debugging.
        # The main tqdm will still advance.
        return f"Skipping already processed CBSA: {cbsa_name}"

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
        return f"No valid data to calculate p_t for {cbsa_name}. Skipping."
        
    p_t_series = np.array([yearly_wins[year] / yearly_totals[year] for year in sorted_years])
    p_t_series = np.clip(p_t_series, 1e-5, 1 - 1e-5)

    # --- Prepare inner parallel tasks for block groups ---
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
        return f"No valid block groups for {cbsa_name}. Skipping."
        
    # This inner Parallel call will be managed by joblib to avoid oversubscription.
    # The progress bar will be cleaned up after completion due to leave=False.
    results = Parallel(n_jobs=-1)(
        tqdm(tasks, desc=f"Analyzing {cbsa_name}", total=len(tasks), leave=False)
    )

    successful_results = [res for res in results if res and res["status"] == "success"]

    with open(output_path, 'wb') as f:
        pickle.dump(successful_results, f)
    
    return f"Saved {len(successful_results)} results for {cbsa_name}"


def main(method='vi', max_block_groups_per_cbsa=None):
    """
    Main function to run the dynamic-p analysis in parallel across CBSAs.
    """
    acs_data_path = "cbsa_acs_data.pkl"
    if not os.path.exists(acs_data_path):
        print(f"Error: ACS data file not found at '{acs_data_path}'")
        print("Please run acs_data_retrieval.py first.")
        return

    with open(acs_data_path, 'rb') as f:
        cbsa_acs_data = pickle.load(f)

    output_dir = "analysis_output_dynamic_p" # New output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Create tasks for each CBSA to be processed in parallel ---
    cbsa_items = list(cbsa_acs_data.items())

    print(f"\nStarting parallel processing for {len(cbsa_items)} CBSAs...")
    
    # Outer parallel loop over cities
    results = Parallel(n_jobs=-1)(
        delayed(process_cbsa)(cbsa_name, cbsa_df, output_dir, method) 
        for cbsa_name, cbsa_df in tqdm(cbsa_items, desc="Processing CBSAs")
    )

    print("\n--- All CBSA processing complete ---")
    for result_msg in results:
        if result_msg:
            print(result_msg)


if __name__ == "__main__":
    main(method='vi') 