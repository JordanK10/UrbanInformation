import os
import pickle
import numpy as np
import arviz as az
import pandas as pd
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
from ssm_model import fit_ssm_random_walk
import traceback

class MockInferenceData:
    """A mock object to replicate the structure of arviz.InferenceData for direct calculation."""
    def __init__(self, trajectory):
        # Add extra dimensions to mimic shape of pymc output (chains, draws, time)
        # The calling code expects to do .mean(axis=(0,1))
        self.posterior = {"x": np.array(trajectory)[np.newaxis, np.newaxis, :]}

def calculate_l_t_series(cbsa_df, p_t_series, p_t_years):
    """
    Finds a representative high growth rate for each year in a CBSA and calculates l directly.
    The logic assumes that for the optimal agent (robust max), y = ln(l*p).
    Therefore, l = exp(y) / p.
    """
    # Step 1: Collect all valid growth rates for each year.
    growth_rates_by_year = {}
    for _, group in cbsa_df.groupby('block_group_fips'):
        group = group.sort_values('year')
        if len(group) > 1 and not (group['mean_income'] <= 0).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                log_income = np.log(group['mean_income'].values)
            y_values = np.diff(log_income)
            years = group['year'].values[1:]
            
            for i, year in enumerate(years):
                if np.isfinite(y_values[i]):
                    if year not in growth_rates_by_year:
                        growth_rates_by_year[year] = []
                    growth_rates_by_year[year].append(y_values[i])
                        
    l_t_dict = {}
    p_t_dict = dict(zip(p_t_years, p_t_series))
    
    # Step 2: For each year, find the representative growth rate and solve for l.
    for year, rates in growth_rates_by_year.items():
        # Require at least 5 data points for a minimally stable std dev
        if year in p_t_dict and len(rates) >= 5:
            
            rates_arr = np.array(rates)
            mean_growth = np.mean(rates_arr)
            std_growth = np.std(rates_arr)
            
            # Define the upper bound for the search
            threshold = mean_growth + 2.5 * std_growth
            
            # Sort rates in descending order and find the first one within the threshold
            # We are only interested in positive growth rates for defining the optimal "win"
            sorted_rates = sorted([r for r in rates if r > 0], reverse=True)
            
            representative_gamma = None
            if sorted_rates:
                for rate in sorted_rates:
                    if rate <= threshold:
                        representative_gamma = rate
                        break
            
            # If we found a representative rate, calculate l directly
            if representative_gamma is not None:
                p = p_t_dict[year]
                if p > 1e-6: # Avoid division by zero or near-zero
                    l_val = np.exp(representative_gamma) / p
                    l_t_dict[year] = l_val
    
    print(f"    Calculated l_t series for {len(l_t_dict)} years using direct method.")
    return l_t_dict

def calculate_x_trajectory_directly(y_values, p_t_series, year, l_t_dict, **kwargs):
    """
    Directly calculate the x trajectory from income growth rates using a time-varying l.
    This function mimics the interface of fit_ssm_dynamic_p_model for easy swapping.
    
    Args:
        y_values: Time series of income growth rates (log differences).
        p_t_series: Not used in this calculation, but kept for compatibility.
        year: The full list of years for the original income data points for a block group.
        l_t_dict: A dictionary mapping year to the calculated l for that year.
        **kwargs: Catches other arguments for compatibility.

    Returns:
        tuple: (MockInferenceData, loss) where loss is always 0.
    """
    raw_x_trajectory = []
    # y_values are diffs, so they correspond to the years from the second year onward.
    y_years = year[1:]
    
    for i, y in enumerate(y_values):
        current_year = y_years[i]
        # Use the l for the current year, or default to 2.0 if not found for robustness.
        l = l_t_dict.get(current_year, 2.0)
        
        if y > 0:
            # For wins, ln(y'/y) = ln(l*x) => x = exp(y)/l
            x_val = np.exp(y) / l
        else:
            # For losses, ln(y'/y) = ln(l*(1-x)) => x = 1 - exp(y)/l
            x_val = 1 - np.exp(y) / l
        
        # Clip to ensure it's a valid probability
        x_val = np.clip(x_val, 0.01, 0.99)
        raw_x_trajectory.append(x_val)

    # Apply smoothing to the raw, noisy trajectory to represent a more realistic evolution of belief
    if len(raw_x_trajectory) > 0:
        raw_series = pd.Series(raw_x_trajectory, index=y_years)
        # A 3-year rolling window provides a good balance of smoothing and responsiveness.
        # min_periods=1 ensures that we get output for the start of the series.
        smoothed_x = raw_series.rolling(window=3, min_periods=1).mean().values
    else:
        smoothed_x = []
        
    # ** User-defined Theoretical Prior **
    # If the environment is unfavorable (p < 0.5) but the agent won on the first turn,
    # their belief should also be < 0.5. If the raw calculation violates this,
    # it suggests we've misinterpreted their entire stance, so we flip the whole trajectory.
    if p_t_series is not None and len(p_t_series) > 0 and len(raw_x_trajectory) > 0:
        if p_t_series[0] < 0.5 and y_values[0] > 0 and raw_x_trajectory[0] > 0.5:
            smoothed_x = 1 - smoothed_x
            
    # For the 'direct' method, we return this as the final result.
    # For the 'vi' method, this serves as the initial guess.
    return smoothed_x

INITIAL_YEAR = 2014
FINAL_YEAR = 2023

def sanitize_filename(filename):
    """Removes or replaces characters from a string to make it a valid filename."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

def calculate_p_t_series(data_df, group_key, group_name):
    """
    Calculate p_t series for a given geographic level (CBSA, county, or ZIP).
    
    Args:
        data_df: DataFrame with block group data
        group_key: Column name to group by ('block_group_fips', 'county_fips', or 'closest_zip')
        group_name: Name for logging purposes
    
    Returns:
        tuple: (p_t_series, sorted_years) or (None, None) if no valid data
    """
    
    yearly_wins = {}
    yearly_totals = {}
    
    # Group by the specified key and calculate p_t
    for group_id, group in data_df.groupby(group_key):
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
    if not sorted_years:
        print(f"      No valid years for {group_name}")
        return None, None
    
    p_t_series = np.array([yearly_wins[year] / yearly_totals[year] for year in sorted_years])
    # Ensure p_t is within (0, 1) for logit transform
    p_t_series = np.clip(p_t_series, 1e-5, 1 - 1e-5)
    
    print(f"      {group_name}: {len(sorted_years)} years, p_t range: {p_t_series.min():.3f} to {p_t_series.max():.3f}")
    return p_t_series, sorted_years

def process_block_group_hierarchical(bg_fips, y_values, year, population_ts, 
                                   p_t_cbsa, p_t_years_cbsa,
                                   p_t_county, p_t_years_county,
                                   p_t_zip, p_t_years_zip,
                                   l_t_dict,
                                   method='vi'):
    """
    Fits the dynamic p SSM model for a single block group using all three p_t series.
    """
    warnings.simplefilter("ignore", UserWarning)
    
    results = {
        "block_group_fips": bg_fips,
        "population_ts": population_ts,
        "year": year,
        "status": "success"
    }
    
    # Process CBSA level
    if p_t_cbsa is not None and p_t_years_cbsa is not None:
        try:
            bg_p_t_years = year[1:]
            p_t_map = pd.Series(p_t_cbsa, index=p_t_years_cbsa)
            aligned_p_t = p_t_map.reindex(bg_p_t_years).values
            
            if not np.any(np.isnan(aligned_p_t)):
                l_t_series = np.array([l_t_dict.get(y, 2.0) for y in bg_p_t_years])

                # --- Two-Step Inference Process ---
                # 1. Get the initial guess from the direct/smoothed method
                init_x_traj = calculate_x_trajectory_directly(
                    y_values, aligned_p_t, year, l_t_dict
                )
                
                # 2. Run the desired inference method
                if method == 'direct':
                    idata, loss = MockInferenceData(init_x_traj), 0
                else: # Default to 'vi'
                    idata, loss = fit_ssm_random_walk(
                        y_values, l_t_series, init_x_traj, n_samples=20000
                    )

                results["x_mean_trajectory_cbsa"] = idata.posterior["x"].mean(axis=(0, 1))
                results["loss_cbsa"] = loss
        except Exception:
            results["status_cbsa"] = f"Failed: {traceback.format_exc()}"

    # Process County level (similar logic)
    if p_t_county is not None and p_t_years_county is not None:
        try:
            bg_p_t_years = year[1:]
            p_t_map = pd.Series(p_t_county, index=p_t_years_county)
            aligned_p_t = p_t_map.reindex(bg_p_t_years).values

            if not np.any(np.isnan(aligned_p_t)):
                l_t_series = np.array([l_t_dict.get(y, 2.0) for y in bg_p_t_years])

                init_x_traj = calculate_x_trajectory_directly(
                    y_values, aligned_p_t, year, l_t_dict
                )

                if method == 'direct':
                    idata, loss = MockInferenceData(init_x_traj), 0
                else:
                    idata, loss = fit_ssm_random_walk(
                        y_values, l_t_series, init_x_traj, n_samples=20000
                    )
                
                results["x_mean_trajectory_county"] = idata.posterior["x"].mean(axis=(0, 1))
                results["loss_county"] = loss
        except Exception:
            results["status_county"] = f"Failed: {traceback.format_exc()}"
    
    # Process ZIP level (similar logic)
    if p_t_zip is not None and p_t_years_zip is not None:
        try:
            bg_p_t_years = year[1:]
            p_t_map = pd.Series(p_t_zip, index=p_t_years_zip)
            aligned_p_t = p_t_map.reindex(bg_p_t_years).values

            if not np.any(np.isnan(aligned_p_t)):
                l_t_series = np.array([l_t_dict.get(y, 2.0) for y in bg_p_t_years])

                init_x_traj = calculate_x_trajectory_directly(
                    y_values, aligned_p_t, year, l_t_dict
                )

                if method == 'direct':
                    idata, loss = MockInferenceData(init_x_traj), 0
                else:
                    idata, loss = fit_ssm_random_walk(
                        y_values, l_t_series, init_x_traj, n_samples=20000
                    )

                results["x_mean_trajectory_zip"] = idata.posterior["x"].mean(axis=(0, 1))
                results["loss_zip"] = loss
        except Exception:
            results["status_zip"] = f"Failed: {traceback.format_exc()}"

    # Only count as a success if at least one level succeeded
    if "x_mean_trajectory_cbsa" not in results and \
       "x_mean_trajectory_county" not in results and \
       "x_mean_trajectory_zip" not in results:
        results["status"] = "failure"
        
    return results

def main(method='vi'):
    """
    Main function to run the hierarchical dynamic-p analysis.
    """
    # Load ACS data with ZIP codes
    acs_data_path = "data/cbsa_acs_data.pkl"
    zip_data_path = "data/blockgroups_with_zips_temporal.pkl"
    
    if not os.path.exists(acs_data_path):
        print(f"Error: ACS data file not found at '{acs_data_path}'")
        print("Please run acs_data_retrieval.py first.")
        return
    
    if not os.path.exists(zip_data_path):
        print(f"Error: ZIP data file not found at '{zip_data_path}'")
        print("Please run match_blockgroups_to_zips3.py first.")
        return

    print("Loading data...")
    with open(acs_data_path, 'rb') as f:
        cbsa_acs_data = pickle.load(f)
    
    with open(zip_data_path, 'rb') as f:
        zip_matched_data = pickle.load(f)

    output_dir = "analysis_output_hierarchical"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Main processing loop for each CBSA ---
    for cbsa_name, cbsa_df in cbsa_acs_data.items():
        sanitized_cbsa_name = sanitize_filename(cbsa_name)
        output_path = os.path.join(output_dir, f"{sanitized_cbsa_name}.pkl")

        if os.path.exists(output_path):
            print(f"Skipping already processed CBSA: {cbsa_name}")
            continue

        print(f"\nProcessing {cbsa_name}...")
        
        # Get the ZIP-matched data for this CBSA
        if cbsa_name not in zip_matched_data:
            print(f"  No ZIP data found for {cbsa_name}, skipping...")
            continue
        
        zip_cbsa_df = zip_matched_data[cbsa_name]
        
        # Add county FIPS column if not present
        if 'county_fips' not in zip_cbsa_df.columns:
            zip_cbsa_df['county_fips'] = zip_cbsa_df['block_group_fips'].str[:5]
        
        # Filter to only include block groups with ZIP codes
        valid_df = zip_cbsa_df[zip_cbsa_df['closest_zip'].notna()].copy()
        
        if len(valid_df) == 0:
            print(f"  No valid ZIP-matched data for {cbsa_name}, skipping...")
            continue
        
        print(f"  Processing {len(valid_df)} ZIP-matched block groups...")
        
        # --- Calculate p_t series for all three levels ---
        
        # 1. CBSA level
        p_t_cbsa, p_t_years_cbsa = calculate_p_t_series(valid_df, 'block_group_fips', 'CBSA')
        
        # Calculate the time-varying l for the entire CBSA
        l_t_cbsa_dict = {}
        if p_t_cbsa is not None:
            l_t_cbsa_dict = calculate_l_t_series(valid_df, p_t_cbsa, p_t_years_cbsa)
        
        # 2. County level (group by county within CBSA)
        p_t_county_dict = {}
        p_t_years_county_dict = {}
        
        for county_fips, county_group in valid_df.groupby('county_fips'):
            p_t_county, p_t_years_county = calculate_p_t_series(county_group, 'block_group_fips', f'County {county_fips}')
            if p_t_county is not None:
                p_t_county_dict[county_fips] = p_t_county
                p_t_years_county_dict[county_fips] = p_t_years_county
        
        # 3. ZIP level (group by ZIP within CBSA)
        p_t_zip_dict = {}
        p_t_years_zip_dict = {}
        
        for zip_code, zip_group in valid_df.groupby('closest_zip'):
            p_t_zip, p_t_years_zip = calculate_p_t_series(zip_group, 'block_group_fips', f'ZIP {zip_code}')
            if p_t_zip is not None:
                p_t_zip_dict[zip_code] = p_t_zip
                p_t_years_zip_dict[zip_code] = p_t_years_zip
        
        print(f"    County p_t series: {len(p_t_county_dict)}")
        print(f"    ZIP p_t series: {len(p_t_zip_dict)}")
        
        # --- Prepare tasks for joblib ---
        tasks = []
        
        for bg_fips, group in valid_df.groupby('block_group_fips'):
            group = group.sort_values('year')
            
            # We need at least 2 data points to calculate a growth rate
            if len(group) < 2 or (group['mean_income'] <= 0).any():
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                log_income = np.log(group['mean_income'].values)
            y_values = np.diff(log_income)

            # Ensure there are no NaNs or Infs in the growth rates
            if not np.all(np.isfinite(y_values)):
                continue

            population_ts = group['population'].values.tolist()
            year = group['year'].values.tolist()
            
            # Get the appropriate p_t series for this block group
            county_fips = group['county_fips'].iloc[0]
            zip_code = group['closest_zip'].iloc[0]
            
            p_t_county = p_t_county_dict.get(county_fips)
            p_t_years_county = p_t_years_county_dict.get(county_fips)
            p_t_zip = p_t_zip_dict.get(zip_code)
            p_t_years_zip = p_t_years_zip_dict.get(zip_code)
            
            tasks.append(delayed(process_block_group_hierarchical)(
                bg_fips, y_values, year, population_ts,
                p_t_cbsa, p_t_years_cbsa,
                p_t_county, p_t_years_county,
                p_t_zip, p_t_years_zip,
                l_t_cbsa_dict,
                method
            ))
        
        if not tasks:
            print(f"  No valid block groups for {cbsa_name}. Skipping.")
            continue
            
        print(f"  Found {len(tasks)} valid block groups. Starting parallel processing...")

        results = Parallel(n_jobs=-1)(
            tqdm(tasks, desc=f"Analyzing {cbsa_name}", total=len(tasks))
        )

        successful_results = [res for res in results if res and res["status"] == "success"]

        with open(output_path, 'wb') as f:
            pickle.dump(successful_results, f)
        
        print(f"  Saved {len(successful_results)} results for {cbsa_name} to {output_path}")

if __name__ == "__main__":
    main(method='vi') 