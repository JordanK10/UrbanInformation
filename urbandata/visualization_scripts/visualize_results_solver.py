import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import warnings
import re
import pandas as pd
from scipy.optimize import root_scalar
from dynamic_p_utils import calculate_dynamic_p, sanitize_filename, pad_to_equal_length, transform_x_to_gamma_dynamic, compute_trajectory_statistics, calculate_divergence

INITIAL_YEAR = 2014
FINAL_YEAR = 2023
PLOT_INDIVIDUAL = True

def solve_for_l(gamma, p):
    """Numerically solves for l given gamma and p based on the Kelly growth equation."""
    # Objective function: f(l) = log(l) + p*log(p) + (1-p)*log((1-p)/(l-1)) - gamma = 0
    def objective(l, gamma, p):
        # This equation is only defined for l > 0 and 0 < p < 1.
        if l <= 0 or p <= 0 or p >= 1:
            return 1e9  # Return a large number for invalid inputs to guide the solver
        term1 = np.log(l)
        term2 = p * np.log(p)
        term3 = (1 - p) * np.log((1 - p) / (l - 1))
        return term1 + term2 + term3 - gamma

    try:
        # Use a better bracket that contains the root
        # Based on the Kelly equation behavior, l should typically be between 1.1 and 10
        # We'll use a wider bracket to be safe
        sol = root_scalar(objective, args=(gamma, p), bracket=[1.1, 10.0], method='brentq')
        if sol.converged:
            return sol.root
        else:
            print(f"  Warning: Solver did not converge for gamma={gamma:.4f}, p={p:.4f}")
            return None
    except Exception as e:
        print(f"  Error in solver: {e}")
        return None

def calculate_l_t_series(cbsa_df, p_t_series, p_t_years):
    """
    Calculate the time series of l values for a CBSA by finding the maximum income growth rate
    in each year and solving for l assuming optimal behavior (x = p).
    """
    l_t_dict = {}
    
    # Group by year and find the maximum growth rate in each year
    for year in p_t_years:
        year_data = cbsa_df[cbsa_df['year'] == year]
        
        if len(year_data) < 2:
            continue
            
        # Calculate growth rates for all block groups in this year
        max_growth_rate = -np.inf
        
        for bg_fips, group in year_data.groupby('block_group_fips'):
            if len(group) > 1 and not (group['mean_income'] <= 0).any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    log_income = np.log(group['mean_income'].values)
                y_values = np.diff(log_income)
                
                if np.all(np.isfinite(y_values)):
                    # Find the maximum growth rate for this block group
                    max_bg_growth = np.max(y_values)
                    if max_bg_growth > max_growth_rate:
                        max_growth_rate = max_bg_growth
        
        # If we found a valid growth rate, solve for l
        if max_growth_rate > -np.inf:
            # Get the p value for this year
            year_idx = np.where(p_t_years == year)[0][0]
            p_val = p_t_series[year_idx]
            
            # Solve for l assuming the agent with max growth rate had x = p (optimal behavior)
            # For optimal behavior: gamma = log(l) + p*log(p) + (1-p)*log((1-p)/(l-1))
            # We'll use the max growth rate as an approximation of gamma
            l_val = solve_for_l(gamma=max_growth_rate, p=p_val)
            
            if l_val is not None:
                l_t_dict[year] = l_val
    
    return l_t_dict

def get_short_name(cbsa_name):
    """Creates a short, safe filename from a CBSA name. e.g., 'Albuquerque, NM' -> 'AlbNM'"""
    parts = re.split(r'[ ,-]+', cbsa_name)
    city = parts[0][:3]
    state = parts[-1]
    return f"{city}{state}"

def load_cbsa_results(result_path):
    """Loads a single .pkl result file."""
    with open(result_path, 'rb') as f:
        return pickle.load(f)






def visualize_dynamic_p_results(cbsa_name, cbsa_results, cbsa_df, output_dir="output_dynamic_p_hierarchical",plot=False):
    """
    Generates and saves a 2x2 plot for a single CBSA from the dynamic-p analysis.
    Handles discontinuous time series by splitting long trajectories and grouping all segments.
    """
    if not cbsa_results:
        print(f"No results found for {cbsa_name}. Skipping visualization.")
        return None

    # --- 1. Recalculate p_t and prepare data structures ---
    p_t_series, p_t_years = calculate_dynamic_p(cbsa_df)

    # Calculate the time series of l values for this CBSA
    l_t_dict = calculate_l_t_series(cbsa_df, p_t_series, p_t_years)
    
    # Calculate the optimal growth rate if an agent's belief perfectly matched the environment
    # Use the mean l value for the optimal gamma calculation
    mean_l = np.mean(list(l_t_dict.values())) if l_t_dict else 2.0
    optimal_gamma = np.log(mean_l) + p_t_series * np.log(p_t_series) + (1 - p_t_series) * np.log(1 - p_t_series)

    p_t_map = pd.Series(p_t_series, index=p_t_years)

    # --- 2. Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(f"Comprehensive Analysis for {cbsa_name}", fontsize=20)

    # --- Panel 1: Environment (p_t series) ---
    ax = axes[0, 0]
    ax.plot(p_t_years, p_t_series, marker='o', linestyle='-', color='blue')
    ax.set_title("Environmental Predictability ($p_t$)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("City-Wide Predictability ($p_t$)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True)

    # --- Group trajectories into three distinct cohorts for accurate averaging ---
    pre_trajs_x, post_split_trajs_x, post_native_trajs_x = [], [], []
    pre_trajs_gamma, post_split_trajs_gamma, post_native_trajs_gamma = [], [], []
    pre_trajs_divergence, post_split_trajs_divergence, post_native_trajs_divergence = [], [], []
    pre_years, post_split_years, post_native_years = None, None, None



    for res in cbsa_results:
        print(res.keys())
        x_traj = res["x_mean_trajectory_cbsa"]
        bg_years = res["year"][1:]  # Years corresponding to y_values/x_trajectory

        aligned_p_t = p_t_map.reindex(bg_years).values
        if np.isnan(aligned_p_t).any():
            continue # Skip if alignment fails

        gamma_traj = transform_x_to_gamma_dynamic(x_traj, aligned_p_t)
        divergence_traj = calculate_divergence(aligned_p_t, x_traj)

        # A trajectory is "long" if its y-values span from pre-2020 to post-2020
        is_long_trajectory = bg_years[0] < 2020 and bg_years[-1] >= 2020

        if is_long_trajectory:
            try:
                # Split at the first year >= 2020
                split_idx = next(i for i, year in enumerate(bg_years) if year >= 2020)
                
                pre_trajs_x.append(x_traj[:split_idx])
                pre_trajs_gamma.append(gamma_traj[:split_idx])
                pre_trajs_divergence.append(divergence_traj[:split_idx])
                if pre_years is None: pre_years = bg_years[:split_idx]
                
                post_split_trajs_x.append(x_traj[split_idx:])
                post_split_trajs_gamma.append(gamma_traj[split_idx:])
                post_split_trajs_divergence.append(divergence_traj[split_idx:])
                if post_split_years is None: post_split_years = bg_years[split_idx:]
            except StopIteration:
                continue # Trajectory does not span 2020, should be caught by other conditions
        
        elif bg_years[-1] < 2020: # Purely pre-2020
            pre_trajs_x.append(x_traj)
            pre_trajs_gamma.append(gamma_traj)
            pre_trajs_divergence.append(divergence_traj)
            if pre_years is None: pre_years = bg_years
        
        elif bg_years[0] >= 2020: # Purely post-2020
            post_native_trajs_x.append(x_traj)
            post_native_trajs_gamma.append(gamma_traj)
            post_native_trajs_divergence.append(divergence_traj)
            if post_native_years is None: post_native_years = bg_years

    # --- Panel 2: Agent Belief (x trajectories) ---
    ax = axes[0, 1]
    # Plot all individual raw trajectories first, in their original, unsplit form
    for res in cbsa_results:
        ax.plot(res["year"][1:], res["x_mean_trajectory_cbsa"], color='crimson', alpha=0.05)

    # Calculate statistics for each period and then concatenate for a single mean plot
    mean_x_segments, ci_low_x_segments, ci_high_x_segments, year_segments = [], [], [], []
    std_x_segments = []
    
    if pre_trajs_x:
        mean_traj, std_traj, ci_low, ci_high = compute_trajectory_statistics(pre_trajs_x, align="right")
        mean_x_segments.append(mean_traj)
        std_x_segments.append(std_traj)
        ci_low_x_segments.append(ci_low)
        ci_high_x_segments.append(ci_high)
        year_segments.append(pre_years)

    # Combine post-2020 cohorts for a single, representative mean
    if post_split_trajs_x:
        arr_split = pad_to_equal_length(post_split_trajs_x, align="right")
        # Ensure native array is 2D for concatenation, even if empty
        arr_native = pad_to_equal_length(post_native_trajs_x, align="right") if post_native_trajs_x else np.empty((0, arr_split.shape[1]))


        # For years where both cohorts have data, combine them for averaging
        overlap_len = min(arr_split.shape[1], arr_native.shape[1])
        if overlap_len > 0:
            split_overlap_data = arr_split[:, -overlap_len:]
            native_overlap_data = arr_native[:, -overlap_len:]
            combined_overlap = np.concatenate((split_overlap_data, native_overlap_data), axis=0)
            
            # Calculate stats for the combined overlapping part (nan-aware)
            mean_overlap = np.nanmean(combined_overlap, axis=0)
            std_overlap = np.nanstd(combined_overlap, axis=0)
            ci_low_overlap = np.nanpercentile(combined_overlap, 2.5, axis=0)
            ci_high_overlap = np.nanpercentile(combined_overlap, 97.5, axis=0)
        else: # Handle case where there's no overlap (only split trajectories exist post-2020)
            mean_overlap, std_overlap, ci_low_overlap, ci_high_overlap = [], [], [], []


        # Get stats for the non-overlapping part (from the longer, split trajectories)
        non_overlap_len = arr_split.shape[1] - overlap_len
        if non_overlap_len > 0:
            non_overlap_data = arr_split[:, :non_overlap_len]
            mean_non_overlap = np.nanmean(non_overlap_data, axis=0)
            std_non_overlap = np.nanstd(non_overlap_data, axis=0)
            ci_low_non_overlap = np.nanpercentile(non_overlap_data, 2.5, axis=0)
            ci_high_non_overlap = np.nanpercentile(non_overlap_data, 97.5, axis=0)
            
            # Stitch the non-overlapping and overlapping parts back together
            final_mean_post = np.concatenate((mean_non_overlap, mean_overlap))
            final_std_post = np.concatenate((std_non_overlap, std_overlap))
            final_ci_low_post = np.concatenate((ci_low_non_overlap, ci_low_overlap))
            final_ci_high_post = np.concatenate((ci_high_non_overlap, ci_high_overlap))
        else:
            final_mean_post = mean_overlap
            final_std_post = std_overlap
            final_ci_low_post = ci_low_overlap
            final_ci_high_post = ci_high_overlap

        if len(final_mean_post) > 0:
            mean_x_segments.append(final_mean_post)
            std_x_segments.append(final_std_post)
            ci_low_x_segments.append(final_ci_low_post)
            ci_high_x_segments.append(final_ci_high_post)
            year_segments.append(post_split_years)

    # Concatenate the segments to plot a single, continuous mean line
    if year_segments:
        full_mean_x = np.concatenate(mean_x_segments)
        full_std_x = np.concatenate(std_x_segments)
        full_ci_low_x = np.concatenate(ci_low_x_segments)
        full_ci_high_x = np.concatenate(ci_high_x_segments)
        full_years_x = np.concatenate(year_segments)
        if plot:
            ax.plot(full_years_x, full_mean_x, color='crimson', linewidth=2.5, label='Mean Belief')
            ax.fill_between(full_years_x, full_ci_low_x, full_ci_high_x, color='crimson', alpha=0.2, label='95% CI')
    if plot:
        ax.set_title("Inferred Agent Belief ($x_t$)", fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Agent Belief ($x_t$)", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend()

    # --- Panel 3: Economic Outcome (gamma trajectories) ---
    ax = axes[1, 0]
    # Plot all individual raw trajectories first, in their original, unsplit form
    for res in cbsa_results:
        aligned_p_t = p_t_map.reindex(res["year"][1:]).values
        if not np.isnan(aligned_p_t).any():
            gamma_traj = transform_x_to_gamma_dynamic(res["x_mean_trajectory_cbsa"], aligned_p_t)
            if plot:
                ax.plot(res["year"][1:], gamma_traj, color='purple', alpha=0.05)

    # Calculate statistics for each period and then concatenate
    mean_g_segments, std_g_segments, ci_low_g_segments, ci_high_g_segments, year_g_segments = [], [], [], [], []
    std_g_segments = []
    mean_d_segments, std_d_segments, year_d_segments = [], [], []


    if pre_trajs_gamma:
        mean_traj, std_traj, ci_low, ci_high = compute_trajectory_statistics(pre_trajs_gamma, align="right")
        mean_g_segments.append(mean_traj)
        std_g_segments.append(std_traj)
        ci_low_g_segments.append(ci_low)
        ci_high_g_segments.append(ci_high)
        year_g_segments.append(pre_years)
    
    if pre_trajs_divergence:
        padded_pre_d = pad_to_equal_length(pre_trajs_divergence, align="right")
        mean_d_segments.append(np.nanmean(padded_pre_d, axis=0))
        std_d_segments.append(np.nanstd(padded_pre_d, axis=0))
        year_d_segments.append(pre_years)

    # Combine post-2020 cohorts for a single, representative mean
    if post_split_trajs_gamma:
        arr_split = pad_to_equal_length(post_split_trajs_gamma, align="right")
        arr_native = pad_to_equal_length(post_native_trajs_gamma, align="right") if post_native_trajs_gamma else np.empty((0, arr_split.shape[1]))

        overlap_len = min(arr_split.shape[1], arr_native.shape[1])
        if overlap_len > 0:
            split_overlap_data = arr_split[:, -overlap_len:]
            native_overlap_data = arr_native[:, -overlap_len:]
            combined_overlap = np.concatenate((split_overlap_data, native_overlap_data), axis=0)

            mean_overlap = np.nanmean(combined_overlap, axis=0)
            std_overlap = np.nanstd(combined_overlap, axis=0)
            ci_low_overlap = np.nanpercentile(combined_overlap, 2.5, axis=0)
            ci_high_overlap = np.nanpercentile(combined_overlap, 97.5, axis=0)
        else:
            mean_overlap, std_overlap, ci_low_overlap, ci_high_overlap = [], [], [], []

        non_overlap_len = arr_split.shape[1] - overlap_len
        if non_overlap_len > 0:
            non_overlap_data = arr_split[:, :non_overlap_len]
            mean_non_overlap = np.nanmean(non_overlap_data, axis=0)
            std_non_overlap = np.nanstd(non_overlap_data, axis=0)
            ci_low_non_overlap = np.nanpercentile(non_overlap_data, 2.5, axis=0)
            ci_high_non_overlap = np.nanpercentile(non_overlap_data, 97.5, axis=0)
            
            final_mean_post = np.concatenate((mean_non_overlap, mean_overlap))
            final_std_post = np.concatenate((std_non_overlap, std_overlap))
            final_ci_low_post = np.concatenate((ci_low_non_overlap, ci_low_overlap))
            final_ci_high_post = np.concatenate((ci_high_non_overlap, ci_high_overlap))
        else:
            final_mean_post = mean_overlap
            final_std_post = std_overlap
            final_ci_low_post = ci_low_overlap
            final_ci_high_post = ci_high_overlap

        if len(final_mean_post) > 0:
            mean_g_segments.append(final_mean_post)
            std_g_segments.append(final_std_post)
            ci_low_g_segments.append(final_ci_low_post)
            ci_high_g_segments.append(final_ci_high_post)
            year_g_segments.append(post_split_years)
    
    # Combine post-2020 cohorts for divergence stats
    if post_split_trajs_divergence:
        arr_split_d = pad_to_equal_length(post_split_trajs_divergence, align="right")
        arr_native_d = pad_to_equal_length(post_native_trajs_divergence, align="right") if post_native_trajs_divergence else np.empty((0, arr_split_d.shape[1]))

        overlap_len_d = min(arr_split_d.shape[1], arr_native_d.shape[1])
        if overlap_len_d > 0:
            combined_overlap_d = np.concatenate((arr_split_d[:, -overlap_len_d:], arr_native_d[:, -overlap_len_d:]), axis=0)
            mean_overlap_d = np.nanmean(combined_overlap_d, axis=0)
            std_overlap_d = np.nanstd(combined_overlap_d, axis=0)
        else:
            mean_overlap_d, std_overlap_d = [], []

        non_overlap_len_d = arr_split_d.shape[1] - overlap_len_d
        if non_overlap_len_d > 0:
            mean_non_overlap_d = np.nanmean(arr_split_d[:, :non_overlap_len_d], axis=0)
            std_non_overlap_d = np.nanstd(arr_split_d[:, :non_overlap_len_d], axis=0)
            final_mean_post_d = np.concatenate((mean_non_overlap_d, mean_overlap_d))
            final_std_post_d = np.concatenate((std_non_overlap_d, std_overlap_d))
        else:
            final_mean_post_d = mean_overlap_d
            final_std_post_d = std_overlap_d

        if len(final_mean_post_d) > 0:
            mean_d_segments.append(final_mean_post_d)
            std_d_segments.append(final_std_post_d)
            year_d_segments.append(post_split_years)


    # Concatenate the segments to plot a single, continuous mean line
    if year_g_segments:
        full_mean_g = np.concatenate(mean_g_segments)
        full_std_g = np.concatenate(std_g_segments)
        full_ci_low_g = np.concatenate(ci_low_g_segments)
        full_ci_high_g = np.concatenate(ci_high_g_segments)
        full_years_g = np.concatenate(year_g_segments)
        if plot:
            ax.plot(full_years_g, full_mean_g, color='purple', linewidth=2.5, label='Mean Growth Rate')
            ax.fill_between(full_years_g, full_ci_low_g, full_ci_high_g, color='purple', alpha=0.2, label='95% CI')
    
    if year_d_segments:
        full_mean_d = np.concatenate(mean_d_segments)
        full_std_d = np.concatenate(std_d_segments)
        full_years_d = np.concatenate(year_d_segments)
    if plot:
        # Add the optimal growth rate plot for comparison
        ax.plot(p_t_years, optimal_gamma, color='black', linestyle='--', linewidth=2, label='Optimal Growth (x$_t$ = p$_t$)')

        ax.set_title("Resulting Growth Rate ($\gamma_t$)", fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Expected Growth Rate (nats/period)", fontsize=12)
        ax.legend()

        # --- Panel 4: Dynamic l Time Series ---
        ax = axes[1, 1]
        if l_t_dict:
            l_years = sorted(l_t_dict.keys())
            l_values = [l_t_dict[year] for year in l_years]
            ax.plot(l_years, l_values, marker='o', linestyle='-', color='green', linewidth=2)
            ax.axhline(mean_l, color='red', linestyle='--', label=f'Mean l: {mean_l:.2f}')
            ax.set_title("Dynamic l Time Series", fontsize=14)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Number of Outcomes (l)", fontsize=12)
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No l values calculated', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Dynamic l Time Series", fontsize=14)

        # --- 3. Save Figure ---
        short_name = get_short_name(cbsa_name)
        output_filename = os.path.join(output_dir, f"{short_name}_visualization.pdf")
        plt.savefig(output_filename, format='pdf')
        plt.close(fig)

    # --- 4. Package statistics for return ---
    # Use the most recent year's total population for the CBSA
    total_population = cbsa_df.groupby('year')['population'].sum().iloc[-1] if not cbsa_df.empty else 0
    
    # Package data into a dictionary
    stats = {
        'p_t_series': pd.Series(p_t_series, index=p_t_years),
        'mean_x_trajectory': pd.Series(full_mean_x, index=full_years_x) if 'full_mean_x' in locals() else pd.Series(dtype=float),
        'std_x_trajectory': pd.Series(full_std_x, index=full_years_x) if 'full_std_x' in locals() else pd.Series(dtype=float),
        'optimal_gamma': pd.Series(optimal_gamma, index=p_t_years),
        'mean_agent_gamma': pd.Series(full_mean_g, index=full_years_g) if 'full_mean_g' in locals() else pd.Series(dtype=float),
        'std_agent_gamma': pd.Series(full_std_g, index=full_years_g) if 'full_std_g' in locals() else pd.Series(dtype=float),
        'mean_divergence': pd.Series(full_mean_d, index=full_years_d) if 'full_mean_d' in locals() else pd.Series(dtype=float),
        'std_divergence': pd.Series(full_std_d, index=full_years_d) if 'full_std_d' in locals() else pd.Series(dtype=float),
        'mean_loss': np.mean(losses) if losses else None,
        'population': total_population
    }
    
    return stats



def main():
    """
    Main function to loop through all result files, generate a plot for each,
    and save it to a dedicated output folder.
    """
    results_dir = "analysis_output_hierarchical"
    output_dir = "output_dynamic_p_hierarchical"
    acs_data_path = "data/data_retrieval/cbsa_acs_data.pkl"
    
    # --- Load all necessary data ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(results_dir) or not os.listdir(results_dir):
        print(f"Error: Analysis results directory '{results_dir}' is empty.")
        return
    if not os.path.exists(acs_data_path):
        print(f"Error: ACS data file not found at '{acs_data_path}'")
        return
        
    with open(acs_data_path, 'rb') as f:
        cbsa_acs_data = pickle.load(f)

    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.pkl')])
    
    print(f"Found {len(result_files)} result files. Generating plots...")
    
    all_cbsa_statistics = {}

    for filename in tqdm(result_files, desc="Generating plots for each CBSA"):
        
        # Robustly find the matching key in the acs_data dictionary
        sanitized_filename_base = filename.replace('.pkl', '')
        cbsa_key = next((key for key in cbsa_acs_data if sanitize_filename(key) == sanitized_filename_base), None)
        
        if cbsa_key:
            result_path = os.path.join(results_dir, filename)
            cbsa_results = load_cbsa_results(result_path)
            cbsa_df = cbsa_acs_data[cbsa_key]
            
            # This function now returns a dictionary of stats
            stats = visualize_dynamic_p_results(cbsa_key, cbsa_results, cbsa_df, output_dir,plot=PLOT_INDIVIDUAL)
            if stats:
                all_cbsa_statistics[cbsa_key] = stats
        else:
            print(f"Warning: Could not find matching ACS data for result file {filename}")
    
    # Save the collected statistics
    output_pkl_path = "cbsa_summary_statistics.pkl"
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(all_cbsa_statistics, f)
    print(f"Successfully saved summary statistics for {len(all_cbsa_statistics)} CBSAs to {output_pkl_path}")

if __name__ == "__main__":
    main() 