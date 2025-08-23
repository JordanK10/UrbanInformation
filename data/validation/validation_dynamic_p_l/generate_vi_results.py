import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import traceback

# Add parent directory to path to import ssm_model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_model import fit_ssm_random_walk, estimate_l_time_series_bayesian

# Suppress pymc warnings
import logging
logging.getLogger("pymc").setLevel(logging.ERROR)

def calculate_l_t_series_from_dummy_data(dummy_data):
    """
    DEPRECATED: Legacy heuristic method for calculating l_t_series. 
    This function is kept as a fallback for the new Bayesian approach.
    
    Calculate l_t_series from dummy data using the same logic as the real data analysis.
    Finds a representative high growth rate for each timestep and calculates l directly.
    The logic assumes that for the optimal agent (robust max), y = ln(l*p).
    Therefore, l = exp(y) / p.
    """
    # Extract data from dummy data
    vi_data = dummy_data['vi_data']
    p_t_series = dummy_data['p_t_series']
    l_t_series = dummy_data['l_t_series']
    
    # Step 1: Collect all valid growth rates for each timestep
    growth_rates_by_timestep = {}
    for agent_data in vi_data:
        y_values = agent_data['income_growth_rates']
        for i, y in enumerate(y_values):
            if np.isfinite(y):
                timestep = i  # y_values correspond to timesteps 1 to n
                if timestep not in growth_rates_by_timestep:
                    growth_rates_by_timestep[timestep] = []
                growth_rates_by_timestep[timestep].append(y)
    
    l_t_dict = {}
    
    # Step 2: For each timestep, find the representative growth rate and solve for l
    for timestep, rates in growth_rates_by_timestep.items():
        # Require at least 5 data points for a minimally stable std dev
        if timestep < len(p_t_series) and len(rates) >= 5:
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
                p = p_t_series[timestep]
                if p > 1e-6:  # Avoid division by zero or near-zero
                    l_val = np.exp(representative_gamma) / p
                    l_t_dict[timestep] = l_val
    
    print(f"    Calculated l_t series for {len(l_t_dict)} timesteps using direct method.")
    return l_t_dict

def calculate_frequentist_p(y_values_all_agents):
    """
    Calculate p using the frequentist approach: fraction of agents with positive growth rates.
    
    Args:
        y_values_all_agents: List of income growth rate arrays for all agents
        
    Returns:
        float: Estimated p value
    """
    total_observations = 0
    positive_observations = 0
    
    for y_values in y_values_all_agents:
        for y in y_values:
            if np.isfinite(y):  # Only count valid observations
                total_observations += 1
                if y > 0:
                    positive_observations += 1
    
    if total_observations == 0:
        return 0.5  # Default to 0.5 if no valid observations
    
    p_hat = positive_observations / total_observations
    return p_hat

def generate_initial_guess(y_values, p_estimated, l):
    """
    Generates a smart initial guess for the x trajectory using the direct calculation method.
    This includes smoothing in logit-space to avoid asymmetric bias from exponential amplification.
    Now uses the estimated p instead of true p.
    """
    raw_x_trajectory = []
    
    for y in y_values:
        if y > 0:
            x_val = np.exp(y) / l
        else:
            x_val = 1 - np.exp(y) / l
        
        x_val = np.clip(x_val, 0.01, 0.99)
        raw_x_trajectory.append(x_val)

    # Convert to logit space for symmetric smoothing
    # This avoids the asymmetric bias from exponential amplification
    raw_series = pd.Series(raw_x_trajectory)
    
    # Apply logit transformation: log(x / (1-x))
    # Add small epsilon to avoid log(0) or log(1)
    epsilon = 1e-6
    x_clipped = np.clip(raw_series.values, epsilon, 1 - epsilon)
    logit_x = np.log(x_clipped / (1 - x_clipped))
    
    # Apply smoothing in logit space
    logit_series = pd.Series(logit_x)
    smoothed_logit = logit_series.rolling(window=3, min_periods=1).mean().values
    
    # Transform back to probability space
    smoothed_x = 1 / (1 + np.exp(-smoothed_logit))

    # Apply the user's theoretical prior on the initial state
    # Now using estimated p instead of true p
    # if p_estimated is not None and len(raw_x_trajectory) > 0:
    #     if p_estimated < 0.5 and y_values[0] > 0 and y[0] > 0.5:
    #         smoothed_x = 1 - smoothed_x
            
    return smoothed_x

def process_agent(agent_data, p_estimated, l_t_dict, dummy_data):
    """
    Processes a single agent's data to fit a VI model.
    
    Args:
        agent_data: Dictionary containing agent's data.
        p_estimated: The estimated p value from frequentist calculation.
        l_t_dict: Dictionary mapping timestep to l value for that timestep.
        dummy_data: The full dummy data dictionary to access belief trajectories.
    
    Returns:
        A dictionary containing the results for the agent.
    """
    agent_id = agent_data['agent_id']
    true_belief = agent_data['initial_belief']
    
    # Get income growth rates and p_t series
    y_values = agent_data['income_growth_rates']
    
    # Get true belief trajectory from dummy data
    true_belief_trajectory = None
    belief_trajectories = dummy_data['agent_trajectories']['beliefs']
    agent_idx = None
    for j, data in enumerate(dummy_data['vi_data']):
        if data['agent_id'] == agent_id:
            agent_idx = j
            break
    
    if agent_idx is not None:
        true_belief_trajectory = belief_trajectories[agent_idx, :]
    
    try:
        # --- New Two-Step Inference Process ---
        # 1. Generate the smart initial guess for the trajectory using estimated p and dynamic l
        # Create l_t_series array for this agent's timesteps
        l_t_series = []
        for i in range(len(y_values)):
            timestep = i  # y_values correspond to timesteps 1 to n
            l_val = l_t_dict.get(timestep, 2.0)  # Default to 2.0 if not found
            l_t_series.append(l_val)
        
        l_t_series = np.array(l_t_series)
        
        # Use the first l value for the initial guess (or average if needed)
        l_for_initial_guess = l_t_series[0] if len(l_t_series) > 0 else 2.0
        
        init_x_traj = generate_initial_guess(y_values, p_estimated, l_for_initial_guess)

        # 2. Fit the robust VI model using the initial guess and dynamic l
        idata, loss = fit_ssm_random_walk(
            y_values=y_values,
            l_t_series=l_t_series,
            init_x_traj=init_x_traj,
            n_samples=7500  # Use a sufficient number of samples for good convergence
        )
        
        # Extract results
        x_mean_trajectory = idata.posterior["x"].mean(axis=(0, 1))
        
        return {
            "agent_id": agent_id,
            "status": "success",
            "loss": loss,
            "true_belief": true_belief,
            "fitted_belief": x_mean_trajectory[-1],
            "x_mean_trajectory_cbsa": x_mean_trajectory, # Keep key for compatibility with plotting script
            "true_belief_trajectory": true_belief_trajectory,
            "p_estimated": p_estimated,  # Store the estimated p used for fitting
            "l_t_series": l_t_series.tolist(),  # Store the l values used for this agent
        }
    except Exception as e:
        error_msg = traceback.format_exc()
        return {
            "agent_id": agent_id,
            "status": "failure",
            "error": error_msg,
            "true_belief": true_belief,
            "true_belief_trajectory": true_belief_trajectory,
            "p_estimated": p_estimated,  # Store the estimated p even for failures
            "l_t_series": l_t_series.tolist() if 'l_t_series' in locals() else None,
        }

def generate_vi_results_for_all_agents(dummy_data, vi_steps=10000, method='vi', seed=42):
    """
    Fit VI models to all agents in the dummy data.
    
    Args:
        dummy_data: Dummy data from generate_dummy_data.py
        vi_steps: Number of VI steps for fitting
        method: Fitting method ('vi' or 'mcmc')
        seed: Random seed for reproducibility
    
    Returns:
        List of VI fitting results for all agents
    """
    
    print(f"Fitting VI models to {len(dummy_data['vi_data'])} agents...")
    print(f"VI steps: {vi_steps}, Method: {method}")
    
    fitted_results = []
    
    for i, agent_data in enumerate(tqdm(dummy_data['vi_data'], desc="Fitting agents")):
        try:
            # Get income growth rates and p_t series
            y_values = agent_data['income_growth_rates']
            p_t_series = dummy_data['p_t_series']
            
            # Use the new two-step inference process with dynamic p and l
            # Create l_t_series array for this agent's timesteps
            l_t_series = []
            for i in range(len(y_values)):
                timestep = i  # y_values correspond to timesteps 1 to n
                l_val = l_t_dict.get(timestep, 2.0)  # Default to 2.0 if not found
                l_t_series.append(l_val)
            
            l_t_series = np.array(l_t_series)
            
            # Use the first l value for the initial guess (or average if needed)
            l_for_initial_guess = l_t_series[0] if len(l_t_series) > 0 else 2.0
            
            init_x_traj = generate_initial_guess(y_values, p_estimated, l_for_initial_guess)

            # Fit the robust VI model using the initial guess and dynamic l
            idata, loss = fit_ssm_random_walk(
                y_values=y_values,
                l_t_series=l_t_series,
                init_x_traj=init_x_traj,
                n_samples=7500  # Use a sufficient number of samples for good convergence
            )
            
            # Extract results
            x_mean_trajectory = idata.posterior["x"].mean(axis=(0, 1))
            
            # Get true belief trajectory from dummy data
            true_belief_trajectory = None
            belief_trajectories = dummy_data['agent_trajectories']['beliefs']
            agent_idx = None
            for j, data in enumerate(dummy_data['vi_data']):
                if data['agent_id'] == agent_data['agent_id']:
                    agent_idx = j
                    break
            
            if agent_idx is not None:
                true_belief_trajectory = belief_trajectories[agent_idx, :]
            
            # Store comprehensive results
            result = {
                'agent_id': agent_data['agent_id'],
                'true_belief': agent_data['initial_belief'],
                'fitted_belief': x_mean_trajectory[-1],
                'loss': loss,
                'x_mean_trajectory_cbsa': x_mean_trajectory,  # For compatibility with hierarchical analysis
                'x_trajectory': x_mean_trajectory,  # Original name
                'true_belief_trajectory': true_belief_trajectory,  # Full true belief trajectory
                'p_estimated': p_estimated,  # Store the estimated p used for fitting
                'l_t_series': l_t_series.tolist(),  # Store the l values used for this agent
                'status': 'success'
            }
            
            fitted_results.append(result)
            
        except Exception as e:
            print(f"    Agent {i+1} failed: {str(e)}")
            
            # Get true belief trajectory from dummy data for failed fits too
            true_belief_trajectory = None
            belief_trajectories = dummy_data['agent_trajectories']['beliefs']
            agent_idx = None
            for j, data in enumerate(dummy_data['vi_data']):
                if data['agent_id'] == agent_data['agent_id']:
                    agent_idx = j
                    break
            
            if agent_idx is not None:
                true_belief_trajectory = belief_trajectories[agent_idx, :]
            
            # Store failure result
            result = {
                'agent_id': agent_data['agent_id'],
                'true_belief': agent_data['initial_belief'],
                'fitted_belief': None,
                'loss': None,
                'x_mean_trajectory_cbsa': None,
                'x_trajectory': None,
                'true_belief_trajectory': true_belief_trajectory,  # Full true belief trajectory
                'theoretical_growth_rate': agent_data['theoretical_growth_rate'],
                'actual_growth_rate': agent_data['actual_growth_rate'],
                'final_resources': agent_data['final_resources'],
                'income_growth_rates': agent_data['income_growth_rates'],
                'p_t_series': dummy_data['p_t_series'],
                'status': 'failure',
                'error': str(e)
            }
            
            fitted_results.append(result)
    
    # Print summary
    successful_fits = [r for r in fitted_results if r['status'] == 'success']
    failed_fits = [r for r in fitted_results if r['status'] == 'failure']
    
    print(f"\nVI Fitting Summary:")
    print(f"  Successful fits: {len(successful_fits)}")
    print(f"  Failed fits: {len(failed_fits)}")
    print(f"  Success rate: {len(successful_fits)/len(fitted_results)*100:.1f}%")
    
    if successful_fits:
        losses = [r['loss'] for r in successful_fits if r['loss'] is not None]
        if losses:
            print(f"  Loss statistics:")
            print(f"    Mean: {np.mean(losses):.4f}")
            print(f"    Median: {np.median(losses):.4f}")
            print(f"    Std: {np.std(losses):.4f}")
            print(f"    Min: {np.min(losses):.4f}")
            print(f"    Max: {np.max(losses):.4f}")
    
    return fitted_results

def save_vi_results(fitted_results, output_path="vi_fitted_results.pkl"):
    """Save the VI fitting results to a pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(fitted_results, f)
    print(f"\nSaved VI results to {output_path}")
    
    # Also save a summary CSV
    summary_data = []
    for result in fitted_results:
        summary_data.append({
            'agent_id': result['agent_id'],
            'true_belief': result['true_belief'],
            'fitted_belief': result.get('fitted_belief', None),
            'loss': result.get('loss', None),
            'status': result['status'],
            'p_estimated': result.get('p_estimated', None),  # Include estimated p
            'l_t_series_length': len(result.get('l_t_series', [])) if result.get('l_t_series') else 0,  # Include l_t_series info
            'has_trajectory_data': result.get('true_belief_trajectory') is not None
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = output_path.replace('.pkl', '_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")


def main():
    """Main function to generate VI results for all agents."""
    
    # Load dummy data
    dummy_data_path = "dummy_data_kelly_betting_dynamic_x.pkl"
    if not os.path.exists(dummy_data_path):
        print(f"Error: Dummy data file not found at '{dummy_data_path}'")
        print("Please run generate_dummy_data.py first.")
        return
    
    print("Loading dummy data...")
    with open(dummy_data_path, 'rb') as f:
        dummy_data = pickle.load(f)
        
    vi_data = dummy_data['vi_data']
    true_p = dummy_data['parameters'].get('p', 0.75) # Get true p from dummy data

    print(f"Loaded {len(vi_data)} agents from dummy data.")
    print(f"True p = {true_p:.3f}")
    
    # Calculate frequentist p from the observed data
    print("\nCalculating frequentist p from observed data...")
    y_values_all_agents = [agent_data['income_growth_rates'] for agent_data in vi_data]
    p_estimated = calculate_frequentist_p(y_values_all_agents)
    
    print(f"Frequentist p estimate: {p_estimated:.3f}")
    print(f"True p: {true_p:.3f}")
    print(f"Absolute error: {abs(p_estimated - true_p):.3f}")
    
    # Calculate dynamic l_t_series using the new Bayesian approach
    print("\nCalculating dynamic l_t_series using Bayesian cross-sectional inference...")
    try:
        l_t_dict = estimate_l_time_series_bayesian(
            dummy_data, 
            delta=0.1,  # Sub-optimality offset (agents are 5% below optimal on average)
            n_samples=8000,  # VI samples per timestep 
            rolling_window=2  # Temporal smoothing window
        )
        
        if not l_t_dict:
            raise ValueError("Bayesian estimation returned empty results")
            
        print(f"\nBayesian l estimation complete!")
        print(f"Estimated l values for timesteps: {list(l_t_dict.keys())}")
        print(f"l range: {min(l_t_dict.values()):.2f} to {max(l_t_dict.values()):.2f}")
        
    except Exception as e:
        print(f"Warning: Bayesian l estimation failed ({str(e)})")
        print("Falling back to heuristic method...")
        l_t_dict = calculate_l_t_series_from_dummy_data(dummy_data)
        
        if not l_t_dict:
            print("Warning: Could not calculate l_t_series, using default l=2.0")
            l_t_dict = {i: 2.0 for i in range(11)}  # Default for 11 timesteps
        
        print(f"Calculated l values for timesteps: {list(l_t_dict.keys())}")
        print(f"l range: {min(l_t_dict.values()):.2f} to {max(l_t_dict.values()):.2f}")
    
    # Prepare tasks for joblib using estimated p and dynamic l
    tasks = [delayed(process_agent)(agent_data, p_estimated, l_t_dict, dummy_data) for agent_data in vi_data[:5]]

    # Run in parallel
    fitted_results = Parallel(n_jobs=-1)(tqdm(tasks, desc="Fitting VI model for agents"))

    # Save results
    save_vi_results(fitted_results)
    
    # Print summary
    print("\nVI results generation complete!")
    print(f"Used estimated p = {p_estimated:.3f} instead of true p = {true_p:.3f}")
    print(f"Used Bayesian l_t_series estimation with {len(l_t_dict)} timesteps")
    print("l estimation used cross-sectional VI + temporal smoothing ('Best of Both Worlds')")
    print("You can now run plot_validation_results.py to generate plots.")

if __name__ == "__main__":
    main() 