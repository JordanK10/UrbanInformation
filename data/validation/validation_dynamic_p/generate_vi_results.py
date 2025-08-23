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
from ssm_model import fit_ssm_random_walk

# Suppress pymc warnings
import logging
logging.getLogger("pymc").setLevel(logging.ERROR)

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
    This includes smoothing and applying the user's theoretical prior.
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

    # Apply smoothing
    raw_series = pd.Series(raw_x_trajectory)
    smoothed_x = raw_series.rolling(window=3, min_periods=1).mean().values

    # Apply the user's theoretical prior on the initial state
    # Now using estimated p instead of true p
    # if p_estimated is not None and len(raw_x_trajectory) > 0:
    #     if p_estimated < 0.5 and y_values[0] > 0 and y[0] > 0.5:
    #         smoothed_x = 1 - smoothed_x
            
    return smoothed_x

def process_agent(agent_data, p_estimated, l, dummy_data):
    """
    Processes a single agent's data to fit a VI model.
    
    Args:
        agent_data: Dictionary containing agent's data.
        p_estimated: The estimated p value from frequentist calculation.
        l: The parameter l for the agent.
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
    belief_trajectories = dummy_data['belief_trajectories']
    agent_idx = None
    for j, data in enumerate(dummy_data['vi_data']):
        if data['agent_id'] == agent_id:
            agent_idx = j
            break
    
    if agent_idx is not None:
        true_belief_trajectory = belief_trajectories[agent_idx, :]
    
    try:
        # --- New Two-Step Inference Process ---
        # 1. Generate the smart initial guess for the trajectory using estimated p
        l_t_series = np.full_like(y_values, l)
        init_x_traj = generate_initial_guess(y_values, p_estimated, l)

        # 2. Fit the robust VI model using the initial guess
        idata, loss = fit_ssm_random_walk(
            y_values=y_values,
            l_t_series=l_t_series,
            init_x_traj=init_x_traj,
            n_samples=20000  # Use a sufficient number of samples for good convergence
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
            
            # Fit VI model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
            idata, loss = fit_ssm_dynamic_p_model(
                y_values, p_t_series, 
                method=method, vi_steps=vi_steps,
                random_seed=seed
            )
            
            # Extract results - now x is directly available
            x_mean_trajectory = idata.posterior["x"].mean(axis=(0, 1)).to_numpy()
            
            # Get true belief trajectory from dummy data
            true_belief_trajectory = None
            belief_trajectories = dummy_data['belief_trajectories']
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
                'theoretical_growth_rate': agent_data['theoretical_growth_rate'],
                'actual_growth_rate': agent_data['actual_growth_rate'],
                'final_resources': agent_data['final_resources'],
                'income_growth_rates': y_values,
                'p_t_series': p_t_series,
                'status': 'success'
            }
            
            fitted_results.append(result)
            
        except Exception as e:
            print(f"    Agent {i+1} failed: {str(e)}")
            
            # Get true belief trajectory from dummy data for failed fits too
            true_belief_trajectory = None
            belief_trajectories = dummy_data['belief_trajectories']
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
            'has_trajectory_data': result.get('true_belief_trajectory') is not None
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = output_path.replace('.pkl', '_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")


def main():
    """Main function to generate VI results for all agents."""
    
    # Load dummy data
    dummy_data_path = "dummy_data_kelly_betting.pkl"
    if not os.path.exists(dummy_data_path):
        print(f"Error: Dummy data file not found at '{dummy_data_path}'")
        print("Please run generate_dummy_data.py first.")
        return
    
    print("Loading dummy data...")
    with open(dummy_data_path, 'rb') as f:
        dummy_data = pickle.load(f)
        
    vi_data = dummy_data['vi_data']
    true_l = dummy_data['parameters'].get('l', 2.0) # Get true l from dummy data
    true_p = dummy_data['parameters'].get('p', 0.75) # Get true p for comparison

    print(f"Loaded {len(vi_data)} agents from dummy data.")
    print(f"True l = {true_l:.2f}, True p = {true_p:.3f}")
    
    # Calculate frequentist p from the observed data
    print("\nCalculating frequentist p from observed data...")
    y_values_all_agents = [agent_data['income_growth_rates'] for agent_data in vi_data]
    p_estimated = calculate_frequentist_p(y_values_all_agents)
    
    print(f"Frequentist p estimate: {p_estimated:.3f}")
    print(f"True p: {true_p:.3f}")
    print(f"Absolute error: {abs(p_estimated - true_p):.3f}")
    
    # Prepare tasks for joblib using estimated p
    tasks = [delayed(process_agent)(agent_data, p_estimated, true_l, dummy_data) for agent_data in vi_data[:20]]

    # Run in parallel
    fitted_results = Parallel(n_jobs=-1)(tqdm(tasks, desc="Fitting VI model for agents"))

    # Save results
    save_vi_results(fitted_results)
    
    # Print summary
    print("\nVI results generation complete!")
    print(f"Used estimated p = {p_estimated:.3f} instead of true p = {true_p:.3f}")
    print("You can now run plot_validation_results.py to generate plots.")

if __name__ == "__main__":
    main() 