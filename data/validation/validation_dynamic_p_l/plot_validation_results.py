import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_loss_distribution(vi_results, save_path="validation_plots"):
    """
    Plot the distribution of VI loss values across all agents.
    
    Args:
        vi_results: List of VI fitting results
        save_path: Directory to save plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extract loss values
    losses = []
    for result in vi_results:
        if 'loss' in result and result['loss'] is not None:
            losses.append(result['loss'])
    
    if not losses:
        print("No loss values found in results")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Histogram of losses
    plt.subplot(1, 2, 1)
    plt.hist(losses, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('VI Loss')
    plt.ylabel('Frequency')
    plt.title('Distribution of VI Loss Values')
    plt.grid(True, alpha=0.3)
    
    # Box plot of losses
    plt.subplot(1, 2, 2)
    plt.boxplot(losses)
    plt.ylabel('VI Loss')
    plt.title('Box Plot of VI Loss Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'vi_loss_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss distribution plot saved to {save_path}/vi_loss_distribution.png")
    
    # Print summary statistics
    print(f"\nLoss Statistics:")
    print(f"  Mean: {np.mean(losses):.4f}")
    print(f"  Median: {np.median(losses):.4f}")
    print(f"  Std: {np.std(losses):.4f}")
    print(f"  Min: {np.min(losses):.4f}")
    print(f"  Max: {np.max(losses):.4f}")

def plot_parameter_comparison(vi_results, dummy_data, save_path="validation_plots"):
    """
    Plot fitted parameters vs true parameters.
    
    Args:
        vi_results: List of VI fitting results
        dummy_data: Original dummy data with true parameters
        save_path: Directory to save plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extract true and fitted parameters
    true_beliefs = []
    fitted_beliefs = []
    losses = []
    
    for result in vi_results:
        if 'loss' in result and result['loss'] is not None:
            # Find corresponding agent in dummy data
            agent_id = result['agent_id']
            for agent_data in dummy_data['vi_data']:
                if agent_data['agent_id'] == agent_id:
                    true_beliefs.append(agent_data['initial_belief'])
                    # Use final fitted belief
                    if 'x_mean_trajectory_cbsa' in result and result['x_mean_trajectory_cbsa'] is not None:
                        fitted_beliefs.append(result['x_mean_trajectory_cbsa'][-1])
                    else:
                        fitted_beliefs.append(np.nan)
                    losses.append(result['loss'])
                    break
    
    # Remove any NaN values
    valid_indices = ~np.isnan(fitted_beliefs)
    true_beliefs = np.array(true_beliefs)[valid_indices]
    fitted_beliefs = np.array(fitted_beliefs)[valid_indices]
    losses = np.array(losses)[valid_indices]
    
    if len(true_beliefs) == 0:
        print("No valid parameter comparisons found")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: True vs Fitted beliefs
    axes[0, 0].scatter(true_beliefs, fitted_beliefs, alpha=0.7, s=50)
    
    # Add perfect prediction line
    min_val = min(np.min(true_beliefs), np.min(fitted_beliefs))
    max_val = max(np.max(true_beliefs), np.max(fitted_beliefs))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    axes[0, 0].set_xlabel('True Belief (x)')
    axes[0, 0].set_ylabel('Fitted Belief (x)')
    axes[0, 0].set_title('True vs Fitted Agent Beliefs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    errors = fitted_beliefs - true_beliefs
    axes[0, 1].hist(errors, bins=15, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8, label='No Error')
    axes[0, 1].set_xlabel('Error (Fitted - True)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Parameter Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error vs True belief
    axes[1, 0].scatter(true_beliefs, errors, alpha=0.7, s=50)
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.8, label='No Error')
    axes[1, 0].set_xlabel('True Belief (x)')
    axes[1, 0].set_ylabel('Error (Fitted - True)')
    axes[1, 0].set_title('Parameter Error vs True Belief')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error vs Loss
    axes[1, 1].scatter(losses, np.abs(errors), alpha=0.7, s=50)
    axes[1, 1].set_xlabel('VI Loss')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Parameter Error vs VI Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'parameter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nParameter Comparison Statistics:")
    print(f"  Mean Absolute Error: {np.mean(np.abs(errors)):.4f}")
    print(f"  Root Mean Square Error: {np.sqrt(np.mean(errors**2)):.4f}")
    print(f"  Correlation (True vs Fitted): {np.corrcoef(true_beliefs, fitted_beliefs)[0,1]:.4f}")
    print(f"  Correlation (Error vs Loss): {np.corrcoef(losses, np.abs(errors))[0,1]:.4f}")

def plot_belief_evolution(dummy_data, save_path="validation_plots"):
    """Plot the evolution of agent beliefs over time."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get belief trajectories from new data structure
    if 'agent_trajectories' in dummy_data and 'beliefs' in dummy_data['agent_trajectories']:
        belief_trajectories = dummy_data['agent_trajectories']['beliefs']
    else:
        print("No belief trajectories found in dummy_data")
        return
    
    n_agents, n_timesteps_plus_1 = belief_trajectories.shape
    timesteps = np.arange(n_timesteps_plus_1)
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Individual agent belief trajectories
    timesteps = np.arange(n_timesteps_plus_1)
    for i in range(min(20, n_agents)):  # Plot first 20 agents
        axes[0, 0].plot(timesteps, belief_trajectories[i, :], alpha=0.6, linewidth=1)
    
    # Add mean trajectory
    mean_beliefs = np.mean(belief_trajectories, axis=0)
    axes[0, 0].plot(timesteps, mean_beliefs, 'r-', linewidth=3, label='Population Mean')
    
    # Add true p value
    p_mean = np.mean(dummy_data['p_t_series'])  # Use mean of dynamic p values
    axes[0, 0].axhline(y=p_mean, color='g', linestyle='--', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
    
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Belief (x)')
    axes[0, 0].set_title('Agent Belief Evolution Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Belief distribution at different timesteps
    timestep_indices = [0, n_timesteps_plus_1//4, n_timesteps_plus_1//2, 3*n_timesteps_plus_1//4, n_timesteps_plus_1-1]
    timestep_labels = ['Initial', '25%', '50%', '75%', 'Final']
    
    for i, (idx, label) in enumerate(zip(timestep_indices, timestep_labels)):
        beliefs_at_timestep = belief_trajectories[:, idx]
        axes[0, 1].hist(beliefs_at_timestep, bins=15, alpha=0.6, label=label, density=True)
    
    axes[0, 1].axvline(x=p_mean, color='g', linestyle='--', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
    axes[0, 1].set_xlabel('Belief (x)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Belief Distribution at Different Timesteps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Convergence to true p
    convergence = np.abs(belief_trajectories - p_mean)
    mean_convergence = np.mean(convergence, axis=0)
    std_convergence = np.std(convergence, axis=0)
    
    axes[1, 0].plot(timesteps, mean_convergence, 'b-', linewidth=2, label='Mean Error')
    axes[1, 0].fill_between(timesteps, 
                            mean_convergence - std_convergence, 
                            mean_convergence + std_convergence, 
                            alpha=0.3, label='±1 Std Dev')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('|Belief - Mean p|')
    axes[1, 0].set_title('Convergence to Mean Probability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Final belief vs initial belief
    initial_beliefs = belief_trajectories[:, 0]
    final_beliefs = belief_trajectories[:, -1]
    
    axes[1, 1].scatter(initial_beliefs, final_beliefs, alpha=0.7, s=50)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='No Change')
    axes[1, 1].axhline(y=p_mean, color='g', linestyle='--', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
    axes[1, 1].set_xlabel('Initial Belief (x₀)')
    axes[1, 1].set_ylabel('Final Belief (x_T)')
    axes[1, 1].set_title('Initial vs Final Beliefs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'belief_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Belief evolution plot saved to {save_path}/belief_evolution.png")

def plot_resource_trajectories(dummy_data, save_path="validation_plots"):
    """Plot the resource trajectories of agents over time."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get resource trajectories from new data structure
    if 'agent_trajectories' in dummy_data and 'resources' in dummy_data['agent_trajectories']:
        resource_trajectories = dummy_data['agent_trajectories']['resources']
    else:
        print("No resource trajectories found in dummy_data")
        return
    
    n_agents, n_timesteps_plus_1 = resource_trajectories.shape
    timesteps = np.arange(n_timesteps_plus_1)
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Individual resource trajectories (log scale)
    for i in range(min(20, n_agents)):  # Plot first 20 agents
        axes[0, 0].plot(timesteps, resource_trajectories[i, :], alpha=0.6, linewidth=1)
    
    # Add mean trajectory
    mean_resources = np.mean(resource_trajectories, axis=0)
    axes[0, 0].plot(timesteps, mean_resources, 'r-', linewidth=3, label='Population Mean')
    
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Resources ($)')
    axes[0, 0].set_title('Resource Trajectories (Log Scale)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Growth rate distribution
    growth_rates = []
    for i in range(n_agents):
        if resource_trajectories[i, -1] > 0:
            growth_rate = np.log(resource_trajectories[i, -1] / resource_trajectories[i, 0]) / (n_timesteps_plus_1 - 1)
            growth_rates.append(growth_rate)
    
    axes[0, 1].hist(growth_rates, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Growth Rate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Growth Rates')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. p_t series
    p_t_series = dummy_data['p_t_series']
    timesteps_p = np.arange(len(p_t_series))
    
    axes[1, 0].plot(timesteps_p, p_t_series, 'b-', linewidth=2)
    p_mean = np.mean(p_t_series)  # Use mean of dynamic p values
    axes[1, 0].axhline(y=p_mean, color='g', linestyle='--', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Fraction of Agents with Gains')
    axes[1, 0].set_title('p_t Series: Dynamic Environment Probability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Final resources vs initial beliefs
    initial_beliefs = dummy_data['agent_trajectories']['beliefs'][:, 0]
    final_resources = resource_trajectories[:, -1]
    
    axes[1, 1].scatter(initial_beliefs, final_resources, alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Initial Belief (x₀)')
    axes[1, 1].set_ylabel('Final Resources ($)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Final Resources vs Initial Beliefs')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'resource_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Resource trajectories plot saved to {save_path}/resource_trajectories.png")

def plot_time_series_parameter_comparison(vi_results, dummy_data, save_path="validation_plots"):
    """
    Plot time series parameter fits vs actual test dummy parameters.
    
    Args:
        vi_results: List of VI fitting results
        dummy_data: Original dummy data with true parameters
        save_path: Directory to save plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extract time series data for successful fits
    successful_results = [r for r in vi_results if r['status'] == 'success' and 'x_mean_trajectory_cbsa' in r]
    
    if not successful_results:
        print("No successful VI results with time series data found")
        return
    
    print(f"Creating time series parameter comparison plots for {len(successful_results)} agents...")
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Time series comparison for sample agents
    n_sample_agents = min(5, len(successful_results))
    timesteps = np.arange(len(successful_results[0]['x_mean_trajectory_cbsa']))
    
    for i in range(n_sample_agents):
        result = successful_results[i]
        agent_id = result['agent_id']
        
        # Get true belief trajectory from dummy data
        true_belief_trajectory = None
        for agent_data in dummy_data['vi_data']:
            if agent_data['agent_id'] == agent_id:
                # Extract belief trajectory for this agent
                belief_trajectories = dummy_data['agent_trajectories']['beliefs']
                agent_idx = None
                for j, data in enumerate(dummy_data['vi_data']):
                    if data['agent_id'] == agent_id:
                        agent_idx = j
                        break
                
                if agent_idx is not None:
                    true_belief_trajectory = belief_trajectories[agent_idx, :]
                break
        
        if true_belief_trajectory is not None:
            # Ensure both trajectories have the same length
            fitted_trajectory = result['x_mean_trajectory_cbsa']
            min_len = min(len(true_belief_trajectory), len(fitted_trajectory))
            timesteps_aligned = np.arange(min_len)
            
            # Plot true vs fitted trajectories
            axes[0, 0].plot(timesteps_aligned, true_belief_trajectory[:min_len], 
                           alpha=0.7, linewidth=2, 
                           label=f'Agent {agent_id}: True x₀={result["true_belief"]:.3f}')
            axes[0, 0].plot(timesteps_aligned, fitted_trajectory[:min_len], 
                           alpha=0.7, linewidth=2, linestyle='--',
                           label=f'Agent {agent_id}: Fitted')
    
    # Add true p value
    p_mean = np.mean(dummy_data['p_t_series'])  # Use mean of dynamic p values
    axes[0, 0].axhline(y=p_mean, color='g', linestyle=':', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
    
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Belief (x)')
    axes[0, 0].set_title('Time Series: True vs Fitted Beliefs (Sample Agents)')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Initial vs Final parameter comparison
    initial_true = []
    final_true = []
    initial_fitted = []
    final_fitted = []
    
    for result in successful_results:
        agent_id = result['agent_id']
        
        # Find corresponding agent in dummy data
        for agent_data in dummy_data['vi_data']:
            if agent_data['agent_id'] == agent_id:
                belief_trajectories = dummy_data['agent_trajectories']['beliefs']
                agent_idx = None
                for j, data in enumerate(dummy_data['vi_data']):
                    if data['agent_id'] == agent_id:
                        agent_idx = j
                        break
                
                if agent_idx is not None:
                    true_trajectory = belief_trajectories[agent_idx, :]
                    initial_true.append(true_trajectory[0])
                    final_true.append(true_trajectory[-1])
                    initial_fitted.append(result['x_mean_trajectory_cbsa'][0])
                    final_fitted.append(result['x_mean_trajectory_cbsa'][-1])
                break
    
    # Initial beliefs comparison
    axes[0, 1].scatter(initial_true, initial_fitted, alpha=0.7, s=50)
    min_val = min(min(initial_true), min(initial_fitted))
    max_val = max(max(initial_true), max(initial_fitted))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Match')
    axes[0, 1].set_xlabel('True Initial Belief (x₀)')
    axes[0, 1].set_ylabel('Fitted Initial Belief (x₀)')
    axes[0, 1].set_title('Initial Beliefs: True vs Fitted')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final beliefs comparison
    axes[0, 2].scatter(final_true, final_fitted, alpha=0.7, s=50)
    min_val = min(min(final_true), min(final_fitted))
    max_val = max(max(final_true), max(final_fitted))
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Match')
    axes[0, 2].axhline(y=p_mean, color='g', linestyle=':', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
    axes[0, 2].axvline(x=p_mean, color='g', linestyle=':', alpha=0.8)
    axes[0, 2].set_xlabel('True Final Belief (x_T)')
    axes[0, 2].set_ylabel('Fitted Final Belief (x_T)')
    axes[0, 2].set_title('Final Beliefs: True vs Fitted')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 3. Parameter evolution comparison
    # Calculate mean trajectories across all agents
    n_timesteps = len(successful_results[0]['x_mean_trajectory_cbsa'])
    mean_true_trajectory = np.zeros(n_timesteps)
    mean_fitted_trajectory = np.zeros(n_timesteps)
    std_true_trajectory = np.zeros(n_timesteps)
    std_fitted_trajectory = np.zeros(n_timesteps)
    
    # Collect all trajectories
    all_true_trajectories = []
    all_fitted_trajectories = []
    
    for result in successful_results:
        agent_id = result['agent_id']
        
        for agent_data in dummy_data['vi_data']:
            if agent_data['agent_id'] == agent_id:
                belief_trajectories = dummy_data['agent_trajectories']['beliefs']
                agent_idx = None
                for j, data in enumerate(dummy_data['vi_data']):
                    if data['agent_id'] == agent_id:
                        agent_idx = j
                        break
                
                if agent_idx is not None:
                    true_trajectory = belief_trajectories[agent_idx, :]
                    fitted_trajectory = result['x_mean_trajectory_cbsa']
                    
                    # Ensure same length
                    min_len = min(len(true_trajectory), len(fitted_trajectory))
                    all_true_trajectories.append(true_trajectory[:min_len])
                    all_fitted_trajectories.append(fitted_trajectory[:min_len])
                break
    
    # Calculate statistics
    if all_true_trajectories:
        all_true_trajectories = np.array(all_true_trajectories)
        all_fitted_trajectories = np.array(all_fitted_trajectories)
        
        min_len = min(all_true_trajectories.shape[1], all_fitted_trajectories.shape[1])
        timesteps_short = np.arange(min_len)
        
        mean_true_trajectory = np.mean(all_true_trajectories[:, :min_len], axis=0)
        mean_fitted_trajectory = np.mean(all_fitted_trajectories[:, :min_len], axis=0)
        std_true_trajectory = np.std(all_true_trajectories[:, :min_len], axis=0)
        std_fitted_trajectory = np.std(all_fitted_trajectories[:, :min_len], axis=0)
        
        # Plot mean trajectories with error bands
        axes[1, 0].plot(timesteps_short, mean_true_trajectory, 'b-', linewidth=2, label='True (Mean)')
        axes[1, 0].fill_between(timesteps_short, 
                                mean_true_trajectory - std_true_trajectory,
                                mean_true_trajectory + std_true_trajectory, 
                                alpha=0.3, color='blue')
        
        axes[1, 0].plot(timesteps_short, mean_fitted_trajectory, 'r-', linewidth=2, label='Fitted (Mean)')
        axes[1, 0].fill_between(timesteps_short, 
                                mean_fitted_trajectory - std_fitted_trajectory,
                                mean_fitted_trajectory + std_fitted_trajectory, 
                                alpha=0.3, color='red')
        
        # Add true p value
        axes[1, 0].axhline(y=p_mean, color='g', linestyle=':', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
        
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Belief (x)')
        axes[1, 0].set_title('Population Mean: True vs Fitted Beliefs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Parameter recovery over time
    # Calculate correlation between true and fitted at each timestep
    if all_true_trajectories is not None and len(all_true_trajectories) > 1:
        correlations = []
        for t in range(min_len):
            true_vals = all_true_trajectories[:, t]
            fitted_vals = all_fitted_trajectories[:, t]
            if len(true_vals) > 1 and len(fitted_vals) > 1:
                corr = np.corrcoef(true_vals, fitted_vals)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
        
        axes[1, 1].plot(timesteps_short, correlations, 'b-', linewidth=2, marker='o')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Correlation (True vs Fitted)')
        axes[1, 1].set_title('Parameter Recovery Quality Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-1, 1)
    
    # 5. Error evolution over time
    if all_true_trajectories is not None:
        mean_errors = np.mean(np.abs(all_fitted_trajectories[:, :min_len] - all_true_trajectories[:, :min_len]), axis=0)
        std_errors = np.std(np.abs(all_fitted_trajectories[:, :min_len] - all_true_trajectories[:, :min_len]), axis=0)
        
        axes[1, 2].plot(timesteps_short, mean_errors, 'b-', linewidth=2, label='Mean Error')
        axes[1, 2].fill_between(timesteps_short, 
                                mean_errors - std_errors,
                                mean_errors + std_errors, 
                                alpha=0.3, color='blue', label='±1 Std Dev')
        axes[1, 2].set_xlabel('Timestep')
        axes[1, 2].set_ylabel('|Fitted - True|')
        axes[1, 2].set_title('Parameter Error Evolution Over Time')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'time_series_parameter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"Time series parameter comparison plot saved to {save_path}/time_series_parameter_comparison.png")
    
    if all_true_trajectories is not None:
        # Calculate overall statistics
        overall_correlation = np.corrcoef(all_true_trajectories.flatten(), all_fitted_trajectories.flatten())[0, 1]
        overall_mae = np.mean(np.abs(all_fitted_trajectories - all_true_trajectories))
        overall_rmse = np.sqrt(np.mean((all_fitted_trajectories - all_true_trajectories)**2))
        
        print(f"\nTime Series Parameter Recovery Statistics:")
        print(f"  Overall Correlation: {overall_correlation:.4f}")
        print(f"  Overall Mean Absolute Error: {overall_mae:.4f}")
        print(f"  Overall Root Mean Square Error: {overall_rmse:.4f}")
        
        if len(correlations) > 0:
            print(f"  Initial Timestep Correlation: {correlations[0]:.4f}")
            print(f"  Final Timestep Correlation: {correlations[-1]:.4f}")
            print(f"  Correlation Improvement: {correlations[-1] - correlations[0]:.4f}")

def plot_individual_agent_trajectories(vi_results, dummy_data, save_path="validation_plots"):
    """
    Plot overlaid true vs fitted x trajectories for individual agents.
    
    Args:
        vi_results: List of VI fitting results
        dummy_data: Original dummy data with true parameters
        save_path: Directory to save plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Extract successful results with trajectory data
    successful_results = [r for r in vi_results if r['status'] == 'success' and 'x_mean_trajectory_cbsa' in r]
    
    if not successful_results:
        print("No successful VI results with trajectory data found")
        return
    
    print(f"Creating individual agent trajectory plots for {len(successful_results)} agents...")
    
    # Create subplots for different groups of agents
    n_agents = len(successful_results)
    n_cols = 3
    n_rows = (n_agents + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get true p value for reference
    p_mean = np.mean(dummy_data['p_t_series'])  # Use mean of dynamic p values
    
    for i, result in enumerate(successful_results):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        agent_id = result['agent_id']
        
        # Get true belief trajectory from dummy data
        true_belief_trajectory = None
        for agent_data in dummy_data['vi_data']:
            if agent_data['agent_id'] == agent_id:
                belief_trajectories = dummy_data['agent_trajectories']['beliefs']
                agent_idx = None
                for j, data in enumerate(dummy_data['vi_data']):
                    if data['agent_id'] == agent_id:
                        agent_idx = j
                        break
                
                if agent_idx is not None:
                    true_belief_trajectory = belief_trajectories[agent_idx, :]
                break
        
        if true_belief_trajectory is not None:
            fitted_trajectory = result['x_mean_trajectory_cbsa']
            
            # Ensure same length
            min_len = min(len(true_belief_trajectory), len(fitted_trajectory))
            timesteps = np.arange(min_len)
            
            # Plot true trajectory
            ax.plot(timesteps, true_belief_trajectory[:min_len], 'b-', linewidth=2, 
                   label=f'True (x₀={true_belief_trajectory[0]:.3f})')
            
            # Plot fitted trajectory
            ax.plot(timesteps, fitted_trajectory[:min_len], 'r--', linewidth=2, 
                   label=f'Fitted (x₀={fitted_trajectory[0]:.3f})')
            
            # Add true p reference line
            ax.axhline(y=p_mean, color='g', linestyle=':', alpha=0.8, label=f'Mean p = {p_mean:.3f}')
            
            # Add perfect prediction line (diagonal)
            ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Perfect Match')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Belief (x)')
            ax.set_title(f'Agent {agent_id}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Calculate correlation for this agent
            true_vals = true_belief_trajectory[:min_len]
            fitted_vals = fitted_trajectory[:min_len]
            if len(true_vals) > 1 and len(fitted_vals) > 1:
                corr = np.corrcoef(true_vals, fitted_vals)[0, 1]
                if not np.isnan(corr):
                    ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top', fontsize=8)
    
    # Hide empty subplots
    for i in range(n_agents, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'individual_agent_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual agent trajectory plots saved to {save_path}/individual_agent_trajectories.png")
    
    # Print summary statistics
    correlations = []
    for result in successful_results:
        agent_id = result['agent_id']
        for agent_data in dummy_data['vi_data']:
            if agent_data['agent_id'] == agent_id:
                belief_trajectories = dummy_data['agent_trajectories']['beliefs']
                agent_idx = None
                for j, data in enumerate(dummy_data['vi_data']):
                    if data['agent_id'] == agent_id:
                        agent_idx = j
                        break
                
                if agent_idx is not None:
                    true_trajectory = belief_trajectories[agent_idx, :]
                    fitted_trajectory = result['x_mean_trajectory_cbsa']
                    
                    min_len = min(len(true_trajectory), len(fitted_trajectory))
                    true_vals = true_trajectory[:min_len]
                    fitted_vals = fitted_trajectory[:min_len]
                    
                    if len(true_vals) > 1 and len(fitted_vals) > 1:
                        corr = np.corrcoef(true_vals, fitted_vals)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                break
    
    if correlations:
        print(f"\nIndividual Agent Trajectory Statistics:")
        print(f"  Mean Correlation: {np.mean(correlations):.4f}")
        print(f"  Median Correlation: {np.median(correlations):.4f}")
        print(f"  Std Correlation: {np.std(correlations):.4f}")
        print(f"  Min Correlation: {np.min(correlations):.4f}")
        print(f"  Max Correlation: {np.max(correlations):.4f}")

def plot_p_estimation_accuracy(vi_results, dummy_data, save_path="validation_plots"):
    """
    Plot the accuracy of the frequentist p estimation.
    
    Args:
        vi_results: List of VI fitting results
        dummy_data: Original dummy data with true parameters
        save_path: Directory to save plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get true p from dummy data
    true_p = dummy_data['parameters'].get('p', 0.75)
    
    # Extract estimated p values
    p_estimated_values = []
    for result in vi_results:
        if 'p_estimated' in result and result['p_estimated'] is not None:
            p_estimated_values.append(result['p_estimated'])
    
    if not p_estimated_values:
        print("No estimated p values found in results")
        return
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Histogram of estimated p values
    axes[0].hist(p_estimated_values, bins=20, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=true_p, color='red', linestyle='--', linewidth=2, label=f'True p = {true_p:.3f}')
    axes[0].axvline(x=np.mean(p_estimated_values), color='green', linestyle='-', linewidth=2, 
                    label=f'Mean estimated p = {np.mean(p_estimated_values):.3f}')
    axes[0].set_xlabel('Estimated p')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Frequentist p Estimates')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Box plot comparing true vs estimated
    axes[1].boxplot([p_estimated_values], labels=['Estimated p'])
    axes[1].axhline(y=true_p, color='red', linestyle='--', linewidth=2, label=f'True p = {true_p:.3f}')
    axes[1].set_ylabel('p value')
    axes[1].set_title('Frequentist p Estimation vs True p')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'p_estimation_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\np Estimation Accuracy Statistics:")
    print(f"  True p: {true_p:.3f}")
    print(f"  Mean estimated p: {np.mean(p_estimated_values):.3f}")
    print(f"  Median estimated p: {np.median(p_estimated_values):.3f}")
    print(f"  Std estimated p: {np.std(p_estimated_values):.3f}")
    print(f"  Absolute error: {abs(np.mean(p_estimated_values) - true_p):.3f}")
    print(f"  Relative error: {abs(np.mean(p_estimated_values) - true_p) / true_p * 100:.1f}%")
    
    print(f"p estimation accuracy plot saved to {save_path}/p_estimation_accuracy.png")

def plot_l_estimation_accuracy(vi_results, dummy_data, save_path="validation_plots"):
    """
    Plot the accuracy of the l estimation over time.
    
    Args:
        vi_results: List of VI fitting results
        dummy_data: Original dummy data with true parameters
        save_path: Directory to save plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get true l_t_series from dummy data
    true_l_t_series = dummy_data['l_t_series']
    
    # Extract estimated l values from VI results
    # Each agent should have an l_t_series in their results
    estimated_l_by_timestep = {}
    
    for result in vi_results:
        if 'l_t_series' in result and result['l_t_series'] is not None:
            l_series = result['l_t_series']
            for timestep, l_val in enumerate(l_series):
                if timestep not in estimated_l_by_timestep:
                    estimated_l_by_timestep[timestep] = []
                estimated_l_by_timestep[timestep].append(l_val)
    
    if not estimated_l_by_timestep:
        print("No estimated l values found in results")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. True vs Estimated l over time
    timesteps = np.arange(len(true_l_t_series))
    
    # Plot true l values
    axes[0, 0].plot(timesteps, true_l_t_series, 'b-', linewidth=3, marker='o', 
                    markersize=8, label='True l', alpha=0.8)
    
    # Plot mean estimated l values with error bars
    estimated_l_means = []
    estimated_l_stds = []
    for timestep in timesteps:
        if timestep in estimated_l_by_timestep:
            l_vals = estimated_l_by_timestep[timestep]
            estimated_l_means.append(np.mean(l_vals))
            estimated_l_stds.append(np.std(l_vals))
        else:
            estimated_l_means.append(np.nan)
            estimated_l_stds.append(np.nan)
    
    axes[0, 0].errorbar(timesteps, estimated_l_means, yerr=estimated_l_stds, 
                         fmt='r--', linewidth=2, marker='s', markersize=6, 
                         label='Estimated l (Mean ± Std)', alpha=0.8, capsize=5)
    
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Number of Outcomes (l)')
    axes[0, 0].set_title('l Estimation Accuracy Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yticks([2, 3])
    
    # 2. Scatter plot: True vs Estimated l
    true_l_flat = []
    estimated_l_flat = []
    
    for timestep in timesteps:
        if timestep in estimated_l_by_timestep:
            true_l = true_l_t_series[timestep]
            for est_l in estimated_l_by_timestep[timestep]:
                true_l_flat.append(true_l)
                estimated_l_flat.append(est_l)
    
    if true_l_flat:
        axes[0, 1].scatter(true_l_flat, estimated_l_flat, alpha=0.7, s=50)
        
        # Add perfect prediction line
        min_l = min(min(true_l_flat), min(estimated_l_flat))
        max_l = max(max(true_l_flat), max(estimated_l_flat))
        axes[0, 1].plot([min_l, max_l], [min_l, max_l], 'r--', alpha=0.8, label='Perfect Match')
        
        # Add discrete l lines
        axes[0, 1].axhline(y=2, color='g', linestyle=':', alpha=0.5, label='l=2')
        axes[0, 1].axhline(y=3, color='g', linestyle=':', alpha=0.5, label='l=3')
        axes[0, 1].axvline(x=2, color='g', linestyle=':', alpha=0.5)
        axes[0, 1].axvline(x=3, color='g', linestyle=':', alpha=0.5)
        
        axes[0, 1].set_xlabel('True l')
        axes[0, 1].set_ylabel('Estimated l')
        axes[0, 1].set_title('l Estimation: True vs Estimated')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(1.5, 3.5)
        axes[0, 1].set_xticks([2, 3])
        axes[0, 1].set_yticks([2, 3])
    
    # 3. Estimation error over time
    errors = []
    for timestep in timesteps:
        if timestep in estimated_l_by_timestep:
            true_l = true_l_t_series[timestep]
            est_l_vals = estimated_l_by_timestep[timestep]
            timestep_errors = [abs(est_l - true_l) for est_l in est_l_vals]
            errors.append(np.mean(timestep_errors))
        else:
            errors.append(np.nan)
    
    axes[1, 0].plot(timesteps, errors, 'b-', linewidth=2, marker='o', markersize=6)
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Mean Absolute Error')
    axes[1, 0].set_title('l Estimation Error Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, max(errors) * 1.1 if errors else 1)
    
    # 4. Distribution of estimated l values by true l
    l2_estimates = []
    l3_estimates = []
    
    for timestep in timesteps:
        if timestep in estimated_l_by_timestep:
            true_l = true_l_t_series[timestep]
            est_l_vals = estimated_l_by_timestep[timestep]
            
            if true_l == 2:
                l2_estimates.extend(est_l_vals)
            elif true_l == 3:
                l3_estimates.extend(est_l_vals)
    
    if l2_estimates or l3_estimates:
        axes[1, 1].hist([l2_estimates, l3_estimates], bins=15, alpha=0.7, 
                        label=['True l=2', 'True l=3'], edgecolor='black')
        axes[1, 1].axvline(x=2, color='g', linestyle='--', alpha=0.8, label='l=2')
        axes[1, 1].axvline(x=3, color='g', linestyle='--', alpha=0.8, label='l=3')
        axes[1, 1].set_xlabel('Estimated l')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Estimated l by True l')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'l_estimation_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nl Estimation Accuracy Statistics:")
    print(f"  True l values: {true_l_t_series}")
    
    if true_l_flat and estimated_l_flat:
        # Calculate overall statistics
        overall_mae = np.mean([abs(t - e) for t, e in zip(true_l_flat, estimated_l_flat)])
        overall_rmse = np.sqrt(np.mean([(t - e)**2 for t, e in zip(true_l_flat, estimated_l_flat)]))
        
        # Calculate accuracy by true l value
        l2_accuracy = np.mean([abs(2 - e) for e in l2_estimates]) if l2_estimates else np.nan
        l3_accuracy = np.mean([abs(3 - e) for e in l3_estimates]) if l3_estimates else np.nan
        
        print(f"  Overall Mean Absolute Error: {overall_mae:.4f}")
        print(f"  Overall Root Mean Square Error: {overall_rmse:.4f}")
        print(f"  l=2 Estimation Error: {l2_accuracy:.4f}")
        print(f"  l=3 Estimation Error: {l3_accuracy:.4f}")
        
        # Count correct classifications
        l2_correct = sum(1 for e in l2_estimates if abs(e - 2) < 0.5)
        l3_correct = sum(1 for e in l3_estimates if abs(e - 3) < 0.5)
        total_l2 = len(l2_estimates)
        total_l3 = len(l3_estimates)
        
        if total_l2 > 0:
            print(f"  l=2 Classification Accuracy: {l2_correct}/{total_l2} = {l2_correct/total_l2*100:.1f}%")
        if total_l3 > 0:
            print(f"  l=3 Classification Accuracy: {l3_correct}/{total_l3} = {l3_correct/total_l3*100:.1f}%")
    
    print(f"l estimation accuracy plot saved to {save_path}/l_estimation_accuracy.png")

def main():
    """Main function to generate all validation plots."""
    
    # Load dummy data
    dummy_data_path = "dummy_data_kelly_betting_dynamic_x.pkl"
    if not os.path.exists(dummy_data_path):
        print(f"Error: Dummy data file not found at '{dummy_data_path}'")
        print("Please run generate_dummy_data.py first.")
        return
    
    print("Loading dummy data...")
    with open(dummy_data_path, 'rb') as f:
        dummy_data = pickle.load(f)
    
    # Load VI results if available
    vi_results_path = "vi_fitted_results.pkl"
    vi_results = None
    if os.path.exists(vi_results_path):
        print("Loading VI results...")
        with open(vi_results_path, 'rb') as f:
            vi_results = pickle.load(f)
    else:
        print("No VI results found. Run generate_vi_results.py first to generate VI results.")
    
    # Create plots directory
    save_path = "validation_plots"
    
    # Generate all plots
    print(f"\nGenerating validation plots in '{save_path}' directory...")
    
    # Plot resource trajectories and belief evolution
    plot_resource_trajectories(dummy_data, save_path)
    plot_belief_evolution(dummy_data, save_path)
    
    # Plot VI results if available
    if vi_results is not None:
        plot_loss_distribution(vi_results, save_path)
        plot_parameter_comparison(vi_results, dummy_data, save_path)
        plot_time_series_parameter_comparison(vi_results, dummy_data, save_path)
        plot_individual_agent_trajectories(vi_results, dummy_data, save_path)
        plot_p_estimation_accuracy(vi_results, dummy_data, save_path)
        plot_l_estimation_accuracy(vi_results, dummy_data, save_path)  # New function call
    
    print(f"\nAll validation plots generated in '{save_path}' directory!")

if __name__ == "__main__":
    main()
