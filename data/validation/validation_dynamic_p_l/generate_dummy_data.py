import numpy as np
import pandas as pd
import pickle
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports if needed
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def update_agent_beliefs(agent_beliefs, p, l, k, timestep):
    """
    Update agent beliefs using the learning equation:
    x = (p*t/(l*k) + x_0)/(1 + t/(l*k))
    
    Args:
        agent_beliefs: Current agent beliefs array
        p: True environment probability
        l: Number of outcomes
        k: Learning rate parameter
        timestep: Current timestep (0-indexed, so add 1 for the equation)
    
    Returns:
        Updated agent beliefs array
    """
    t = timestep + 1  # Convert to 1-indexed for the equation
    denominator = 1 + t/(l*k)
    
    updated_beliefs = np.zeros_like(agent_beliefs)
    for i, x_0 in enumerate(agent_beliefs):
        # x_0 is the initial belief for this agent
        numerator = p * t / (l * k) + x_0 + np.random.normal(0, 0.01)
        updated_beliefs[i] = numerator / denominator
    
    return updated_beliefs

def generate_dummy_data(n_agents=1000, n_timesteps=11, 
                       initial_resources=50000, n_dice_rolls=3, seed=42,
                       dynamic_x=True):
    """
    Generate dummy data for calibrating the VI model based on Kelly betting theory.
    Now with dynamic p and continuous l that change at each timestep.
    
    NEW: Continuous reward system where:
    - Win (r < p): reward = l * x (agent gets rewarded proportionally to confidence)
    - Loss (r ≥ p): reward = (1-x) * l (agent gets penalized proportionally to confidence)
    
    Args:
        n_agents: Number of agents in the simulation
        n_timesteps: Number of time steps to simulate
        initial_resources: Initial resources for each agent
        n_dice_rolls: Number of dice rolls per timestep (for geometric mean)
        seed: Random seed for reproducibility
        dynamic_x: If True, agent beliefs evolve over time; if False, beliefs remain constant
    
    Returns:
        dict: Contains agent trajectories, environment outcomes, and parameters
    """
    np.random.seed(seed)
    
    print(f"Generating dummy data with parameters:")
    print(f"  n_agents = {n_agents}")
    print(f"  n_timesteps = {n_timesteps}")
    print(f"  initial_resources = {initial_resources:,}")
    print(f"  n_dice_rolls = {n_dice_rolls}")
    print(f"  Dynamic l: continuous uniform between 1.5 and 2.5")
    print(f"  Dynamic p: uniform 0.6-0.8 when l<2.0, uniform 0.4-0.5 when l≥2.0")
    print(f"  NEW: Continuous rewards: l*x for wins, (1-x)*l for losses")
    if dynamic_x:
        print(f"  ✓ Dynamic x: agent beliefs evolve over time (learning enabled)")
    else:
        print(f"  ⚠ Static x: agent beliefs remain constant (no learning)")
    
    # Initialize agent beliefs (x values) between 0.45 and 0.65
    # These represent agents' estimates of the conditional probability
    agent_beliefs = np.random.uniform(0.45, 0.65, n_agents)
    
    # Initialize agent resources
    agent_resources = np.full(n_agents, initial_resources, dtype=float)
    
    # Storage for trajectories
    resource_trajectories = np.zeros((n_agents, n_timesteps + 1))
    resource_trajectories[:, 0] = agent_resources.copy()
    
    # Storage for belief trajectories
    belief_trajectories = np.zeros((n_agents, n_timesteps + 1))
    belief_trajectories[:, 0] = agent_beliefs.copy()
    
    # Storage for environment outcomes and agent allocations
    environment_outcomes = []
    agent_allocations = []
    agent_signals = []
    
    # Storage for dynamic p and l values at each timestep
    p_t_series = []
    l_t_series = []
    
    # Main simulation loop
    print(f"\nRunning simulation...")
    for t in tqdm(range(n_timesteps), desc="Timesteps"):
        # Sample dynamic l and p for this timestep
        # l: continuous uniform between 1.5 and 2.5
        l = np.random.uniform(1.5, 2.5)
        
        # p: depends on l (negative correlation)
        if l < 2.0:
            p = np.random.uniform(0.6, 0.8)  # l<2.0: p uniform 0.6-0.8
        else:  # l >= 2.0
            p = np.random.uniform(0.4, 0.5)  # l>=2.0: p uniform 0.4-0.5
        
        # Store the dynamic values
        p_t_series.append(p)
        l_t_series.append(l)
        
        timestep_data = {
            'timestep': t,
            'outcomes': [],
            'allocations': [],
            'signals': [],
            'rewards': [],
            'p': p,
            'l': l
        }
        
        # For each agent, simulate their experience
        for agent_id in range(n_agents):
            # Agent's current belief
            x = agent_beliefs[agent_id]
            
            # NEW: Continuous reward system
            # Agent draws random number between 0 and 1
            r = np.random.random()
            
            # Determine if agent "wins" or "loses"
            if r < p:
                # Win: reward = l * x (proportional to confidence)
                reward = l * x
                agent_guessed_correctly = True
            else:
                # Loss: reward = (1-x) * l (proportional to lack of confidence)
                reward = (1 - x) * l
                agent_guessed_correctly = False
            
            # Multiple dice rolls per timestep with geometric mean
            # This suppresses volatility while preserving mean growth rate
            dice_rewards = []
            for _ in range(n_dice_rolls):
                # Each dice roll represents the agent's bet outcome
                # Use the same reward structure but with small noise
                noise = np.random.normal(0, 0.01)
                if agent_guessed_correctly:
                    dice_reward = l * x + noise
                else:
                    dice_reward = (1 - x) * l + noise
                
                # Ensure reward is positive
                dice_reward = max(0.1, dice_reward)
                dice_rewards.append(dice_reward)
            
            # Geometric mean of rewards
            geometric_mean_reward = np.exp(np.mean(np.log(dice_rewards)))
            
            # Update agent resources: r_t = r_{t-1} * geometric_mean_reward
            agent_resources[agent_id] *= geometric_mean_reward
            
            # Store data (simplified for new system)
            timestep_data['outcomes'].append(1 if agent_guessed_correctly else 0)
            timestep_data['allocations'].append([x, 1-x])  # Simplified allocation
            timestep_data['signals'].append(x)  # Signal is now the belief x
            timestep_data['rewards'].append(geometric_mean_reward)
        
        # Store timestep data
        environment_outcomes.append(timestep_data['outcomes'])
        agent_allocations.append(timestep_data['allocations'])
        agent_signals.append(timestep_data['signals'])
        
        # Store resource trajectories
        resource_trajectories[:, t + 1] = agent_resources.copy()
        
        # Update agent beliefs at the end of each timestep
        # Using the learning equation with k=20
        if dynamic_x:
            agent_beliefs = update_agent_beliefs(agent_beliefs, p, l, k=20, timestep=t)
        
        # Store belief trajectories
        belief_trajectories[:, t + 1] = agent_beliefs.copy()
    
    # Report the dynamic l and p time series
    print(f"\nDynamic Environment Parameters:")
    print(f"Timestep | l     | p")
    print(f"---------|-------|-----")
    for t in range(n_timesteps):
        print(f"   {t:2d}    | {l_t_series[t]:.3f} | {p_t_series[t]:.3f}")
    
    # Calculate summary statistics
    print(f"\nSummary Statistics:")
    print(f"l range: {min(l_t_series):.3f} to {max(l_t_series):.3f}")
    print(f"p range: {min(p_t_series):.3f} to {max(p_t_series):.3f}")
    print(f"Mean l: {np.mean(l_t_series):.3f}")
    print(f"Mean p: {np.mean(p_t_series):.3f}")
    
    # Calculate actual growth rates from simulation
    actual_growth_rates = []
    for agent_id in range(n_agents):
        # Calculate growth rate as log(final_resources / initial_resources) / n_timesteps
        final_resources = resource_trajectories[agent_id, -1]
        growth_rate = np.log(final_resources / initial_resources) / n_timesteps
        actual_growth_rates.append(growth_rate)
    
    print(f"\nActual growth rates from simulation:")
    print(f"  Min: {min(actual_growth_rates):.4f}")
    print(f"  Max: {max(actual_growth_rates):.4f}")
    print(f"  Mean: {np.mean(actual_growth_rates):.4f}")
    
    # Calculate theoretical growth rates and compare with empirical
    theoretical_growth_rates = []
    for agent_id in range(n_agents):
        x_initial = belief_trajectories[agent_id, 0]  # Initial belief
        
        # Calculate theoretical growth rate for this agent
        theoretical_growth = 0
        for t in range(n_timesteps):
            p_t = p_t_series[t]
            l_t = l_t_series[t]
            
            # Expected log reward for this timestep
            # Win: log(l * x) with probability p
            # Loss: log((1-x) * l) with probability (1-p)
            expected_log_reward = p_t * np.log(l_t * x_initial) + (1 - p_t) * np.log((1 - x_initial) * l_t)
            theoretical_growth += expected_log_reward
        
        # Average growth rate per timestep
        theoretical_growth /= n_timesteps
        theoretical_growth_rates.append(theoretical_growth)
    
    print(f"\nTheoretical growth rates (based on initial beliefs):")
    print(f"  Min: {min(theoretical_growth_rates):.4f}")
    print(f"  Max: {max(theoretical_growth_rates):.4f}")
    print(f"  Mean: {np.mean(theoretical_growth_rates):.4f}")
    
    # Compare theoretical vs empirical
    theoretical_mean = np.mean(theoretical_growth_rates)
    empirical_mean = np.mean(actual_growth_rates)
    difference = empirical_mean - theoretical_mean
    relative_error = abs(difference) / abs(theoretical_mean) * 100
    
    print(f"\nTheoretical vs Empirical Comparison:")
    print(f"  Theoretical mean: {theoretical_mean:.4f}")
    print(f"  Empirical mean: {empirical_mean:.4f}")
    print(f"  Difference: {difference:.4f}")
    print(f"  Relative error: {relative_error:.2f}%")
    
    # Check if they're close (within 5%)
    if relative_error < 5.0:
        print(f"  ✓ Theoretical and empirical means are close (within 5%)")
    else:
        print(f"  ⚠ Theoretical and empirical means differ by more than 5%")
    
    # Calculate correlation between theoretical and empirical
    correlation = np.corrcoef(theoretical_growth_rates, actual_growth_rates)[0, 1]
    print(f"  Correlation between theoretical and empirical: {correlation:.3f}")
    
    # Prepare data for VI model fitting
    vi_data = []
    
    for agent_id in range(n_agents):
        # Calculate income growth rates (log differences)
        resources = resource_trajectories[agent_id, :]
        log_resources = np.log(resources)
        income_growth = np.diff(log_resources)
        
        # Only include if we have valid growth rates
        if np.all(np.isfinite(income_growth)) and len(income_growth) > 0:
            vi_data.append({
                'agent_id': agent_id,
                'initial_belief': agent_beliefs[agent_id],
                'theoretical_growth_rate': theoretical_growth_rates[agent_id], # Use theoretical growth rate
                'actual_growth_rate': actual_growth_rates[agent_id],
                'income_growth_rates': income_growth,
                'resource_trajectory': resources,
                'final_resources': resources[-1]
            })
    
    # Create results dictionary
    results = {
        'parameters': {
            'n_agents': n_agents,
            'n_timesteps': n_timesteps,
            'initial_resources': initial_resources,
            'n_dice_rolls': n_dice_rolls,
            'seed': seed
        },
        'agent_trajectories': {
            'resources': resource_trajectories,
            'beliefs': belief_trajectories
        },
        'environment_data': {
            'outcomes': environment_outcomes,
            'allocations': agent_allocations,
            'signals': agent_signals
        },
        'p_t_series': p_t_series,
        'l_t_series': l_t_series,
        'n_timesteps': n_timesteps,
        'n_agents': n_agents,
        'vi_data': vi_data,  # Add back the vi_data structure
        # Add old keys for backward compatibility with plotting functions
        'resource_trajectories': resource_trajectories,
        'belief_trajectories': belief_trajectories
    }
    
    return results

def save_dummy_data(results, suffix=""):
    """Save the dummy data to a pickle file."""
    output_path = f"dummy_data_kelly_betting{suffix}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved dummy data to {output_path}")
    
    # Also save a summary CSV
    summary_data = []
    for agent_data in results['vi_data']:
        summary_data.append({
            'agent_id': agent_data['agent_id'],
            'initial_belief': agent_data['initial_belief'],
            'theoretical_growth_rate': agent_data['theoretical_growth_rate'],
            'actual_growth_rate': agent_data['actual_growth_rate'],
            'final_resources': agent_data['final_resources']
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = output_path.replace('.pkl', '_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")

def plot_dummy_data_results(results, save_path="dummy_data_plots"):
    """
    Plot comprehensive results from the dummy data generation.
    
    Args:
        results: Results dictionary from generate_dummy_data
        save_path: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print(f"\nGenerating plots in '{save_path}' directory...")
        
        # Create comprehensive figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Resource trajectories (log scale)
        plt.subplot(3, 3, 1)
        resource_trajectories = results['agent_trajectories']['resources']
        n_agents, n_timesteps = resource_trajectories.shape
        timesteps = np.arange(n_timesteps)
        
        # Plot individual trajectories (sample)
        for i in range(min(15, n_agents)):
            plt.plot(timesteps, resource_trajectories[i, :], alpha=0.4, linewidth=1)
        
        # Plot mean trajectory
        mean_resources = np.mean(resource_trajectories, axis=0)
        plt.plot(timesteps, mean_resources, 'r-', linewidth=3, label='Population Mean')
        
        plt.xlabel('Timestep')
        plt.ylabel('Resources ($)')
        plt.title('Resource Trajectories (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Belief evolution over time
        plt.subplot(3, 3, 2)
        belief_trajectories = results['agent_trajectories']['beliefs']
        
        # Plot individual belief trajectories (sample)
        for i in range(min(15, n_agents)):
            plt.plot(timesteps, belief_trajectories[i, :], alpha=0.4, linewidth=1)
        
        # Plot mean belief trajectory
        mean_beliefs = np.mean(belief_trajectories, axis=0)
        plt.plot(timesteps, mean_beliefs, 'b-', linewidth=3, label='Population Mean')
        
        # Add true p value
        p = np.mean(results['p_t_series'])  # Use the mean of all sampled p values
        plt.axhline(y=p, color='g', linestyle='--', alpha=0.8, label=f'Mean p = {p:.3f}')
        
        plt.xlabel('Timestep')
        plt.ylabel('Belief (x)')
        plt.title('Agent Belief Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. p_t series
        plt.subplot(3, 3, 3)
        p_t_series = results['p_t_series']
        timesteps_p = np.arange(len(p_t_series))
        
        plt.plot(timesteps_p, p_t_series, 'b-', linewidth=2, marker='o', markersize=4)
        plt.axhline(y=p, color='g', linestyle='--', alpha=0.8, label=f'Mean p = {p:.3f}')
        plt.xlabel('Timestep')
        plt.ylabel('Fraction of Agents with Gains')
        plt.title('p_t Series: Dynamic Environment Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(save_path, 'p_t_series.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  p_t series plot saved to {save_path}/p_t_series.png")
        
        # Plot dynamic l values
        plt.figure(figsize=(10, 6))
        l_t_series = results['l_t_series']  # Get l_t_series from results
        timesteps_l = np.arange(len(l_t_series))
        plt.plot(timesteps_l, l_t_series, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Timestep')
        plt.ylabel('Number of Outcomes (l)')
        plt.title('Dynamic l Series: Number of Outcomes Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(1.5, 3.5)  # Adjust to show l=2 and l=3 clearly
        plt.yticks([2, 3])  # Only show the discrete values
        plt.savefig(os.path.join(save_path, 'l_t_series.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  l_t series plot saved to {save_path}/l_t_series.png")
        
        # 4. Growth rate distribution
        plt.subplot(3, 3, 4)
        actual_growth_rates = results['actual_growth_rates']
        # theoretical_growth_rates = results['theoretical_growth_rates'] # No longer available
        
        plt.hist(actual_growth_rates, bins=20, alpha=0.7, label='Actual', density=True)
        # plt.hist(theoretical_growth_rates, bins=20, alpha=0.7, label='Theoretical', density=True) # No longer available
        plt.xlabel('Growth Rate')
        plt.ylabel('Density')
        plt.title('Distribution of Growth Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Theoretical vs Actual growth rates
        plt.subplot(3, 3, 5)
        # plt.scatter(theoretical_growth_rates, actual_growth_rates, alpha=0.7, s=50) # No longer available
        
        # Add perfect prediction line
        min_rate = min(min(actual_growth_rates), min(actual_growth_rates)) # Use actual_growth_rates for both
        max_rate = max(max(actual_growth_rates), max(actual_growth_rates)) # Use actual_growth_rates for both
        plt.plot([min_rate, max_rate], [min_rate, max_rate], 'r--', alpha=0.8, label='Perfect Match')
        
        plt.xlabel('Theoretical Growth Rate')
        plt.ylabel('Actual Growth Rate')
        plt.title('Theoretical vs Actual Growth Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Initial vs Final beliefs
        plt.subplot(3, 3, 6)
        initial_beliefs = belief_trajectories[:, 0]
        final_beliefs = belief_trajectories[:, -1]
        
        plt.scatter(initial_beliefs, final_beliefs, alpha=0.7, s=50)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='No Change')
        plt.axhline(y=p, color='g', linestyle='--', alpha=0.8, label=f'True p = {p}')
        plt.xlabel('Initial Belief (x₀)')
        plt.ylabel('Final Belief (x_T)')
        plt.title('Initial vs Final Beliefs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Final resources vs initial beliefs
        plt.subplot(3, 3, 7)
        final_resources = resource_trajectories[:, -1]
        
        plt.scatter(initial_beliefs, final_resources, alpha=0.7, s=50)
        plt.xlabel('Initial Belief (x₀)')
        plt.ylabel('Final Resources ($)')
        plt.yscale('log')
        plt.title('Final Resources vs Initial Beliefs')
        plt.grid(True, alpha=0.3)
        
        # 8. Belief convergence over time
        plt.subplot(3, 3, 8)
        convergence = np.abs(belief_trajectories - p)
        mean_convergence = np.mean(convergence, axis=0)
        std_convergence = np.std(convergence, axis=0)
        
        plt.plot(timesteps, mean_convergence, 'b-', linewidth=2, label='Mean Error')
        plt.fill_between(timesteps, 
                        mean_convergence - std_convergence, 
                        mean_convergence + std_convergence, 
                        alpha=0.3, label='±1 Std Dev')
        plt.xlabel('Timestep')
        plt.ylabel('|Belief - True p|')
        plt.title('Convergence to True Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Summary statistics
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Calculate summary statistics
        total_initial = results['parameters']['n_agents'] * results['parameters']['initial_resources']
        total_final = sum(agent_data['final_resources'] for agent_data in results['vi_data'])
        overall_growth = total_final / total_initial
        
        # Calculate correlation
        # correlation = np.corrcoef(theoretical_growth_rates, actual_growth_rates)[0, 1] # No longer available
        
        # Calculate belief convergence
        final_belief_std = np.std(final_beliefs)
        belief_convergence = np.mean(np.abs(final_beliefs - p))
        
        stats_text = f"""Simulation Summary
        
Parameters:
n_agents = {results['parameters']['n_agents']}
n_timesteps = {results['parameters']['n_timesteps']}
initial_resources = {results['parameters']['initial_resources']:,}
n_dice_rolls = {results['parameters']['n_dice_rolls']}

Results:
Total Resources: ${total_initial:,.0f} → ${total_final:,.0f}
Overall Growth: {overall_growth:.2f}x
Growth Rate Correlation: N/A (theoretical growth rate not available)
Final Belief Std: {final_belief_std:.3f}
Belief Convergence Error: {belief_convergence:.3f}
p_t Series Mean: {np.mean(p_t_series):.3f}"""
        
        plt.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plt.savefig(os.path.join(save_path, 'dummy_data_comprehensive.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Comprehensive plot saved to {save_path}/dummy_data_comprehensive.png")
        
        # Create additional focused plots
        
        # Resource trajectories only
        plt.figure(figsize=(12, 8))
        for i in range(min(20, n_agents)):
            plt.plot(timesteps, resource_trajectories[i, :], alpha=0.3, linewidth=1)
        plt.plot(timesteps, mean_resources, 'r-', linewidth=3, label='Population Mean')
        plt.xlabel('Timestep')
        plt.ylabel('Resources ($)')
        plt.title('Resource Trajectories (l=2, p=0.7)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'resource_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Resource trajectories plot saved to {save_path}/resource_trajectories.png")
        
        # Belief evolution only
        plt.figure(figsize=(12, 8))
        for i in range(min(20, n_agents)):
            plt.plot(timesteps, belief_trajectories[i, :], alpha=0.3, linewidth=1)
        plt.plot(timesteps, mean_beliefs, 'b-', linewidth=3, label='Population Mean')
        plt.axhline(y=p, color='g', linestyle='--', alpha=0.8, label=f'True p = {p}')
        plt.xlabel('Timestep')
        plt.ylabel('Belief (x)')
        plt.title('Agent Belief Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'belief_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Belief evolution plot saved to {save_path}/belief_evolution.png")
        
        # p_t series only
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps_p, p_t_series, 'b-', linewidth=2, marker='o', markersize=4)
        plt.axhline(y=p, color='g', linestyle='--', alpha=0.8, label=f'True p = {p}')
        plt.xlabel('Timestep')
        plt.ylabel('Fraction of Agents with Gains')
        plt.title('p_t Series: Fraction of Agents Experiencing Gains')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(save_path, 'p_t_series.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  p_t series plot saved to {save_path}/p_t_series.png")
        
        print(f"\nAll plots generated successfully in '{save_path}' directory!")
        
    except ImportError:
        print("  matplotlib not available, skipping plots")
    except Exception as e:
        print(f"  Plotting failed: {str(e)}")
        print("  This might be due to matplotlib not being available or other issues")

def main():
    """Main function to generate and save dummy data."""
    
    # Toggle between dynamic and static x data
    USE_DYNAMIC_X = True  # Set to False for static beliefs, True for dynamic beliefs
    
    if USE_DYNAMIC_X:
        print("=" * 60)
        print("GENERATING DYNAMIC X DATA (agent beliefs evolve over time)")
        print("=" * 60)
        
        # Generate dummy data with dynamic beliefs (learning enabled)
        results = generate_dummy_data(
            n_agents=3500,           # 1000 agents
            n_timesteps=11,         # 11 timesteps
            initial_resources=50000, # $50,000 initial resources
            n_dice_rolls=10,         # 2 dice rolls per timestep
            seed=np.random.randint(1, 10000),  # Random seed each run2,                # Reproducible random seed
            dynamic_x=True          # Enable belief updates (learning)
        )
        
        # Save the dynamic data
        save_dummy_data(results, suffix="_dynamic_x")
        
    else:
        print("=" * 60)
        print("GENERATING STATIC X DATA (agent beliefs remain constant)")
        print("=" * 60)
        
        # Generate dummy data with static beliefs (no learning)
        results = generate_dummy_data(
            n_agents=1000,           # 1000 agents
            n_timesteps=11,         # 11 timesteps
            initial_resources=50000, # $50,000 initial resources
            n_dice_rolls=2,         # 2 dice rolls per timestep
            seed=42,                # Reproducible random seed
            dynamic_x=False         # Disable belief updates (no learning)
        )
        
        # Save the static data
        save_dummy_data(results, suffix="_static_x")
    
    # Plot the results
    plot_dummy_data_results(results)
    
    # Print validation statistics
    print(f"\nValidation Statistics:")
    print(f"  Total agents processed: {len(results['vi_data'])}")
    print(f"  Resource trajectories shape: {results['agent_trajectories']['resources'].shape}")
    print(f"  p_t series length: {len(results['p_t_series'])}")
    
    # Check if any agents went bankrupt
    bankrupt_agents = sum(1 for agent_data in results['vi_data'] 
                         if agent_data['final_resources'] <= 0)
    print(f"  Bankrupt agents: {bankrupt_agents}")
    
    # Check resource growth
    total_initial = results['parameters']['n_agents'] * results['parameters']['initial_resources']
    total_final = sum(agent_data['final_resources'] for agent_data in results['vi_data'])
    print(f"  Total resources: ${total_initial:,.0f} → ${total_final:,.0f}")
    print(f"  Overall growth factor: {total_final/total_initial:.2f}x")
    
    # Print x behavior summary
    if USE_DYNAMIC_X:
        print(f"  Agent beliefs: Dynamic (evolving over time)")
    else:
        print(f"  Agent beliefs: Static (constant over time)")
    
    # Final toggle report
    print(f"\n" + "=" * 60)
    print("DATA GENERATOR TOGGLE STATUS")
    print("=" * 60)
    print(f"Toggle Setting: USE_DYNAMIC_X = {USE_DYNAMIC_X}")
    if USE_DYNAMIC_X:
        print(f"Data Type Generated: DYNAMIC X (agent beliefs evolve over time)")
        print(f"Learning: ENABLED (agents update beliefs based on outcomes)")
        print(f"Use Case: Realistic agent behavior simulation")
    else:
        print(f"Data Type Generated: STATIC X (agent beliefs remain constant)")
        print(f"Learning: DISABLED (beliefs fixed at initial values)")
        print(f"Use Case: Validation of reward mechanism")
    print("=" * 60)

if __name__ == "__main__":
    main() 