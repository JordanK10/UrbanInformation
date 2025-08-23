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

def generate_dummy_data(l=2, p=0.7, n_agents=100, n_timesteps=50, 
                       initial_resources=50000, n_dice_rolls=3, seed=42):
    """
    Generate dummy data for calibrating the VI model based on Kelly betting theory.
    
    Args:
        l: Number of possible outcomes (l=2 for binary choice)
        p: True probability of correct prediction (environment parameter)
        n_agents: Number of agents in the simulation
        n_timesteps: Number of time steps to simulate
        initial_resources: Initial resources for each agent
        n_dice_rolls: Number of dice rolls per timestep (for geometric mean)
        seed: Random seed for reproducibility
    
    Returns:
        dict: Contains agent trajectories, environment outcomes, and parameters
    """
    np.random.seed(seed)
    
    print(f"Generating dummy data with parameters:")
    print(f"  l = {l} (number of outcomes)")
    print(f"  p = {p} (true probability)")
    print(f"  n_agents = {n_agents}")
    print(f"  n_timesteps = {n_timesteps}")
    print(f"  initial_resources = {initial_resources:,}")
    print(f"  n_dice_rolls = {n_dice_rolls}")
    
    # Initialize agent beliefs (x values) between 0.5 and 0.7
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
    
    # Calculate theoretical growth rates for each agent
    theoretical_growth_rates = []
    for x in agent_beliefs:
        # From theory: γ = log(l) + p*log(x) + (1-p)*log((1-x)/(l-1))
        gamma = np.log(l) + p * np.log(x) + (1-p) * np.log((1-x)/(l-1))
        theoretical_growth_rates.append(gamma)
    
    print(f"\nTheoretical growth rates:")
    print(f"  Min: {min(theoretical_growth_rates):.4f}")
    print(f"  Max: {max(theoretical_growth_rates):.4f}")
    print(f"  Mean: {np.mean(theoretical_growth_rates):.4f}")
    
    # Main simulation loop
    print(f"\nRunning simulation...")
    for t in tqdm(range(n_timesteps), desc="Timesteps"):
        timestep_data = {
            'timestep': t,
            'outcomes': [],
            'allocations': [],
            'signals': [],
            'rewards': []
        }
        
        # For each agent, simulate their experience
        for agent_id in range(n_agents):
            # Agent observes a signal (uniform random from 1 to l)
            signal = np.random.randint(1, l + 1)
            
            # Agent allocates resources based on their belief x
            x = agent_beliefs[agent_id]
            
            # Allocation strategy: f(x,l) from theory
            # If signal = outcome: allocate x proportion
            # If signal ≠ outcome: allocate (1-x)/(l-1) proportion
            allocation = np.zeros(l)
            for outcome in range(1, l + 1):
                if outcome == signal:
                    allocation[outcome-1] = x
                else:
                    allocation[outcome-1] = (1 - x) / (l - 1)
            
            # Environment reveals true outcome
            # For l=2, p=0.7: 70% chance signal matches outcome, 30% chance it doesn't
            if np.random.random() < p:
                # Signal matches outcome
                true_outcome = signal
            else:
                # Signal doesn't match outcome (uniform random among other outcomes)
                other_outcomes = [o for o in range(1, l + 1) if o != signal]
                true_outcome = np.random.choice(other_outcomes)
            
            # Determine if agent guessed correctly
            agent_guessed_correctly = (signal == true_outcome)
            
            # In Kelly betting with fair odds, reward multiplier is always l
            reward_multiplier = l  # Always 2 for fair odds
            
            # Multiple dice rolls per timestep with geometric mean
            # This suppresses volatility while preserving mean growth rate
            dice_rewards = []
            for _ in range(n_dice_rolls):
                # Each dice roll represents the agent's bet outcome
                # If agent guessed correctly: allocate x fraction, get reward l*x
                # If agent guessed incorrectly: allocate (1-x) fraction, get reward l*(1-x)
                if agent_guessed_correctly:
                    # Agent allocated x to the correct outcome
                    allocation_to_outcome = x
                else:
                    # Agent allocated (1-x)/(l-1) to the correct outcome
                    allocation_to_outcome = (1 - x) / (l - 1)
                
                # Reward is l * allocation_to_outcome
                dice_reward = l * allocation_to_outcome
                dice_rewards.append(dice_reward)
            
            # Geometric mean of rewards
            geometric_mean_reward = np.exp(np.mean(np.log(dice_rewards)))
            
            # Update agent resources: r_t = r_{t-1} * geometric_mean_reward
            agent_resources[agent_id] *= geometric_mean_reward
            
            # Store data
            timestep_data['outcomes'].append(true_outcome)
            timestep_data['allocations'].append(allocation)
            timestep_data['signals'].append(signal)
            timestep_data['rewards'].append(geometric_mean_reward)
        
        # Store timestep data
        environment_outcomes.append(timestep_data['outcomes'])
        agent_allocations.append(timestep_data['allocations'])
        agent_signals.append(timestep_data['signals'])
        
        # Store resource trajectories
        resource_trajectories[:, t + 1] = agent_resources.copy()
        
        # Update agent beliefs at the end of each timestep
        # Using the learning equation with k=5
        agent_beliefs = update_agent_beliefs(agent_beliefs, p, l, k=20, timestep=t)
        
        # Store belief trajectories
        belief_trajectories[:, t + 1] = agent_beliefs.copy()
    
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
    
    # Calculate p_t series (fraction of agents experiencing gains each timestep)
    p_t_series = []
    for t in range(n_timesteps):
        gains = resource_trajectories[:, t+1] > resource_trajectories[:, t]
        p_t = np.mean(gains)
        p_t_series.append(p_t)
    
    print(f"\np_t series (fraction of agents with gains):")
    print(f"  Min: {min(p_t_series):.3f}")
    print(f"  Max: {max(p_t_series):.3f}")
    print(f"  Mean: {np.mean(p_t_series):.3f}")
    
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
                'theoretical_growth_rate': theoretical_growth_rates[agent_id],
                'actual_growth_rate': actual_growth_rates[agent_id],
                'income_growth_rates': income_growth,
                'resource_trajectory': resources,
                'final_resources': resources[-1]
            })
    
    # Create results dictionary
    results = {
        'parameters': {
            'l': l,
            'p': p,
            'n_agents': n_agents,
            'n_timesteps': n_timesteps,
            'initial_resources': initial_resources,
            'n_dice_rolls': n_dice_rolls,
            'seed': seed
        },
        'agent_beliefs': agent_beliefs,
        'theoretical_growth_rates': theoretical_growth_rates,
        'actual_growth_rates': actual_growth_rates,
        'resource_trajectories': resource_trajectories,
        'belief_trajectories': belief_trajectories,
        'p_t_series': p_t_series,
        'environment_outcomes': environment_outcomes,
        'agent_allocations': agent_allocations,
        'agent_signals': agent_signals,
        'vi_data': vi_data
    }
    
    return results

def save_dummy_data(results, output_path="dummy_data_kelly_betting.pkl"):
    """Save the dummy data to a pickle file."""
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
        resource_trajectories = results['resource_trajectories']
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
        belief_trajectories = results['belief_trajectories']
        
        # Plot individual belief trajectories (sample)
        for i in range(min(15, n_agents)):
            plt.plot(timesteps, belief_trajectories[i, :], alpha=0.4, linewidth=1)
        
        # Plot mean belief trajectory
        mean_beliefs = np.mean(belief_trajectories, axis=0)
        plt.plot(timesteps, mean_beliefs, 'b-', linewidth=3, label='Population Mean')
        
        # Add true p value
        p = results['parameters']['p']
        plt.axhline(y=p, color='g', linestyle='--', alpha=0.8, label=f'True p = {p}')
        
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
        plt.axhline(y=p, color='g', linestyle='--', alpha=0.8, label=f'True p = {p}')
        plt.xlabel('Timestep')
        plt.ylabel('Fraction of Agents with Gains')
        plt.title('p_t Series: Fraction of Agents Experiencing Gains')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 4. Growth rate distribution
        plt.subplot(3, 3, 4)
        actual_growth_rates = results['actual_growth_rates']
        theoretical_growth_rates = results['theoretical_growth_rates']
        
        plt.hist(actual_growth_rates, bins=20, alpha=0.7, label='Actual', density=True)
        plt.hist(theoretical_growth_rates, bins=20, alpha=0.7, label='Theoretical', density=True)
        plt.xlabel('Growth Rate')
        plt.ylabel('Density')
        plt.title('Distribution of Growth Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Theoretical vs Actual growth rates
        plt.subplot(3, 3, 5)
        plt.scatter(theoretical_growth_rates, actual_growth_rates, alpha=0.7, s=50)
        
        # Add perfect prediction line
        min_rate = min(min(theoretical_growth_rates), min(actual_growth_rates))
        max_rate = max(max(theoretical_growth_rates), max(actual_growth_rates))
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
        correlation = np.corrcoef(theoretical_growth_rates, actual_growth_rates)[0, 1]
        
        # Calculate belief convergence
        final_belief_std = np.std(final_beliefs)
        belief_convergence = np.mean(np.abs(final_beliefs - p))
        
        stats_text = f"""Simulation Summary
        
Parameters:
l = {results['parameters']['l']}
p = {results['parameters']['p']}
n_agents = {results['parameters']['n_agents']}
n_timesteps = {results['parameters']['n_timesteps']}
n_dice_rolls = {results['parameters']['n_dice_rolls']}

Results:
Total Resources: ${total_initial:,.0f} → ${total_final:,.0f}
Overall Growth: {overall_growth:.2f}x
Growth Rate Correlation: {correlation:.3f}
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
    
    # Generate dummy data with default parameters
    results = generate_dummy_data(
        l=2,                    # Binary choice environment
        p=0.75,                  # 70% probability of correct prediction
        n_agents=100,           # 100 agents (scaled up for testing frequentist p)
        n_timesteps=11,         # 50 timesteps
        initial_resources=50000, # $50,000 initial resources
        n_dice_rolls=2,         # 3 dice rolls per timestep
        seed=42                 # Reproducible random seed
    )
    
    # Save the data
    save_dummy_data(results)
    
    # Plot the results
    plot_dummy_data_results(results)
    
    # Print some validation statistics
    print(f"\nValidation Statistics:")
    print(f"  Total agents processed: {len(results['vi_data'])}")
    print(f"  Resource trajectories shape: {results['resource_trajectories'].shape}")
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

if __name__ == "__main__":
    main() 