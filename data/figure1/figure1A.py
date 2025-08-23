#!/usr/bin/env python3
"""
Figure 1A: Log Incomes vs Growth Rates
======================================

This script creates Figure 1A for the paper, showing the relationship between
log incomes and growth rates from the synthetic data.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from pathlib import Path

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_dummy_data(data_path):
    """
    Load the dummy data from the pickle file.
    
    Parameters:
    -----------
    data_path : str
        Path to the dummy data pickle file
        
    Returns:
    --------
    dict
        Loaded dummy data
    """
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded dummy data from {data_path}")
        return data
    except Exception as e:
        print(f"✗ Error loading dummy data: {e}")
        return None

def calculate_growth_rates(resources):
    """
    Calculate growth rates from resource trajectories.
    
    Parameters:
    -----------
    resources : np.ndarray
        Array of shape (n_agents, n_timesteps + 1) with resource values
        
    Returns:
    --------
    np.ndarray
        Array of shape (n_agents, n_timesteps) with growth rates
    """
    # Calculate growth rates: (r_t - r_{t-1}) / r_{t-1}
    growth_rates = np.diff(resources, axis=1) / resources[:, :-1]
    return growth_rates

def create_figure1a(dummy_data, save_path="figure1"):
    """
    Create Figure 1A: Time Series of Income vs Growth Rates.
    
    Parameters:
    -----------
    dummy_data : dict
        Loaded dummy data
    save_path : str
        Directory to save the figure
    """
    # Extract data
    resources = dummy_data['agent_trajectories']['resources']  # Shape: (n_agents, n_timesteps + 1)
    n_agents, n_timesteps_plus_1 = resources.shape
    n_timesteps = n_timesteps_plus_1 - 1
    
    print(f"Data shape: {n_agents} agents, {n_timesteps} timesteps")
    
    # Calculate growth rates
    growth_rates = calculate_growth_rates(resources)  # Shape: (n_agents, n_timesteps)
    
    # Calculate log incomes (using resources at each timestep)
    log_incomes = np.log(resources)  # Shape: (n_agents, n_timesteps + 1)
    
    # Calculate cumulative growth rates for coloring
    cumulative_growth_rates = np.zeros(n_agents)
    for agent_id in range(n_agents):
        # Calculate total growth from start to end
        initial_resources = resources[agent_id, 0]
        final_resources = resources[agent_id, -1]
        cumulative_growth_rates[agent_id] = (np.log(final_resources) - np.log(initial_resources)) 
    
    # Convert to relative growth rate (percentile rank) for coloring
    from scipy.stats import rankdata
    relative_growth_ranks = rankdata(cumulative_growth_rates) / len(cumulative_growth_rates)  # 0 to 1
    
    # Sort agents by cumulative growth rate for consistent coloring
    sorted_indices = np.argsort(cumulative_growth_rates)
    
    # Create the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create time array for x-axis
    time_steps = np.arange(n_timesteps + 1)
    time_steps_growth = np.arange(1, n_timesteps + 1)  # Growth rates start from timestep 1
    
    # Custom color spectrum as requested - map to relative growth ranks
    # custom_colors1 = ['#0F0F11', '#AD8F78', '#C7D4E4']
    custom_colors1 = ['#172741', '#5F8089', '#C2C4B8', '#C3A983', '#483326']
    
    # Plot 1: Time series of log incomes for all agents
    print("Plotting log income time series...")
    print(f"Cumulative growth rates range: {cumulative_growth_rates.min():.3f} to {cumulative_growth_rates.max():.3f}")
    print(f"Using relative growth ranks (percentiles) for coloring")
    
    # Create color mapping based on relative growth ranks
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', custom_colors1)
    
    # Normalize relative growth ranks for color mapping (0 to 1)
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)
    
    for agent_id in range(n_agents):
        # Color based on relative growth rank (percentile)
        color = cmap(norm(relative_growth_ranks[agent_id]))
        ax1.plot(time_steps, log_incomes[agent_id, :], 
                color=color, alpha=0.3, linewidth=0.8)
    
    # Plot mean trajectory
    mean_log_income = np.mean(log_incomes, axis=0)
    ax1.plot(time_steps, mean_log_income, 'k-', linewidth=3, alpha=0.8, 
             label=f'Mean (n={n_agents})')
    
    # Plot median trajectory
    median_log_income = np.median(log_incomes, axis=0)
    ax1.plot(time_steps, median_log_income, 'r--', linewidth=2, alpha=0.8, 
             label=f'Median (n={n_agents})')
    

    ax1.legend()
    
    # Plot 2: Time series of growth rates for all agents
    print("Plotting growth rate time series...")
    
    for agent_id in range(n_agents):
        # Color based on relative growth rank (percentile)
        color = cmap(norm(relative_growth_ranks[agent_id]))
        ax2.plot(time_steps_growth, growth_rates[agent_id, :], 
                color=color, alpha=0.3, linewidth=0.8)
    
    # Plot mean growth rate trajectory
    mean_growth_rate = np.mean(growth_rates, axis=0)
    ax2.plot(time_steps_growth, mean_growth_rate, 'k-', linewidth=3, alpha=0.8, 
             label=f'Mean (n={n_agents})')
    
    # Plot median growth rate trajectory
    median_growth_rate = np.median(growth_rates, axis=0)
    ax2.plot(time_steps_growth, median_growth_rate, 'r--', linewidth=2, alpha=0.8, 
             label=f'Median (n={n_agents})')
    
   
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    
    # Add colorbar to show cumulative growth rate mapping
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], shrink=0.8, aspect=30)
    cbar.set_label('Relative Growth Rank (Percentile)', fontsize=12, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Figure 1A: Time Series of Income and Growth Rates\nSynthetic Agent Data', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Add statistics text
    stats_text = f'Agents: {n_agents}\nTimesteps: {n_timesteps}\nTotal observations: {n_agents * n_timesteps}'
    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    figure_path = os.path.join(save_path, 'figure1A_time_series.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 1A saved to: {figure_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(save_path, 'figure1A_time_series.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Figure 1A PDF saved to: {pdf_path}")
    
    
    return fig

def main():
    """Main function to create Figure 1A."""
    print("=" * 60)
    print("CREATING FIGURE 1A: INCOME AND GROWTH RATE TIME SERIES")
    print("=" * 60)
    
    # Path to dummy data
    data_path = "validation/validation_dynamic_p_l/dummy_data_kelly_betting_dynamic_x.pkl"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("Please run generate_dummy_data.py first to create the data.")
        return
    
    # Load dummy data
    dummy_data = load_dummy_data(data_path)
    if dummy_data is None:
        return
    
    # Create Figure 1A
    print("\nCreating Figure 1A...")
    fig = create_figure1a(dummy_data)
    
    print("\n" + "=" * 60)
    print("FIGURE 1A CREATION COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - figure1/figure1A_time_series.png")
    print("  - figure1/figure1A_time_series.pdf")
    print("\nThe figure shows time series of:")
    print("  - Top: Log income trajectories for all agents")
    print("  - Bottom: Growth rate trajectories for all agents")
    print("  - Mean and median trajectories highlighted")

if __name__ == "__main__":
    main() 