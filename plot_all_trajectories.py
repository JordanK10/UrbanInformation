import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_summary_statistics(data_path="cbsa_summary_statistics.pkl"):
    """
    Loads summary statistics and generates plots comparing all CBSAs.

    Args:
        data_path (str): Path to the input .pkl file containing the summary statistics.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        return

    with open(data_path, 'rb') as f:
        all_stats = pickle.load(f)

    if not all_stats:
        print("No data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.suptitle("Cross-CBSA Summary Statistics", fontsize=20)

    # --- Plotting Trajectories ---
    ax_p = axes[0, 0]
    ax_x = axes[0, 1]
    ax_gamma = axes[1, 0]
    ax_divergence = axes[1, 1] # New axis for divergence
    
    losses = []

    for city_name, stats in all_stats.items():
        # Plot p_t trajectory
        if not stats['p_t_series'].empty:
            stats['p_t_series'].plot(ax=ax_p, alpha=0.6, linewidth=1.5)

        # Plot x_t trajectory
        if not stats['mean_x_trajectory'].empty:
            stats['mean_x_trajectory'].plot(ax=ax_x, alpha=0.6, linewidth=1.5)

        # Plot gamma trajectories
        if not stats['optimal_gamma'].empty:
            stats['optimal_gamma'].plot(ax=ax_gamma, alpha=0.6, linestyle='--', linewidth=1.5)
        if not stats['mean_agent_gamma'].empty:
            stats['mean_agent_gamma'].plot(ax=ax_gamma, alpha=0.6, linestyle='-', linewidth=1.5, label=city_name if len(all_stats) < 10 else None)
        
        # Plot divergence trajectory
        if 'mean_divergence' in stats and not stats['mean_divergence'].empty:
            stats['mean_divergence'].plot(ax=ax_divergence, alpha=0.6, linewidth=1.5)

        if stats['mean_loss'] is not None:
            losses.append(stats['mean_loss'])

    # --- Formatting ---
    ax_p.set_title("Environmental Predictability ($p_t$)", fontsize=14)
    ax_p.set_xlabel("Year", fontsize=12)
    ax_p.set_ylabel("Predictability ($p_t$)", fontsize=12)
    ax_p.set_ylim(0, 1)

    ax_x.set_title("Mean Agent Belief ($x_t$)", fontsize=14)
    ax_x.set_xlabel("Year", fontsize=12)
    ax_x.set_ylabel("Belief ($x_t$)", fontsize=12)
    ax_x.set_ylim(0, 1)

    ax_gamma.set_title("Mean Growth Rates ($\gamma_t$)", fontsize=14)
    ax_gamma.set_xlabel("Year", fontsize=12)
    ax_gamma.set_ylabel("Growth Rate (nats/period)", fontsize=12)
    
    # Custom legend for growth plot
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='gray', linestyle='--', label='Optimal'),
                       Line2D([0], [0], color='gray', linestyle='-', label='Agent')]
    ax_gamma.legend(handles=legend_elements, title="Growth Type")


    # --- Divergence Plot Formatting ---
    ax_divergence.set_title("Mean Information Divergence", fontsize=14)
    ax_divergence.set_xlabel("Year", fontsize=12)
    ax_divergence.set_ylabel("Divergence ($D_{KL}$)", fontsize=12)


    # Save the plot
    output_filename = "cross_cbsa_summary_plot.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f"Saved summary plot to {output_filename}")

    # --- Second Plot: Mean, CI, and Variance ---
    
    # 1. Collect all series into lists
    p_series_list = [stats['p_t_series'] for stats in all_stats.values() if not stats['p_t_series'].empty]
    x_series_list = [stats['mean_x_trajectory'] for stats in all_stats.values() if not stats['mean_x_trajectory'].empty]
    optimal_gamma_series_list = [stats['optimal_gamma'] for stats in all_stats.values() if not stats['optimal_gamma'].empty]
    agent_gamma_series_list = [stats['mean_agent_gamma'] for stats in all_stats.values() if not stats['mean_agent_gamma'].empty]
    divergence_series_list = [stats['mean_divergence'] for stats in all_stats.values() if 'mean_divergence' in stats and not stats['mean_divergence'].empty]

    # 2. Align data by concatenating into DataFrames
    df_p = pd.concat(p_series_list, axis=1)
    df_x = pd.concat(x_series_list, axis=1)
    df_optimal_gamma = pd.concat(optimal_gamma_series_list, axis=1)
    df_agent_gamma = pd.concat(agent_gamma_series_list, axis=1)
    df_divergence = pd.concat(divergence_series_list, axis=1)

    # 3. Create the second figure
    fig2, axes2 = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
    fig2.suptitle("Aggregated Cross-CBSA Statistics", fontsize=20)

    def plot_mean_ci_var(ax, df, title, ylabel, color):
        """Helper function to plot mean, CI, and variance inset."""
        mean = df.mean(axis=1)
        ci_low = df.quantile(0.025, axis=1)
        ci_high = df.quantile(0.975, axis=1)
        variance = df.var(axis=1)

        mean.plot(ax=ax, color=color, linewidth=2.5, label='Mean')
        ax.fill_between(mean.index, ci_low, ci_high, color=color, alpha=0.2, label='95% CI')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()

        # Inset for variance
        ax_inset = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
        variance.plot(ax=ax_inset, color='black', linewidth=2)
        ax_inset.set_title("Variance", fontsize=10)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax_inset.tick_params(axis='x', labelsize=8)
        ax_inset.tick_params(axis='y', labelsize=8)

    # Plot for each metric
    plot_mean_ci_var(axes2[0, 0], df_p, "Mean Environmental Predictability ($p_t$)", "Predictability ($p_t$)", "blue")
    axes2[0, 0].set_ylim(0, 1)

    plot_mean_ci_var(axes2[0, 1], df_x, "Mean Agent Belief ($x_t$)", "Belief ($x_t$)", "crimson")
    axes2[0, 1].set_ylim(0, 1)

    plot_mean_ci_var(axes2[0, 2], df_optimal_gamma, "Mean Information ($I_t$)", "Growth Rate", "green")
    
    plot_mean_ci_var(axes2[1, 0], df_agent_gamma, "Mean Agent Growth Rate ($\gamma_t$)", "Growth Rate", "purple")

    plot_mean_ci_var(axes2[1, 1], df_divergence, "Mean Information Divergence", "Divergence ($D_{KL}$)", "orange")

    # --- Histogram of Losses ---
    ax_loss = axes2[1, 2]
    if losses:
        ax_loss.hist(losses, bins=30, color='black', alpha=0.7, density=True)
        mean_loss = np.mean(losses)
        ax_loss.axvline(mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.2f}')
        ax_loss.legend()
    ax_loss.set_title("Distribution of Mean Model Loss", fontsize=14)
    ax_loss.set_xlabel("Mean Loss", fontsize=12)
    ax_loss.set_ylabel("Density", fontsize=12)


    # Save the second plot
    output_filename_2 = "cross_cbsa_aggregated_plot.pdf"
    plt.savefig(output_filename_2, format='pdf', bbox_inches='tight')

    print(f"Saved aggregated statistics plot to {output_filename_2}")
    plt.show()


if __name__ == "__main__":
    plot_summary_statistics()
