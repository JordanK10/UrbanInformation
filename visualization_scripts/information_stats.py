import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.collections import LineCollection

def analyze_information_stats(data_path="cbsa_summary_statistics.pkl"):
    """
    Loads city summary statistics to analyze and plot relationships between
    information-theoretic quantities like divergence and growth.

    Args:
        data_path (str): Path to the .pkl file with summary statistics.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        return

    with open(data_path, 'rb') as f:
        all_stats = pickle.load(f)

    # 1. Filter for cities with population > 50k
    for city in all_stats.keys():
        print(city,all_stats[city].get('population', 0))
    filtered_stats = {city: data for city, data in all_stats.items() if abs(data.get('population', 0)) > 500000}
    print(len(filtered_stats))
    if not filtered_stats:
        print("No cities with population > 50,000 found in the data.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig.suptitle("Information-Theoretic Analysis of Economic Growth", fontsize=18)

    # --- Panel 1: Optimal Growth vs. Divergence ---
    city_divergence_variances = []
    all_p_series = []
    
    # Find the global min/max years for consistent coloring
    all_years = [year for stats in filtered_stats.values() for year in stats.get('optimal_gamma', pd.Series()).index]
    min_year, max_year = (min(all_years), max(all_years)) if all_years else (2015, 2023)
    
    norm = plt.Normalize(vmin=min_year, vmax=max_year)
    cmap = plt.get_cmap('viridis')

    for city_name, stats in filtered_stats.items():
        optimal_gamma = stats.get('optimal_gamma')
        mean_divergence = stats.get('mean_divergence')
        std_divergence = stats.get('std_divergence')
        p_series = stats.get('p_t_series')
        x_series = stats.get('mean_x_trajectory')

        if optimal_gamma is not None and not optimal_gamma.empty and \
           mean_divergence is not None and not mean_divergence.empty:
            
            # Align data on the year index
            df = pd.concat([optimal_gamma.rename('gamma'), mean_divergence.rename('divergence')], axis=1).dropna()
            
            if not df.empty:
                # Create line segments
                points = df[['divergence', 'gamma']].to_numpy().reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create a LineCollection with colors mapped to years
                lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.3, linewidths=1)
                lc.set_array(df.index[:-1])
                ax1.add_collection(lc)
                
                # Plot scatter points on top, also colored by year
                ax1.scatter(df['divergence'], df['gamma'], c=df.index, cmap=cmap, norm=norm, s=20, zorder=10,alpha=0.5)

                # Add city label to the point with the highest y-value
                max_y_row = df.loc[df['gamma'].idxmax()]
                short_name = city_name.split(',')[0]
                ax1.text(max_y_row['divergence'] - 0.001, max_y_row['gamma'], short_name, 
                         fontsize=7, ha='right', va='center', alpha=0.8)

        if p_series is not None and not p_series.empty and \
           x_series is not None and not x_series.empty:
            
            # Align p and x data
            df_px = pd.concat([p_series.rename('p'), x_series.rename('x')], axis=1).dropna()

            if not df_px.empty:
                # Create line segments for p vs x plot
                points_px = df_px[['x', 'p']].to_numpy().reshape(-1, 1, 2)
                segments_px = np.concatenate([points_px[:-1], points_px[1:]], axis=1)

                # Create LineCollection colored by year
                lc_px = LineCollection(segments_px, cmap=cmap, norm=norm, alpha=0.3, linewidths=1)
                lc_px.set_array(df_px.index[:-1])
                ax2.add_collection(lc_px)

                # Plot scatter points on top
                ax2.scatter(df_px['x'], df_px['p'], c=df_px.index, cmap=cmap, norm=norm, s=20, zorder=10,alpha=0.5)
                
                # Add city label to the point with the highest x-value
                max_x_row = df_px.loc[df_px['x'].idxmax()]
                short_name = city_name.split(',')[0]
                ax2.text(max_x_row['x'] - 0.01, max_x_row['p'], short_name,
                         fontsize=7, ha='right', va='center', alpha=0.8)


        if std_divergence is not None and not std_divergence.empty:
            # Calculate the mean of the variance for this city
            mean_variance = (std_divergence**2).mean()
            city_divergence_variances.append(mean_variance)

        if p_series is not None and not p_series.empty:
            all_p_series.append(p_series)

    ax1.set_title("Optimal Growth vs. Information Divergence", fontsize=14)
    ax1.set_xlabel("Mean Divergence ($D_{KL}(p || x)$)", fontsize=12)
    ax1.set_ylabel("City-wide Information ($I$)", fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a colorbar for the year mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])


    # --- Panel 2: p vs. x ---
    ax2.set_title("Environment vs. Agent Belief (p vs. x)", fontsize=14)
    ax2.set_xlabel("Mean Agent Belief ($x_t$)", fontsize=12)
    ax2.set_ylabel("Environmental Predictability ($p_t$)", fontsize=12)
    ax2.set_xlim(.45, .6)
    ax2.set_ylim(.4, .9)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()
    cbar_px = fig.colorbar(sm, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_px.set_label('Year')


    # --- Panel 3: Histogram of Divergence Variance ---
    if city_divergence_variances:
        ax3.hist(city_divergence_variances, bins=12, color='purple', alpha=0.7, density=True)
    
    # Calculate the cross-city divergence in p
    if all_p_series:
        df_p = pd.concat(all_p_series, axis=1)
        # Variance of p across cities for each year, then take the mean
        mean_p_variance = df_p.var(axis=1).mean()
        ax3.axvline(mean_p_variance, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean Cross-City p Variance: {mean_p_variance:.4f}')
        ax3.legend()

    ax3.set_title("Distribution of Within-City Divergence Variance", fontsize=14)
    ax3.set_xlabel("Mean Variance of Divergence per City", fontsize=12)
    ax3.set_ylabel("Density", fontsize=12)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Save and Show Plot ---
    output_filename = "information_stats_analysis.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f"Saved analysis plot to {output_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_information_stats() 