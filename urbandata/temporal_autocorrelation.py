import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_autocorrelation(series):
    """Calculates the lag-1 autocorrelation of a pandas Series."""
    if len(series) < 2:
        return np.nan
    return series.autocorr(lag=1)

def pivot_city_data(city_df):
    """Pivots the city DataFrame from long format to wide format."""
    try:
        wide_df = city_df.pivot_table(index='block_group_fips', 
                                    columns='year', 
                                    values='mean_income')
        return wide_df
    except Exception as e:
        print(f"Could not pivot data. Error: {e}")
        return None

def analyze_city_autocorrelation(city_name, city_df_wide, output_dir):
    """
    Analyzes and plots the temporal autocorrelation of income gains and wins
    for all block groups in a single city.
    
    Returns:
        dict: Summary statistics for this city, or None if analysis failed
    """
    gains_autocorrs = []
    wins_autocorrs = []

    for fips, row in city_df_wide.iterrows():
        income_series = row.dropna().astype(float)
        
        if len(income_series) >= 3: # Need at least 3 points for 2 growth rates
            # Calculate income gains (log differences)
            growth_rates = np.log(income_series).diff().dropna()
            
            if len(growth_rates) >= 2:
                # Calculate wins (binary series)
                wins = (growth_rates > 0).astype(int)

                # Calculate autocorrelations
                gains_autocorrs.append(calculate_autocorrelation(growth_rates))
                wins_autocorrs.append(calculate_autocorrelation(wins))

    # Filter out NaN values from lists
    gains_autocorrs = [ac for ac in gains_autocorrs if not np.isnan(ac)]
    wins_autocorrs = [ac for ac in wins_autocorrs if not np.isnan(ac)]

    if not gains_autocorrs or not wins_autocorrs:
        print(f"City {city_name}: Not enough valid data for autocorrelation analysis.")
        return None

    # Print summary stats to terminal
    mean_gains_ac = np.mean(gains_autocorrs)
    mean_wins_ac = np.mean(wins_autocorrs)
    print(f"City: {city_name}")
    print(f"  - Mean Income Gains Autocorrelation: {mean_gains_ac:.3f}")
    print(f"  - Mean Wins Autocorrelation: {mean_wins_ac:.3f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Temporal Autocorrelation for {city_name}', fontsize=16)

    # Plot for Income Gains Autocorrelation
    axes[0].hist(gains_autocorrs, bins=20, alpha=0.7, color='navy')
    axes[0].axvline(mean_gains_ac, color='red', linestyle='--', label=f'Mean = {mean_gains_ac:.2f}')
    axes[0].set_title('Distribution of Income Gain Autocorrelation')
    axes[0].set_xlabel('Lag-1 Autocorrelation')
    axes[0].set_ylabel('Number of Block Groups')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot for Wins Autocorrelation
    axes[1].hist(wins_autocorrs, bins=20, alpha=0.7, color='teal')
    axes[1].axvline(mean_wins_ac, color='red', linestyle='--', label=f'Mean = {mean_wins_ac:.2f}')
    axes[1].set_title('Distribution of "Wins" Autocorrelation')
    axes[1].set_xlabel('Lag-1 Autocorrelation')
    axes[1].set_ylabel('Number of Block Groups')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the combined plot
    safe_city_name = city_name.replace('/', '_').replace(' ', '_').split(',')[0]
    plot_path = os.path.join(output_dir, f'{safe_city_name}_autocorrelation.png')
    plt.savefig(plot_path)
    plt.close()

    # Return summary statistics for this city
    return {
        'city_name': city_name,
        'mean_gains_autocorr': mean_gains_ac,
        'mean_wins_autocorr': mean_wins_ac,
        'n_block_groups': len(gains_autocorrs)
    }

def create_summary_plots(city_summary_stats, output_dir):
    """
    Creates summary plots showing the relationship between mean autocorrelations
    across all cities.
    """
    # Convert to DataFrame for easier plotting
    summary_df = pd.DataFrame(city_summary_stats)
    
    # Create the summary figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-City Summary: Temporal Autocorrelation Analysis', fontsize=16)
    
    # 1. Scatter plot: Mean Income Gains vs Mean Wins Autocorrelation
    axes[0, 0].scatter(summary_df['mean_wins_autocorr'], 
                       summary_df['mean_gains_autocorr'], 
                       alpha=0.7, s=50)
    
    # Add city labels for points
    for _, row in summary_df.iterrows():
        city_short = row['city_name'].split(',')[0][:15]  # Truncate long names
        axes[0, 0].annotate(city_short, 
                           (row['mean_wins_autocorr'], row['mean_gains_autocorr']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add correlation line
    correlation = np.corrcoef(summary_df['mean_wins_autocorr'], 
                             summary_df['mean_gains_autocorr'])[0, 1]
    axes[0, 0].set_title(f'Mean Wins vs Mean Income Gains Autocorrelation\nCorrelation: {correlation:.3f}')
    axes[0, 0].set_xlabel('Mean Wins Autocorrelation')
    axes[0, 0].set_ylabel('Mean Income Gains Autocorrelation')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram of Mean Income Gains Autocorrelation
    axes[0, 1].hist(summary_df['mean_gains_autocorr'], bins=15, alpha=0.7, color='navy', edgecolor='black')
    axes[0, 1].axvline(summary_df['mean_gains_autocorr'].mean(), color='red', linestyle='--', 
                       label=f'Overall Mean: {summary_df["mean_gains_autocorr"].mean():.3f}')
    axes[0, 1].set_title('Distribution of Mean Income Gains Autocorrelation Across Cities')
    axes[0, 1].set_xlabel('Mean Income Gains Autocorrelation')
    axes[0, 1].set_ylabel('Number of Cities')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of Mean Wins Autocorrelation
    axes[1, 0].hist(summary_df['mean_wins_autocorr'], bins=15, alpha=0.7, color='teal', edgecolor='black')
    axes[1, 0].axvline(summary_df['mean_wins_autocorr'].mean(), color='red', linestyle='--',
                       label=f'Overall Mean: {summary_df["mean_wins_autocorr"].mean():.3f}')
    axes[1, 0].set_title('Distribution of Mean Wins Autocorrelation Across Cities')
    axes[1, 0].set_xlabel('Mean Wins Autocorrelation')
    axes[1, 0].set_ylabel('Number of Cities')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    axes[1, 1].axis('off')  # Turn off the axis for the table
    
    # Calculate summary statistics
    stats_text = f"""
    Summary Statistics Across {len(summary_df)} Cities:
    
    Income Gains Autocorrelation:
    - Mean: {summary_df['mean_gains_autocorr'].mean():.3f}
    - Std:  {summary_df['mean_gains_autocorr'].std():.3f}
    - Min:  {summary_df['mean_gains_autocorr'].min():.3f}
    - Max:  {summary_df['mean_gains_autocorr'].max():.3f}
    
    Wins Autocorrelation:
    - Mean: {summary_df['mean_wins_autocorr'].mean():.3f}
    - Std:  {summary_df['mean_wins_autocorr'].std():.3f}
    - Min:  {summary_df['mean_wins_autocorr'].min():.3f}
    - Max:  {summary_df['mean_wins_autocorr'].max():.3f}
    
    Cross-Correlation: {correlation:.3f}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the summary plot
    summary_plot_path = os.path.join(output_dir, 'cross_city_summary_autocorrelation.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the summary data as CSV
    summary_csv_path = os.path.join(output_dir, 'city_autocorrelation_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"Summary plots and data saved to {output_dir}")
    print(f"Cross-city correlation: {correlation:.3f}")

def main():
    """Main function to load data and run autocorrelation analysis for all cities."""
    data_path = 'data/data_retrieval/cbsa_acs_data.pkl'
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            full_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    output_dir = 'temporal_autocorrelation_plots'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to '{output_dir}' directory.")

    # Store summary statistics for all cities
    city_summary_stats = []

    if isinstance(full_data, dict):
        city_iterator = tqdm(full_data.items(), desc="Processing cities")
        for city_name, city_df_long in city_iterator:
            city_df_wide = pivot_city_data(city_df_long)
            if city_df_wide is not None:
                # Get the summary stats for this city
                city_stats = analyze_city_autocorrelation(city_name, city_df_wide, output_dir)
                if city_stats is not None:
                    city_summary_stats.append(city_stats)
    else:
        print("Data is not in the expected dictionary format.")
        return

    # Create summary plots across all cities
    if city_summary_stats:
        create_summary_plots(city_summary_stats, output_dir)
        print(f"\nSummary analysis complete. Check '{output_dir}' for all plots.")

if __name__ == '__main__':
    main() 