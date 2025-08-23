import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

def analyze_city_data(city_cbsa, city_df, output_dir):
    """
    Analyzes the block group data for a single city.

    For each block group, it calculates:
    1. The mean log-income growth rate over the time series.
    2. The total number of years with positive income growth ("wins").

    It then creates a scatter plot of these two metrics, calculates their
    correlation, and saves the plot.

    Args:
        city_cbsa (str): The CBSA code for the city.
        city_df (pd.DataFrame): DataFrame containing the data for the city.
        output_dir (str): The directory to save the output plot.
    """
    block_group_stats = []

    # Identify the year columns
    year_cols = sorted([col for col in city_df.columns if col.startswith('B19013')])
    
    if len(year_cols) < 2:
        # Not enough data for growth calculation
        return

    for _, row in city_df.iterrows():
        income_series = row[year_cols].dropna()
        
        if len(income_series) >= 2:
            # Calculate log growth rates
            log_income = np.log(income_series.astype(float))
            growth_rates = log_income.diff().dropna()
            
            if not growth_rates.empty:
                avg_growth = growth_rates.mean()
                num_wins = (growth_rates > 0).sum()
                block_group_stats.append({
                    'avg_growth': avg_growth,
                    'num_wins': num_wins
                })

    if not block_group_stats:
        print(f"City {city_cbsa}: No block groups with sufficient data.")
        return

    stats_df = pd.DataFrame(block_group_stats)

    # Calculate correlation
    if len(stats_df) >= 2 and stats_df['avg_growth'].nunique() > 1 and stats_df['num_wins'].nunique() > 1:
        correlation, p_value = pearsonr(stats_df['avg_growth'], stats_df['num_wins'])
        print(f"City {city_cbsa}: Correlation = {correlation:.3f} (p-value = {p_value:.3f})")
    else:
        correlation = np.nan
        print(f"City {city_cbsa}: Not enough data to calculate correlation.")


    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(stats_df['num_wins'], stats_df['avg_growth'], alpha=0.6)
    
    # Add a regression line
    if not np.isnan(correlation):
        m, b = np.polyfit(stats_df['num_wins'], stats_df['avg_growth'], 1)
        plt.plot(stats_df['num_wins'], m * stats_df['num_wins'] + b, color='red', linestyle='--')

    plt.title(f'City (CBSA {city_cbsa}): Average Growth vs. Number of Wins\nCorrelation: {correlation:.3f}')
    plt.xlabel('Number of Time Steps with Positive Growth (Wins)')
    plt.ylabel('Average Log-Income Growth Rate')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'cbsa_{city_cbsa}_corr.png')
    plt.savefig(plot_path)
    plt.close()

def pivot_city_data(city_df):
    """
    Pivots the city DataFrame from long format to wide format.

    Args:
        city_df (pd.DataFrame): DataFrame in long format with columns
                                ['year', 'block_group_fips', 'mean_income'].

    Returns:
        pd.DataFrame: DataFrame in wide format where each row is a block group
                      and columns are mean incomes for each year.
    """
    try:
        # Use pivot_table to handle potential duplicate entries robustly
        wide_df = city_df.pivot_table(index='block_group_fips', 
                                    columns='year', 
                                    values='mean_income')
        
        # Rename columns to be more descriptive for the analysis function
        wide_df.columns = [f"B19013_{year}" for year in wide_df.columns]
        return wide_df.reset_index()
    except Exception as e:
        print(f"Could not pivot data. Error: {e}")
        return None

def main():
    """
    Main function to load data, iterate through cities, and run the analysis.
    """
    # Load the data
    data_path = 'data/data_retrieval/cbsa_acs_data.pkl'
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            full_data = pickle.load(f)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file is in the 'data' directory.")
        return

    # Create output directory
    output_dir = 'wins_growth_correlation'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to '{output_dir}' directory.")

    # Data is a dictionary of DataFrames, one for each city
    if isinstance(full_data, dict):
        print(f"Data is a dictionary with {len(full_data)} cities.")
        
        city_iterator = tqdm(full_data.items(), desc="Processing cities")
        
        for city_name, city_df_long in city_iterator:
            # Sanitize the city name to be used as a filename
            # Replace slashes and other problematic characters with underscores
            safe_city_name = city_name.replace('/', '_').replace(' ', '_').split(',')[0]

            # Pivot the data from long to wide format
            city_df_wide = pivot_city_data(city_df_long)
            
            if city_df_wide is not None:
                analyze_city_data(safe_city_name, city_df_wide, output_dir)
            
    else:
        print("Data is not in the expected dictionary format.")
        print(f"Data type is: {type(full_data)}")
        print("Please check the structure of 'cbsa_acs_data.pkl'.")


if __name__ == '__main__':
    main() 