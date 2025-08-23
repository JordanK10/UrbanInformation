import os
import pickle
import pandas as pd
from census import Census

# --- Configuration ---
CENSUS_API_KEY = "35d314060d56f894db2f7621b0e5e5f7eca9af27"
if not CENSUS_API_KEY:
    raise ValueError("CENSUS_API_KEY environment variable not set. Please get a key from https://api.census.gov/data/key_signup.html")

YEARS = list(range(2014, 2024))
ACS_DATASET = 'acs5'

# Income bins and their midpoints for mean income calculation
INCOME_BINS = {
    'B19001_002E': 5000,
    'B19001_003E': 12500,
    'B19001_004E': 17500,
    'B19001_005E': 22500,
    'B19001_006E': 27500,
    'B19001_007E': 32500,
    'B19001_008E': 37500,
    'B19001_009E': 42500,
    'B19001_010E': 47500,
    'B19001_011E': 55000,
    'B19001_012E': 67500,
    'B19001_013E': 87500,
    'B19001_014E': 112500,
    'B19001_015E': 137500,
    'B19001_016E': 175000,
    'B19001_017E': 300000,  # Midpoint of $200,000 - $400,000 (user assumption)
}

# Variables to retrieve from ACS
ACS_VARIABLES = [
    'NAME',
    'B19001_001E',  # Total households
    'B25010_001E',  # Average household size
] + list(INCOME_BINS.keys())


def load_msa_data(filename="msa_fips_data.pkl"):
    """Loads the CBSA to FIPS codes mapping from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_acs_data(census_client, year, state_fips, county_fips):
    """Fetches ACS data for all block groups in a given county and year."""
    try:
        return census_client.acs5.get(
            ACS_VARIABLES,
            {'for': 'block group:*', 'in': f'state:{state_fips} county:{county_fips}'},
            year=year
        )
    except Exception as e:
        print(f"Could not retrieve data for state {state_fips}, county {county_fips}, year {year}: {e}")
        return None

def calculate_mean_income(row):
    """Calculates the mean household income from income distribution data."""
    total_households = row['B19001_001E']
    if total_households == 0:
        return 0
    
    weighted_income_sum = sum(row[bin_var] * midpoint for bin_var, midpoint in INCOME_BINS.items())
    return weighted_income_sum / total_households

def process_cbsa_data(cbsa_name, fips_codes, census_client):
    """Processes ACS data for a single CBSA over all specified years."""
    all_cbsa_data = []
    print(f"Processing CBSA: {cbsa_name}")
    
    # Track filtering statistics
    total_observations = 0
    filtered_observations = 0

    for year in YEARS:
        print(f"  Year: {year}")
        for state_fips, county_fips in fips_codes:
            raw_data = get_acs_data(census_client, year, state_fips, county_fips)
            if not raw_data:
                continue

            df = pd.DataFrame(raw_data)
            
            # Convert numeric columns to numeric types, coercing errors
            for col in ACS_VARIABLES:
                if col != 'NAME':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()

            if df.empty:
                continue

            # Apply data quality filters to remove problematic observations
            initial_count = len(df)
            total_observations += initial_count
            print(f"    Before filtering: {initial_count} block groups")
            
            # Filter out negative or zero household counts
            df = df[df['B19001_001E'] > 0]
            household_filtered = initial_count - len(df)
            
            # Filter out negative or zero household size
            df = df[df['B25010_001E'] > 0]
            size_filtered = initial_count - household_filtered - len(df)
            
            # Filter out extremely large household sizes (likely data errors)
            df = df[df['B25010_001E'] <= 50]  # Reasonable upper bound
            large_size_filtered = initial_count - household_filtered - size_filtered - len(df)
            
            # Filter out negative income values (though this should be redundant)
            df = df[df['B19001_001E'] > 0]  # Household count should be non-negative
            
            final_count = len(df)
            filtered_observations += (initial_count - final_count)
            
            print(f"    After filtering: {final_count} block groups")
            print(f"    Filtered out: {initial_count - final_count} observations")
            if household_filtered > 0:
                print(f"      - Invalid household count: {household_filtered}")
            if size_filtered > 0:
                print(f"      - Invalid household size: {size_filtered}")
            if large_size_filtered > 0:
                print(f"      - Extremely large household size: {large_size_filtered}")
            
            if df.empty:
                print(f"    No valid data after filtering for {cbsa_name}, {year}")
                continue

            df['mean_income'] = df.apply(calculate_mean_income, axis=1)
            df['population'] = df['B19001_001E'] * df['B25010_001E']
            df['year'] = year
            df['block_group_fips'] = df['state'] + df['county'] + df['tract'] + df['block group']

            all_cbsa_data.append(df[['year', 'block_group_fips', 'mean_income', 'population']])
    
    if not all_cbsa_data:
        return None

    print(f"  Total observations processed: {total_observations:,}")
    print(f"  Total observations filtered: {filtered_observations:,}")
    print(f"  Filtering rate: {filtered_observations/total_observations*100:.2f}%")

    return pd.concat(all_cbsa_data, ignore_index=True)


def main():
    """Main function to orchestrate the data retrieval and processing."""
    
    # Check if the data already exists to avoid re-downloading
    if os.path.exists('cbsa_acs_data.pkl'):
        print("ACS data file 'cbsa_acs_data.pkl' already exists. Skipping download.")
        print("If you want to re-download the data, please delete the file and run this script again.")
        return

    c = Census(CENSUS_API_KEY)
    msa_fips_data = load_msa_data()
    
    all_cbsa_acs_data = {}
    for cbsa, fips in msa_fips_data.items():
        cbsa_df = process_cbsa_data(cbsa, fips, c)
        if cbsa_df is not None:
            all_cbsa_acs_data[cbsa] = cbsa_df

    with open('cbsa_acs_data.pkl', 'wb') as f:
        pickle.dump(all_cbsa_acs_data, f)

    print("\nSuccessfully retrieved and processed ACS data for all CBSAs.")
    print("Data saved to cbsa_acs_data.pkl")


if __name__ == "__main__":
    main() 