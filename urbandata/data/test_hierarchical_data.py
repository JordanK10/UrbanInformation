import pickle
import pandas as pd
import numpy as np

def test_hierarchical_data():
    """Test the hierarchical data structure and p_t calculations."""
    
    print("Testing hierarchical data structure...")
    
    # Load data
    print("Loading data...")
    with open('cbsa_acs_data.pkl', 'rb') as f:
        cbsa_acs_data = pickle.load(f)
    
    with open('blockgroups_with_zips_temporal.pkl', 'rb') as f:
        zip_matched_data = pickle.load(f)
    
    print(f"ACS data CBSAs: {len(cbsa_acs_data)}")
    print(f"ZIP matched CBSAs: {len(zip_matched_data)}")
    
    # Test with a small CBSA first
    test_cbsa = 'Austin, MN'  # Small CBSA for testing
    
    if test_cbsa in cbsa_acs_data and test_cbsa in zip_matched_data:
        print(f"\nTesting with {test_cbsa}...")
        
        acs_df = cbsa_acs_data[test_cbsa]
        zip_df = zip_matched_data[test_cbsa]
        
        print(f"  ACS data shape: {acs_df.shape}")
        print(f"  ZIP data shape: {zip_df.shape}")
        print(f"  ZIP data columns: {zip_df.columns.tolist()}")
        
        # Check ZIP matching
        valid_df = zip_df[zip_df['closest_zip'].notna()].copy()
        print(f"  Valid ZIP-matched block groups: {len(valid_df)}")
        
        if 'county_fips' not in valid_df.columns:
            valid_df['county_fips'] = valid_df['block_group_fips'].str[:5]
        
        # Test p_t calculations
        print(f"\n  Testing p_t calculations...")
        
        # CBSA level
        print(f"    CBSA level:")
        yearly_wins = {}
        yearly_totals = {}
        
        for bg_fips, group in valid_df.groupby('block_group_fips'):
            group = group.sort_values('year')
            if len(group) > 1 and not (group['mean_income'] <= 0).any():
                log_income = np.log(group['mean_income'].values)
                y_values = np.diff(log_income)
                if np.all(np.isfinite(y_values)):
                    years = group['year'].values[1:]
                    pop_end = group['population'].values[1:]
                    for i, year in enumerate(years):
                        weight = float(pop_end[i]) if np.isfinite(pop_end[i]) and pop_end[i] > 0 else 0.0
                        if year not in yearly_totals:
                            yearly_totals[year] = 0.0
                            yearly_wins[year] = 0.0
                        yearly_totals[year] += weight
                        if y_values[i] > 0:
                            yearly_wins[year] += weight
        
        sorted_years = [yr for yr in sorted(yearly_totals.keys()) if yearly_totals[yr] > 0]
        if sorted_years:
            p_t_series = np.array([yearly_wins[year] / yearly_totals[year] for year in sorted_years])
            p_t_series = np.clip(p_t_series, 1e-5, 1 - 1e-5)
            print(f"      Years: {sorted_years}")
            print(f"      p_t values: {p_t_series}")
        else:
            print(f"      No valid years found")
        
        # County level
        print(f"    County level:")
        counties = valid_df['county_fips'].unique()
        print(f"      Counties: {counties}")
        
        for county in counties:
            county_group = valid_df[valid_df['county_fips'] == county]
            print(f"      County {county}: {len(county_group)} block groups")
        
        # ZIP level
        print(f"    ZIP level:")
        zip_codes = valid_df['closest_zip'].unique()
        print(f"      ZIP codes: {len(zip_codes)}")
        print(f"      Sample ZIPs: {zip_codes[:5]}")
        
        # Check data quality
        print(f"\n  Data quality check:")
        print(f"    Negative population values: {(valid_df['population'] < 0).sum()}")
        print(f"    Zero population values: {(valid_df['population'] == 0).sum()}")
        print(f"    Negative income values: {(valid_df['mean_income'] < 0).sum()}")
        print(f"    Population range: {valid_df['population'].min():.2f} to {valid_df['population'].max():.2f}")
        print(f"    Income range: {valid_df['mean_income'].min():.2f} to {valid_df['mean_income'].max():.2f}")
        
    else:
        print(f"Test CBSA {test_cbsa} not found in both datasets")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_hierarchical_data() 