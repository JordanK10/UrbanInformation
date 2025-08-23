import pickle
import pandas as pd
import numpy as np

def analyze_population_coverage():
    """Analyze population coverage before and after ZIP code matching."""
    
    print("Loading data...")
    
    # Load original ACS data
    with open('cbsa_acs_data.pkl', 'rb') as f:
        acs_data = pickle.load(f)
    
    # Load matched data
    with open('blockgroups_with_zips_temporal.pkl', 'rb') as f:
        matched_data = pickle.load(f)
    
    print("Analyzing population coverage...")
    
    # Debug: Check data structure
    sample_cbsa = list(acs_data.values())[0]
    print(f"Sample CBSA data structure:")
    print(f"  Columns: {sample_cbsa.columns.tolist()}")
    print(f"  Population data type: {sample_cbsa['population'].dtype}")
    print(f"  Sample population values: {sample_cbsa['population'].head().tolist()}")
    print(f"  Population range: {sample_cbsa['population'].min():.2f} to {sample_cbsa['population'].max():.2f}")
    
    # Calculate population statistics for original data
    original_stats = {}
    total_original_pop = 0
    total_original_blocks = 0
    
    for cbsa_name, cbsa_df in acs_data.items():
        # Sum population across all years for each block group
        # Use absolute values to handle any negative numbers
        block_group_pop = cbsa_df.groupby('block_group_fips')['population'].sum()
        
        original_stats[cbsa_name] = {
            'total_population': abs(block_group_pop.sum()),
            'block_groups': len(block_group_pop),
            'avg_pop_per_bg': abs(block_group_pop.mean())
        }
        
        total_original_pop += original_stats[cbsa_name]['total_population']
        total_original_blocks += original_stats[cbsa_name]['block_groups']
    
    # Calculate population statistics for matched data
    matched_stats = {}
    total_matched_pop = 0
    total_matched_blocks = 0
    
    for cbsa_name, cbsa_df in matched_data.items():
        # Only count block groups that have ZIP codes
        matched_bg = cbsa_df[cbsa_df['closest_zip'].notna()]
        
        if len(matched_bg) > 0:
            # Sum population across all years for matched block groups
            block_group_pop = matched_bg.groupby('block_group_fips')['population'].sum()
            
            matched_stats[cbsa_name] = {
                'total_population': abs(block_group_pop.sum()),
                'block_groups': len(block_group_pop),
                'avg_pop_per_bg': abs(block_group_pop.mean()),
                'match_rate': len(matched_bg) / len(cbsa_df)
            }
            
            total_matched_pop += matched_stats[cbsa_name]['total_population']
            total_matched_blocks += matched_stats[cbsa_name]['block_groups']
    
    # Print summary statistics
    print(f"\n=== POPULATION COVERAGE ANALYSIS ===")
    print(f"Original data:")
    print(f"  Total population: {total_original_pop:,.0f}")
    print(f"  Total block groups: {total_original_blocks:,}")
    print(f"  Average population per block group: {total_original_pop/total_original_blocks:,.0f}")
    
    print(f"\nMatched data:")
    print(f"  Total population: {total_matched_pop:,.0f}")
    print(f"  Total block groups: {total_matched_blocks:,}")
    print(f"  Average population per block group: {total_matched_pop/total_matched_blocks:,.0f}")
    
    print(f"\nCoverage summary:")
    print(f"  Population coverage: {total_matched_pop/total_original_pop*100:.2f}%")
    print(f"  Block group coverage: {total_matched_blocks/total_original_blocks*100:.2f}%")
    
    # Find CBSAs with biggest population drops
    print(f"\n=== CBSAs WITH BIGGEST POPULATION DROPS ===")
    population_drops = []
    
    for cbsa_name in original_stats.keys():
        if cbsa_name in matched_stats:
            original_pop = original_stats[cbsa_name]['total_population']
            matched_pop = matched_stats[cbsa_name]['total_population']
            drop_pct = (original_pop - matched_pop) / original_pop * 100
            
            population_drops.append({
                'cbsa': cbsa_name,
                'original_pop': original_pop,
                'matched_pop': matched_pop,
                'drop_pct': drop_pct,
                'match_rate': matched_stats[cbsa_name]['match_rate']
            })
    
    # Sort by population drop percentage
    population_drops.sort(key=lambda x: x['drop_pct'], reverse=True)
    
    for i, drop in enumerate(population_drops[:10]):
        print(f"{i+1:2d}. {drop['cbsa']:<40} {drop['drop_pct']:6.2f}% drop")
        print(f"     Original: {drop['original_pop']:>12,.0f}, Matched: {drop['matched_pop']:>12,.0f}")
    
    # Check if any CBSAs have 0% population coverage
    zero_coverage = [drop for drop in population_drops if drop['drop_pct'] == 100]
    if zero_coverage:
        print(f"\nCBSAs with 0% population coverage:")
        for drop in zero_coverage:
            print(f"  {drop['cbsa']}")
    
    return original_stats, matched_stats, population_drops

if __name__ == "__main__":
    analyze_population_coverage() 