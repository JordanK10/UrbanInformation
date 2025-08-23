import pickle
import pandas as pd

def investigate_unmatched():
    """Investigate why some block groups are unmatched."""
    
    # Load results
    print("Loading results...")
    with open('blockgroups_with_zips_temporal.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Find unmatched block groups
    total_unmatched = 0
    unmatched_fips = []
    
    for cbsa_name, cbsa_df in results.items():
        unmatched = cbsa_df[cbsa_df['closest_zip'].isna()]
        if len(unmatched) > 0:
            print(f"{cbsa_name}: {len(unmatched)} unmatched")
            unmatched_fips.extend(unmatched['block_group_fips'].tolist())
            total_unmatched += len(unmatched)
    
    print(f"\nTotal unmatched: {total_unmatched}")
    
    if unmatched_fips:
        print(f"\nSample unmatched FIPS: {unmatched_fips[:10]}")
        
        # Check if these FIPS exist in our centroid data
        print("\nChecking centroid availability...")
        
        # Load 2010 centroids
        from match_blockgroups_to_zips3 import load_centroids_2010, load_centroids_2020
        
        print("Loading 2010 centroids...")
        centroids_2010 = load_centroids_2010()
        
        print("Loading 2020 centroids...")
        centroids_2020 = load_centroids_2020()
        
        # Check each unmatched FIPS
        for fips in unmatched_fips[:5]:  # Check first 5
            in_2010 = fips in centroids_2010
            in_2020 = fips in centroids_2020
            print(f"FIPS {fips}: 2010={in_2010}, 2020={in_2020}")
            
            # Extract county FIPS
            county_fips = fips[:5]
            print(f"  County FIPS: {county_fips}")
            
            # Check if county has ZIP codes
            zip_data = pd.read_csv('uszips.csv')
            county_zips = zip_data[zip_data['county_fips'] == int(county_fips)]
            print(f"  ZIP codes in county: {len(county_zips)}")

if __name__ == "__main__":
    investigate_unmatched() 