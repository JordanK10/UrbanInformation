import pickle
from match_blockgroups_to_zips3 import load_centroids_2010, load_centroids_2020

def check_centroids():
    """Check centroid availability for specific block groups."""
    
    # Load ACS data
    with open('cbsa_acs_data.pkl', 'rb') as f:
        acs_data = pickle.load(f)
    
    # Check Tucson, AZ (which has unmatched block groups)
    tucson = acs_data['Tucson, AZ']
    print("Tucson, AZ block groups:")
    print(tucson['block_group_fips'].head(10).tolist())
    
    # Load centroids
    print("\nLoading centroids...")
    centroids_2010 = load_centroids_2010()
    centroids_2020 = load_centroids_2020()
    
    print(f"2010 centroids loaded: {len(centroids_2010):,}")
    print(f"2020 centroids loaded: {len(centroids_2020):,}")
    
    # Check specific unmatched FIPS
    unmatched_fips = ['040190041211', '040190041251', '040190041252']
    
    print("\nChecking unmatched FIPS:")
    for fips in unmatched_fips:
        in_2010 = fips in centroids_2010
        in_2020 = fips in centroids_2020
        print(f"{fips}: 2010={in_2010}, 2020={in_2020}")
        
        # Check if similar FIPS exist
        similar_2010 = [k for k in centroids_2010.keys() if k.startswith('040190041')]
        similar_2020 = [k for k in centroids_2020.keys() if k.startswith('040190041')]
        print(f"  Similar FIPS in 2010: {len(similar_2010)}")
        print(f"  Similar FIPS in 2020: {len(similar_2020)}")
        
        if similar_2010:
            print(f"  Sample 2010: {similar_2010[:3]}")
        if similar_2020:
            print(f"  Sample 2020: {similar_2020[:3]}")

if __name__ == "__main__":
    check_centroids() 