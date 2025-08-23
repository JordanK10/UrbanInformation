import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def load_data():
    """Load data files."""
    with open('cbsa_acs_data.pkl', 'rb') as f:
        cbsa_acs_data = pickle.load(f)
    zip_data = pd.read_csv('uszips.csv')
    return cbsa_acs_data, zip_data

def load_centroids_2010():
    """Load block group centroids from 2010 Census."""
    dbf_file = 'nhgis0001_shapefile_cenpop2010_us_blck_grp_cenpop_2010/US_blck_grp_cenpop_2010.dbf'
    
    with open(dbf_file, 'rb') as f:
        header = f.read(32)
        num_records = int.from_bytes(header[4:8], 'little')
        header_length = int.from_bytes(header[8:10], 'little')
        
        # Read field descriptors
        num_fields = (header_length - 33) // 32
        fields = []
        for i in range(num_fields):
            field_desc = f.read(32)
            fields.append({
                'name': field_desc[:11].decode('ascii').strip('\x00'),
                'length': field_desc[16]
            })
        
        # Read records
        f.seek(header_length)
        centroids = {}
        
        for i in range(num_records):
            record = f.read(88)  # Fixed record length
            if len(record) < 88:
                break
            
            # Parse record
            offset = 1
            record_data = {}
            for field in fields:
                value = record[offset:offset + field['length']].decode('ascii').strip()
                record_data[field['name']] = value
                offset += field['length']
            
            # Extract coordinates
            geoid = record_data.get('GEOID', '')
            lat = record_data.get('LATITUDE', '')
            lon = record_data.get('LONGITUDE', '')
            
            if geoid and lat and lon:
                try:
                    centroids[geoid] = (float(lat), float(lon))
                except ValueError:
                    continue
    
    return centroids

def load_centroids_2020():
    """Load block group centroids from 2020 Census."""
    dbf_file = 'nhgis0001_shapefile_cenpop2020_us_blck_grp_cenpop_2020/US_blck_grp_cenpop_2020.dbf'
    
    with open(dbf_file, 'rb') as f:
        header = f.read(32)
        num_records = int.from_bytes(header[4:8], 'little')
        header_length = int.from_bytes(header[8:10], 'little')
        
        # Read field descriptors
        num_fields = (header_length - 33) // 32
        fields = []
        for i in range(num_fields):
            field_desc = f.read(32)
            fields.append({
                'name': field_desc[:11].decode('ascii').strip('\x00'),
                'length': field_desc[16]
            })
        
        # Read records
        f.seek(header_length)
        centroids = {}
        
        for i in range(num_records):
            record = f.read(88)  # Fixed record length
            if len(record) < 88:
                break
            
            # Parse record
            offset = 1
            record_data = {}
            for field in fields:
                value = record[offset:offset + field['length']].decode('ascii').strip()
                record_data[field['name']] = value
                offset += field['length']
            
            # Extract coordinates
            geoid = record_data.get('GEOID', '')
            lat = record_data.get('LATITUDE', '')
            lon = record_data.get('LONGITUDE', '')
            
            if geoid and lat and lon:
                try:
                    centroids[geoid] = (float(lat), float(lon))
                except ValueError:
                    continue
    
    return centroids

def find_closest_zip(block_group_fips, centroids, county_zips):
    """Find the closest ZIP code for a block group."""
    if block_group_fips not in centroids:
        return None
    
    bg_coords = np.array([centroids[block_group_fips]])
    zip_coords = county_zips[['lat', 'lng']].values
    
    if len(zip_coords) == 0:
        return None
    
    distances = cdist(bg_coords, zip_coords)[0]
    closest_idx = np.argmin(distances)
    return county_zips.iloc[closest_idx]['zip']

def match_blockgroups_to_zips(cbsa_acs_data, zip_data, centroids_2010, centroids_2020):
    """Match block groups to closest ZIP codes using both 2010 and 2020 centroids."""
    results = {}
    total_matched = 0
    total_blocks = 0
    matched_2010 = 0
    matched_2020 = 0
    
    for cbsa_name, cbsa_df in cbsa_acs_data.items():
        # Extract county FIPS and add ZIP codes
        cbsa_df = cbsa_df.copy()
        cbsa_df['county_fips'] = cbsa_df['block_group_fips'].str[:5]
        cbsa_df['closest_zip'] = None
        cbsa_df['centroid_source'] = None  # Track which centroid source was used
        
        # Process each county
        for county_fips, county_group in cbsa_df.groupby('county_fips'):
            county_zips = zip_data[zip_data['county_fips'] == int(county_fips)]
            if county_zips.empty:
                continue
            
            # Match each block group
            for idx, row in county_group.iterrows():
                block_group_fips = row['block_group_fips']
                
                # First try 2010 centroids
                closest_zip = find_closest_zip(block_group_fips, centroids_2010, county_zips)
                if closest_zip is not None:
                    cbsa_df.loc[idx, 'closest_zip'] = closest_zip
                    cbsa_df.loc[idx, 'centroid_source'] = '2010'
                    matched_2010 += 1
                else:
                    # Try 2020 centroids
                    closest_zip = find_closest_zip(block_group_fips, centroids_2020, county_zips)
                    if closest_zip is not None:
                        cbsa_df.loc[idx, 'closest_zip'] = closest_zip
                        cbsa_df.loc[idx, 'centroid_source'] = '2020'
                        matched_2020 += 1
        
        matched = cbsa_df['closest_zip'].notna().sum()
        total_matched += matched
        total_blocks += len(cbsa_df)
        results[cbsa_name] = cbsa_df
    
    print(f"Matched {total_matched:,}/{total_blocks:,} block groups")
    print(f"  - 2010 centroids: {matched_2010:,}")
    print(f"  - 2020 centroids: {matched_2020:,}")
    return results

def main():
    """Main function."""
    print("Loading data...")
    cbsa_acs_data, zip_data = load_data()
    
    print("Loading 2010 centroids...")
    centroids_2010 = load_centroids_2010()
    
    print("Loading 2020 centroids...")
    centroids_2020 = load_centroids_2020()
    
    print("Matching block groups to ZIP codes...")
    results = match_blockgroups_to_zips(cbsa_acs_data, zip_data, centroids_2010, centroids_2020)
    
    print("Saving results...")
    with open('blockgroups_with_zips_temporal.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Done!")

if __name__ == "__main__":
    main() 