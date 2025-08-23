import csv
import collections
import pickle

def process_msa_data():
    """
    Reads city and MSA data from CSV files, processes it, and returns a dictionary
    mapping CBSA titles to a list of their state and county FIPS codes.
    """
    cities = set()
    with open('cities.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                # Clean up city name and add to set
                cities.add(row[0].strip())

    msa_data = collections.defaultdict(list)
    with open('list1_2023.csv', 'r') as f:
        # Skip the first two header lines
        next(f)
        next(f)
        reader = csv.DictReader(f)
        for row in reader:
            cbsa_title = row.get('CBSA Title')
            if cbsa_title:
                for city_state in cities:
                    # Match if the CBSA title starts with the city name (e.g., "Austin")
                    city_name = city_state.split(',')[0].strip()
                    if cbsa_title.startswith(city_name):
                        state_fips = row.get('FIPS State Code')
                        county_fips = row.get('FIPS County Code')
                        if state_fips and county_fips:
                            msa_data[cbsa_title].append((state_fips, county_fips))
                        # Found a match for this row, move to the next one.
                        break
    
    return dict(msa_data)

if __name__ == "__main__":
    msa_fips_data = process_msa_data()
    # Save the resulting dictionary to a .pkl file
    with open('msa_fips_data.pkl', 'wb') as f:
        pickle.dump(msa_fips_data, f)
    
    print("Successfully saved data to msa_fips_data.pkl") 