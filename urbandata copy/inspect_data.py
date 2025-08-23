import pickle
import pandas as pd

def inspect_pickle(file_path):
    """
    Loads a pickle file and inspects its contents.
    """
    print(f"Inspecting pickle file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded. Data type: {type(data)}")
        
        if isinstance(data, dict):
            print("Data is a dictionary. Inspecting keys and value types:")
            for key, value in data.items():
                print(f"  - Key: '{key}', Value Type: {type(value)}")
                if isinstance(value, pd.DataFrame):
                    print(f"    - DataFrame has {len(value)} rows and columns: {list(value.columns)}")
                elif isinstance(value, list) and value:
                    print(f"    - List contains {len(value)} items. First item type: {type(value[0])}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Path to the pickle file, adjusted to the new location
    path = 'data/data_retrieval/cbsa_acs_data.pkl'
    inspect_pickle(path) 