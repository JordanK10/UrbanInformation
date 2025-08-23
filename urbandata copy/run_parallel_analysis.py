import os
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from process_cbsa import process_one_cbsa

def main_parallel(method='vi'):
    """
    Main function to run the dynamic-p analysis in parallel for each CBSA.
    """
    acs_data_path = "data/cbsa_acs_data.pkl"
    if not os.path.exists(acs_data_path):
        print(f"Error: ACS data file not found at '{acs_data_path}'")
        print("Please run acs_data_retrieval.py first.")
        return

    with open(acs_data_path, 'rb') as f:
        cbsa_acs_data = pickle.load(f)

    output_dir = "analysis_output_dynamic_p"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Prepare tasks for each CBSA
    tasks = []
    for cbsa_name, cbsa_df in cbsa_acs_data.items():
        tasks.append(delayed(process_one_cbsa)(cbsa_name, cbsa_df, output_dir, method))
        
    print(f"Prepared {len(tasks)} CBSA tasks. Starting parallel processing...")

    # Run CBSA processing in parallel
    # Note: We are using n_jobs=-2 to leave one core free for system processes.
    # The inner parallelization (for block groups) will use all available cores on its node.
    results = Parallel(n_jobs=-2)(tqdm(tasks, desc="Processing all CBSAs"))

    print("\n--- Parallel Analysis Complete ---")
    for result in results:
        print(result)

if __name__ == "__main__":
    main_parallel(method='vi') 