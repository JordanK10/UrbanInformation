"""
Runs the analysis pipeline for a single, specified CBSA via command-line.

This script loads the main configuration, selects a target city, and uses the
urbaninformation library to process all of its block groups, saving the raw
results to a specified output directory.

Example Usage:
# Run with defaults (Chicago, output/scratch)
uv python scripts/run_single_cbsa.py

# Run for a different CBSA
uv python scripts/run_single_cbsa.py --cbsa-name "New York-Newark-Jersey City, NY-NJ"

# Specify a different output directory
uv python scripts/run_single_cbsa.py --output-dir "output/nyc_test"
"""

import argparse
import os
import pickle
import tomllib

from urbaninformation.analysis.cbsa_processing import run_cbsa_analysis
from urbaninformation.analysis.utils import sanitize_filename


def main(args):
    """Main function to execute the single-CBSA analysis."""
    print(f"--- Running Analysis for: {args.cbsa_name} ---")

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ACS data from: {config['paths']['acs_data']}")
    with open(config["paths"]["acs_data"], "rb") as f:
        all_cbsa_data = pickle.load(f)

    cbsa_df = all_cbsa_data.get(args.cbsa_name)
    if cbsa_df is None:
        raise KeyError(f"Target CBSA '{args.cbsa_name}' not found in the dataset.")

    raw_results = run_cbsa_analysis(
        cbsa_name=args.cbsa_name,
        cbsa_df=cbsa_df,
        method=config["analysis"]["model_method"],
    )

    sanitized_name = sanitize_filename(args.cbsa_name)
    output_path = os.path.join(args.output_dir, f"{sanitized_name}_raw_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(raw_results, f)

    print(f"\nAnalysis complete. Saved {len(raw_results)} results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis for a single CBSA.")
    parser.add_argument(
        "--cbsa-name",
        type=str,
        default="Chicago-Naperville-Elgin, IL-IN",
        help="The full name of the target CBSA to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/scratch",
        help="Directory to save the raw result pickle file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Path to the TOML configuration file.",
    )
    args = parser.parse_args()
    main(args)
