"""
Summarizes and plots the results for a single CBSA via command-line.

This script loads raw results and uses the urbaninformation library to create
the final summary visualization, configured by a TOML file.

Example Usage:
# Run with defaults (Chicago, reading from/writing to output/scratch)
uv python scripts/plot_single_cbsa.py

# Plot a different CBSA from a different results directory
uv python scripts/plot_single_cbsa.py \
    --cbsa-name "New York-Newark-Jersey City, NY-NJ" \
    --results-dir "output/nyc_test"
"""

import argparse
import os
import pickle
import tomllib

from urbaninformation.analysis.cbsa_processing import summarize_cbsa_results
from urbaninformation.analysis.utils import sanitize_filename
from urbaninformation.viz.single_cbsa_plots import plot_cbsa_summary


def main(args):
    """Main function to execute the single-CBSA plotting."""
    print(f"--- Plotting Results for: {args.cbsa_name} ---")

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    sanitized_name = sanitize_filename(args.cbsa_name)
    results_path = os.path.join(args.results_dir, f"{sanitized_name}_raw_results.pkl")

    with open(results_path, "rb") as f:
        raw_results = pickle.load(f)

    with open(config["paths"]["acs_data"], "rb") as f:
        all_cbsa_data = pickle.load(f)
    cbsa_df = all_cbsa_data[args.cbsa_name]

    summary_stats = summarize_cbsa_results(raw_results, cbsa_df)

    plot_cbsa_summary(
        cbsa_name=args.cbsa_name,
        summary_stats=summary_stats,
        cbsa_results=raw_results,
        output_dir=args.results_dir,  # Save plot in the same dir as the results
        plot_individual_trajectories=config["plotting"]["plot_individual_cbsa"],
    )

    plot_path = os.path.join(args.results_dir, f"{sanitized_name}_summary_plot.pdf")
    print(f"Successfully created plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results for a single CBSA.")
    parser.add_argument(
        "--cbsa-name",
        type=str,
        default="Chicago-Naperville-Elgin, IL-IN",
        help="The full name of the target CBSA to plot.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="output/scratch",
        help="Directory where raw results are located and the plot will be saved.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Path to the TOML configuration file.",
    )
    args = parser.parse_args()
    main(args)
