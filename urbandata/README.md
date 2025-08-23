# Urban Data Analysis Project

This project contains the core data analysis scripts and tools for analyzing urban income dynamics and socioeconomic mobility using ACS (American Community Survey) data.

## Project Structure

- **`data/`**: Contains all data files, data retrieval scripts, and data processing utilities
- **`analysis_output_*/`**: Output directories from various hierarchical analyses
- **`output_dynamic_p*/`**: Output directories from dynamic probability analyses
- **`visualization_scripts/`**: Scripts for creating plots and visualizations
- **`wins_growth_correlation/`**: Analysis of correlation between growth rates and "wins"
- **`temporal_autocorrelation_plots/`**: Analysis of temporal persistence in income dynamics

## Key Scripts

- **`run_cbsa_analysis_dynamic_p_hierarchical.py`**: Main analysis script for hierarchical CBSA analysis
- **`wins_growth_correlation.py`**: Analyzes correlation between income growth and winning streaks
- **`temporal_autocorrelation.py`**: Measures temporal persistence in income dynamics
- **`ssm_model.py`**: State Space Model implementations for belief inference

## Data Sources

- ACS income data by block group and CBSA
- ZIP code to block group mappings
- Census shapefiles for geographic analysis

## Usage

This project focuses on the application of the information inference methodology to real-world urban data, while the companion project `UrbanInformation` contains the validation and methodological development components. 