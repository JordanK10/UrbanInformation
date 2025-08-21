"""
Core functions for processing and analyzing CBSA-level data.

This module orchestrates the main analysis pipeline for a single CBSA:
1.  Calculates the city-wide environmental predictability signal (p_t).
2.  Runs the Bayesian state-space model in parallel for all block groups.
3.  Aggregates and summarizes the raw model outputs into meaningful statistics.
"""

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from urbaninformation.analysis.utils import (
    calculate_divergence,
    transform_x_to_gamma_dynamic,
)
from urbaninformation.modeling.ssm import fit_ssm_dynamic_p_model


def calculate_cbsa_level_p_t(cbsa_df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    """
    Calculates the longitudinal, population-weighted p_t series for a CBSA.

    Origin: Adapted from `visualization_scripts/dynamic_p_utils.py`.

    Args:
        cbsa_df: DataFrame for a single CBSA with columns
                 ['block_group_fips', 'year', 'mean_income', 'population'].

    Returns:
        A tuple containing:
        - p_t_series (np.ndarray): The calculated predictability series.
        - sorted_years (List[int]): The years corresponding to the p_t series.
    """
    yearly_wins: Dict[int, float] = {}
    yearly_totals: Dict[int, float] = {}

    for _, group in cbsa_df.groupby("block_group_fips"):
        group = group.sort_values("year")
        if len(group) > 1 and not (group["mean_income"] <= 0).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                log_income = np.log(group["mean_income"].values)
            y_values = np.diff(log_income)

            if np.all(np.isfinite(y_values)):
                years = group["year"].values[1:]
                pop_end = group["population"].values[1:]
                for i, year in enumerate(years):
                    weight = (
                        float(pop_end[i])
                        if np.isfinite(pop_end[i]) and pop_end[i] > 0
                        else 0.0
                    )
                    yearly_totals.setdefault(year, 0.0)
                    yearly_wins.setdefault(year, 0.0)
                    yearly_totals[year] += weight
                    if y_values[i] > 0:
                        yearly_wins[year] += weight

    valid_years = [yr for yr in sorted(yearly_totals.keys()) if yearly_totals[yr] > 0]
    if not valid_years:
        return np.array([]), []

    p_t_series = np.array(
        [yearly_wins[year] / yearly_totals[year] for year in valid_years]
    )
    # Clip to prevent logit(0) or logit(1) issues
    p_t_series_clipped = np.clip(p_t_series, 1e-5, 1 - 1e-5)

    return p_t_series_clipped, valid_years


def _process_block_group(
    bg_fips: str,
    y_values: np.ndarray,
    year: List[int],
    population_ts: List[float],
    p_t_series: np.ndarray,
    p_t_years: List[int],
    method: str,
) -> Dict[str, Any]:
    """
    Helper function to fit the SSM model for a single block group.

    Origin: `run_cbsa_analysis_dynamic_p.py`.
    """
    warnings.simplefilter("ignore", UserWarning)
    try:
        bg_p_t_years = year[1:]
        p_t_map = pd.Series(p_t_series, index=p_t_years)
        aligned_p_t = p_t_map.reindex(bg_p_t_years).values

        if np.isnan(aligned_p_t).any():
            return {"status": "failure", "error": "Could not align p_t series."}

        idata, loss = fit_ssm_dynamic_p_model(
            y_values, aligned_p_t, method=method, vi_steps=30000
        )
        x_mean_trajectory = idata.posterior["x"].mean(axis=(0, 1)).to_numpy()

        return {
            "block_group_fips": bg_fips,
            "x_mean_trajectory": x_mean_trajectory,
            "p_hat_frequentist": aligned_p_t,
            "population_ts": population_ts,
            "loss": loss,
            "year": year,
            "status": "success",
        }
    except Exception as e:
        # It can be useful to know which block group failed during a long run
        # print(f"ERROR processing {bg_fips}: {e}")
        return {"block_group_fips": bg_fips, "status": "failure", "error": str(e)}


def run_cbsa_analysis(
    cbsa_name: str, cbsa_df: pd.DataFrame, method: str = "vi"
) -> List[Dict[str, Any]]:
    """
    Runs the full modeling pipeline for all valid block groups within a single CBSA.

    Args:
        cbsa_name: The name of the CBSA being processed.
        cbsa_df: The DataFrame containing data for this CBSA.
        method: The inference method ('vi' or 'mcmc').

    Returns:
        A list of result dictionaries, one for each successfully processed block group.
    """
    p_t_series, p_t_years = calculate_cbsa_level_p_t(cbsa_df)
    if p_t_series.size == 0:
        print(f"No valid data to calculate p_t for {cbsa_name}. Skipping.")
        return []

    tasks = []
    for bg_fips, group in cbsa_df.groupby("block_group_fips"):
        group = group.sort_values("year")
        if len(group) < 2 or (group["mean_income"] <= 0).any():
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            log_income = np.log(group["mean_income"].values)
        y_values = np.diff(log_income)

        if not np.all(np.isfinite(y_values)):
            continue

        tasks.append(
            delayed(_process_block_group)(
                bg_fips,
                y_values,
                group["year"].tolist(),
                group["population"].tolist(),
                p_t_series,
                p_t_years,
                method,
            )
        )

    if not tasks:
        print(f"No valid block groups for {cbsa_name}. Skipping.")
        return []

    results = Parallel(n_jobs=-1)(
        tqdm(tasks, desc=f"Analyzing {cbsa_name}", total=len(tasks), leave=False)
    )

    successful_results = [
        res for res in results if res and res.get("status") == "success"
    ]
    return successful_results


def summarize_cbsa_results(
    cbsa_results: List[Dict[str, Any]], cbsa_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Aggregates raw block group results into summary statistics for a CBSA.

    Origin: Logic from `visualize_results.py`. This function separates the
    calculation from the plotting.

    Args:
        cbsa_results: The list of raw results from `run_cbsa_analysis`.
        cbsa_df: The original DataFrame for the CBSA, used for population data.

    Returns:
        A dictionary containing summary statistics (as pandas Series) for the CBSA.
    """
    if not cbsa_results:
        return {}

    p_t_series, p_t_years = calculate_cbsa_level_p_t(cbsa_df)
    p_t_map = pd.Series(p_t_series, index=p_t_years)
    l = 2
    optimal_gamma = (
        np.log(l)
        + p_t_series * np.log(p_t_series)
        + (1 - p_t_series) * np.log(1 - p_t_series)
    )
    losses = [res["loss"] for res in cbsa_results if res.get("loss") is not None]

    # Process all trajectories to extract mean x, gamma, and divergence
    all_trajectories = []
    for res in cbsa_results:
        bg_years = res["year"][1:]
        aligned_p_t = p_t_map.reindex(bg_years).values
        if not np.isnan(aligned_p_t).any():
            all_trajectories.append(
                {
                    "years": bg_years,
                    "x": res["x_mean_trajectory"],
                    "gamma": transform_x_to_gamma_dynamic(
                        res["x_mean_trajectory"], aligned_p_t
                    ),
                    "divergence": calculate_divergence(
                        aligned_p_t, res["x_mean_trajectory"]
                    ),
                }
            )

    # Combine all trajectories for each metric into a list of series for alignment
    x_series_list = [pd.Series(t["x"], index=t["years"]) for t in all_trajectories]
    gamma_series_list = [
        pd.Series(t["gamma"], index=t["years"]) for t in all_trajectories
    ]
    divergence_series_list = [
        pd.Series(t["divergence"], index=t["years"]) for t in all_trajectories
    ]

    # Align by year and compute mean
    mean_x = pd.concat(x_series_list, axis=1).mean(axis=1)
    std_x = pd.concat(x_series_list, axis=1).std(axis=1)
    mean_gamma = pd.concat(gamma_series_list, axis=1).mean(axis=1)
    std_gamma = pd.concat(gamma_series_list, axis=1).std(axis=1)
    mean_divergence = pd.concat(divergence_series_list, axis=1).mean(axis=1)
    std_divergence = pd.concat(divergence_series_list, axis=1).std(axis=1)

    total_population = (
        cbsa_df.groupby("year")["population"].sum().iloc[-1] if not cbsa_df.empty else 0
    )

    return {
        "p_t_series": pd.Series(p_t_series, index=p_t_years),
        "mean_x_trajectory": mean_x.sort_index(),
        "std_x_trajectory": std_x.sort_index(),
        "optimal_gamma": pd.Series(optimal_gamma, index=p_t_years),
        "mean_agent_gamma": mean_gamma.sort_index(),
        "std_agent_gamma": std_gamma.sort_index(),
        "mean_divergence": mean_divergence.sort_index(),
        "std_divergence": std_divergence.sort_index(),
        "mean_loss": np.mean(losses) if losses else None,
        "population": total_population,
    }
