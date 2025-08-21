"""
Generates visualizations for a single Core-Based Statistical Area (CBSA).

This module takes the summarized statistics for one CBSA and produces a
comprehensive 2x2 plot showing the environmental signal, agent belief,
resulting growth rate, and model diagnostics.
"""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from urbaninformation.analysis.utils import sanitize_filename


def plot_cbsa_summary(
    cbsa_name: str,
    summary_stats: Dict[str, Any],
    cbsa_results: List[Dict[str, Any]],
    output_dir: str,
    plot_individual_trajectories: bool = False,
):
    """
    Generates and saves a 2x2 summary plot for a single CBSA.

    Origin: This function is the refactored plotting component of the original
    `visualize_results.py` script.

    Args:
        cbsa_name: The name of the CBSA for titles.
        summary_stats: A dictionary of aggregated statistics from
                       `analysis.cbsa_processing.summarize_cbsa_results`.
        cbsa_results: The raw list of results for the CBSA, used here only for
                      plotting the loss distribution.
        output_dir: The directory where the plot will be saved.
        plot_individual_trajectories: If True, plots the faint individual
                                      block group trajectories behind the mean.
    """
    if not summary_stats:
        print(f"No summary statistics available for {cbsa_name}. Skipping plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(f"Comprehensive Analysis for {cbsa_name}", fontsize=20)

    # --- Panel 1: Environmental Predictability (p_t) ---
    ax1 = axes[0, 0]
    p_t_series = summary_stats.get("p_t_series")
    if p_t_series is not None and not p_t_series.empty:
        ax1.plot(
            p_t_series.index, p_t_series.values, marker="o", linestyle="-", color="blue"
        )
    ax1.set_title("Environmental Predictability ($p_t$)", fontsize=14)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("City-Wide Predictability ($p_t$)", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True)

    # --- Panel 2: Inferred Agent Belief (x_t) ---
    ax2 = axes[0, 1]
    mean_x = summary_stats.get("mean_x_trajectory")
    std_x = summary_stats.get("std_x_trajectory")

    if plot_individual_trajectories:
        for res in cbsa_results:
            ax2.plot(
                res["year"][1:], res["x_mean_trajectory"], color="crimson", alpha=0.05
            )

    if mean_x is not None and not mean_x.empty:
        ci_low_x = mean_x - 1.96 * std_x
        ci_high_x = mean_x + 1.96 * std_x
        ax2.plot(
            mean_x.index,
            mean_x.values,
            color="crimson",
            linewidth=2.5,
            label="Mean Belief",
        )
        ax2.fill_between(
            mean_x.index,
            ci_low_x,
            ci_high_x,
            color="crimson",
            alpha=0.2,
            label="95% CI",
        )

    ax2.set_title("Inferred Agent Belief ($x_t$)", fontsize=14)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Agent Belief ($x_t$)", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()

    # --- Panel 3: Resulting Economic Growth (gamma_t) ---
    ax3 = axes[1, 0]
    mean_gamma = summary_stats.get("mean_agent_gamma")
    std_gamma = summary_stats.get("std_agent_gamma")
    optimal_gamma = summary_stats.get("optimal_gamma")

    if mean_gamma is not None and not mean_gamma.empty:
        ci_low_g = mean_gamma - 1.96 * std_gamma
        ci_high_g = mean_gamma + 1.96 * std_gamma
        ax3.plot(
            mean_gamma.index,
            mean_gamma.values,
            color="purple",
            linewidth=2.5,
            label="Mean Agent Growth",
        )
        ax3.fill_between(
            mean_gamma.index,
            ci_low_g,
            ci_high_g,
            color="purple",
            alpha=0.2,
            label="95% CI",
        )

    if optimal_gamma is not None and not optimal_gamma.empty:
        ax3.plot(
            optimal_gamma.index,
            optimal_gamma.values,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Optimal Growth ($x_t=p_t$)",
        )

    ax3.set_title(r"Resulting Growth Rate ($\gamma_t$)", fontsize=14)
    ax3.set_xlabel("Year", fontsize=12)
    ax3.set_ylabel("Expected Growth Rate (nats/period)", fontsize=12)
    ax3.legend()

    # --- Panel 4: Model Loss Distribution ---
    ax4 = axes[1, 1]
    losses = [res["loss"] for res in cbsa_results if res.get("loss") is not None]
    if losses:
        ax4.hist(losses, bins=50, color="black", density=True, alpha=0.7)
        mean_loss = np.mean(losses)
        ax4.axvline(
            mean_loss, color="red", linestyle="--", label=f"Mean: {mean_loss:.2f}"
        )
    ax4.set_title("Model Loss Distribution", fontsize=14)
    ax4.set_xlabel("Final Average Loss (VI)", fontsize=12)
    ax4.set_ylabel("Density", fontsize=12)
    ax4.legend()

    # --- Save Figure ---
    sanitized_name = sanitize_filename(cbsa_name)
    output_filename = os.path.join(output_dir, f"{sanitized_name}_summary_plot.pdf")
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.close(fig)
