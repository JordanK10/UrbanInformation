"""
Generates summary visualizations that compare and aggregate across all CBSAs.

This module provides functions to create high-level summary plots from the
fully processed `cbsa_summary_statistics.pkl` file. It includes:
- Overlay plots of all individual CBSA trajectories.
- Aggregated plots showing the mean, CI, and variance across all CBSAs.
- Information-theoretic analysis plots comparing growth, belief, and divergence.
"""

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def plot_all_cbsa_trajectories(all_stats: Dict[str, Any], output_dir: str):
    """
    Generates a plot comparing the trajectories of all CBSAs.

    Origin: This is the first figure from `visualization_scripts/plot_all_trajectories.py`.
    It creates "spaghetti plots" to show the variation across all cities.

    Args:
        all_stats: The dictionary of summary statistics for all CBSAs.
        output_dir: The directory where the plot will be saved.
    """
    if not all_stats:
        print("No data available to plot all CBSA trajectories.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.suptitle("Cross-CBSA Summary Statistics", fontsize=20)

    ax_p, ax_x, ax_gamma, ax_divergence = axes.flatten()

    for city_name, stats in all_stats.items():
        if not stats["p_t_series"].empty:
            stats["p_t_series"].plot(ax=ax_p, alpha=0.6, linewidth=1.5)
        if not stats["mean_x_trajectory"].empty:
            stats["mean_x_trajectory"].plot(ax=ax_x, alpha=0.6, linewidth=1.5)
        if not stats["optimal_gamma"].empty:
            stats["optimal_gamma"].plot(
                ax=ax_gamma, alpha=0.6, linestyle="--", linewidth=1.5
            )
        if not stats["mean_agent_gamma"].empty:
            stats["mean_agent_gamma"].plot(
                ax=ax_gamma, alpha=0.6, linestyle="-", linewidth=1.5
            )
        if "mean_divergence" in stats and not stats["mean_divergence"].empty:
            stats["mean_divergence"].plot(ax=ax_divergence, alpha=0.6, linewidth=1.5)

    # --- Formatting ---
    ax_p.set_title("Environmental Predictability ($p_t$)", fontsize=14)
    ax_p.set_xlabel("Year", fontsize=12)
    ax_p.set_ylabel("Predictability ($p_t$)", fontsize=12)
    ax_p.set_ylim(0, 1)

    ax_x.set_title("Mean Agent Belief ($x_t$)", fontsize=14)
    ax_x.set_xlabel("Year", fontsize=12)
    ax_x.set_ylabel("Belief ($x_t$)", fontsize=12)
    ax_x.set_ylim(0, 1)

    ax_gamma.set_title(r"Mean Growth Rates ($\gamma_t$)", fontsize=14)
    ax_gamma.set_xlabel("Year", fontsize=12)
    ax_gamma.set_ylabel("Growth Rate (nats/period)", fontsize=12)
    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="--", label="Optimal"),
        Line2D([0], [0], color="gray", linestyle="-", label="Agent"),
    ]
    ax_gamma.legend(handles=legend_elements, title="Growth Type")

    ax_divergence.set_title("Mean Information Divergence", fontsize=14)
    ax_divergence.set_xlabel("Year", fontsize=12)
    ax_divergence.set_ylabel("Divergence ($D_{KL}$)", fontsize=12)

    output_path = os.path.join(output_dir, "cross_cbsa_summary_plot.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary trajectory plot to {output_path}")


def plot_aggregated_cbsa_statistics(all_stats: Dict[str, Any], output_dir: str):
    """
    Generates plots of aggregated (mean, CI, variance) statistics across all CBSAs.

    Origin: This is the second figure from `visualization_scripts/plot_all_trajectories.py`.

    Args:
        all_stats: The dictionary of summary statistics for all CBSAs.
        output_dir: The directory where the plot will be saved.
    """
    if not all_stats:
        print("No data available to plot aggregated CBSA statistics.")
        return

    # Collect all series into lists and then align into DataFrames
    p_series_list = [
        s["p_t_series"] for s in all_stats.values() if not s["p_t_series"].empty
    ]
    x_series_list = [
        s["mean_x_trajectory"]
        for s in all_stats.values()
        if not s["mean_x_trajectory"].empty
    ]
    optimal_gamma_list = [
        s["optimal_gamma"] for s in all_stats.values() if not s["optimal_gamma"].empty
    ]
    agent_gamma_list = [
        s["mean_agent_gamma"]
        for s in all_stats.values()
        if not s["mean_agent_gamma"].empty
    ]
    divergence_list = [
        s["mean_divergence"]
        for s in all_stats.values()
        if "mean_divergence" in s and not s["mean_divergence"].empty
    ]
    losses = [s["mean_loss"] for s in all_stats.values() if s["mean_loss"] is not None]

    df_p = pd.concat(p_series_list, axis=1)
    df_x = pd.concat(x_series_list, axis=1)
    df_optimal_gamma = pd.concat(optimal_gamma_list, axis=1)
    df_agent_gamma = pd.concat(agent_gamma_list, axis=1)
    df_divergence = pd.concat(divergence_list, axis=1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
    fig.suptitle("Aggregated Cross-CBSA Statistics", fontsize=20)

    def plot_mean_ci_var(ax, df, title, ylabel, color):
        """Helper to plot mean, CI, and variance inset."""
        mean = df.mean(axis=1)
        ci_low = df.quantile(0.025, axis=1)
        ci_high = df.quantile(0.975, axis=1)
        variance = df.var(axis=1)

        mean.plot(ax=ax, color=color, linewidth=2.5, label="Mean")
        ax.fill_between(
            mean.index, ci_low, ci_high, color=color, alpha=0.2, label="95% CI"
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        # Inset for variance
        ax_inset = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
        variance.plot(ax=ax_inset, color="black", linewidth=2)
        ax_inset.set_title("Variance", fontsize=10)

    plot_mean_ci_var(
        axes[0, 0],
        df_p,
        "Mean Environmental Predictability ($p_t$)",
        "Predictability ($p_t$)",
        "blue",
    )
    axes[0, 0].set_ylim(0, 1)
    plot_mean_ci_var(
        axes[0, 1], df_x, "Mean Agent Belief ($x_t$)", "Belief ($x_t$)", "crimson"
    )
    axes[0, 1].set_ylim(0, 1)
    plot_mean_ci_var(
        axes[0, 2], df_optimal_gamma, "Mean Information ($I_t$)", "Growth Rate", "green"
    )
    plot_mean_ci_var(
        axes[1, 0],
        df_agent_gamma,
        r"Mean Agent Growth Rate ($\gamma_t$)",
        "Growth Rate",
        "purple",
    )
    plot_mean_ci_var(
        axes[1, 1],
        df_divergence,
        "Mean Information Divergence",
        "Divergence ($D_{KL}$)",
        "orange",
    )

    # Histogram of Losses
    ax_loss = axes[1, 2]
    if losses:
        ax_loss.hist(losses, bins=30, color="black", alpha=0.7, density=True)
        mean_loss = np.mean(losses)
        ax_loss.axvline(
            mean_loss,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_loss:.2f}",
        )
        ax_loss.legend()
    ax_loss.set_title("Distribution of Mean Model Loss", fontsize=14)
    ax_loss.set_xlabel("Mean Loss", fontsize=12)
    ax_loss.set_ylabel("Density", fontsize=12)

    output_path = os.path.join(output_dir, "cross_cbsa_aggregated_plot.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aggregated statistics plot to {output_path}")


def plot_information_theoretic_analysis(all_stats: Dict[str, Any], output_dir: str):
    """
    Creates plots analyzing relationships between divergence, growth, and belief.

    Origin: `visualization_scripts/information_stats.py`.

    Args:
        all_stats: The dictionary of summary statistics for all CBSAs.
        output_dir: The directory where the plot will be saved.
    """
    filtered_stats = {
        c: d for c, d in all_stats.items() if d.get("population", 0) > 500000
    }
    if not filtered_stats:
        print(
            "No cities with population > 500,000 found. Skipping info-theoretic plot."
        )
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    fig.suptitle("Information-Theoretic Analysis of Economic Growth", fontsize=18)

    all_years = [
        year
        for s in filtered_stats.values()
        for year in s.get("optimal_gamma", pd.Series()).index
    ]
    min_year, max_year = (min(all_years), max(all_years)) if all_years else (2015, 2023)
    norm = plt.Normalize(vmin=min_year, vmax=max_year)
    cmap = plt.get_cmap("viridis")

    for city_name, stats in filtered_stats.items():
        df_gamma_div = pd.concat(
            [stats.get("optimal_gamma"), stats.get("mean_divergence")], axis=1
        ).dropna()
        df_p_x = pd.concat(
            [stats.get("p_t_series"), stats.get("mean_x_trajectory")], axis=1
        ).dropna()

        # Plot Growth vs. Divergence
        if not df_gamma_div.empty:
            points = df_gamma_div.to_numpy().reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.3, linewidths=1)
            lc.set_array(df_gamma_div.index[:-1])
            ax1.add_collection(lc)
            ax1.scatter(
                df_gamma_div.iloc[:, 1],
                df_gamma_div.iloc[:, 0],
                c=df_gamma_div.index,
                cmap=cmap,
                norm=norm,
                s=20,
                zorder=10,
            )

        # Plot p vs. x
        if not df_p_x.empty:
            points_px = df_p_x.to_numpy().reshape(-1, 1, 2)
            segments_px = np.concatenate([points_px[:-1], points_px[1:]], axis=1)
            lc_px = LineCollection(
                segments_px, cmap=cmap, norm=norm, alpha=0.3, linewidths=1
            )
            lc_px.set_array(df_p_x.index[:-1])
            ax2.add_collection(lc_px)
            ax2.scatter(
                df_p_x.iloc[:, 1],
                df_p_x.iloc[:, 0],
                c=df_p_x.index,
                cmap=cmap,
                norm=norm,
                s=20,
                zorder=10,
            )

    ax1.set_title("Optimal Growth vs. Information Divergence", fontsize=14)
    ax1.set_xlabel("Mean Divergence ($D_{KL}(p || x)$)", fontsize=12)
    ax1.set_ylabel("City-wide Information ($I$)", fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, orientation="vertical", label="Year")

    ax2.set_title("Environment vs. Agent Belief (p vs. x)", fontsize=14)
    ax2.set_xlabel("Mean Agent Belief ($x_t$)", fontsize=12)
    ax2.set_ylabel("Environmental Predictability ($p_t$)", fontsize=12)
    ax2.set_xlim(0.45, 0.6)
    ax2.set_ylim(0.4, 0.9)
    fig.colorbar(sm, ax=ax2, orientation="vertical", label="Year")

    # Histogram of Divergence Variance
    city_div_variances = [
        (s["std_divergence"] ** 2).mean()
        for s in filtered_stats.values()
        if s.get("std_divergence") is not None
    ]
    if city_div_variances:
        ax3.hist(city_div_variances, bins=12, color="purple", alpha=0.7, density=True)

    df_p_all = pd.concat([s["p_t_series"] for s in filtered_stats.values()], axis=1)
    mean_p_variance = df_p_all.var(axis=1).mean()
    ax3.axvline(
        mean_p_variance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Cross-City p Variance: {mean_p_variance:.4f}",
    )
    ax3.legend()

    ax3.set_title("Distribution of Within-City Divergence Variance", fontsize=14)
    ax3.set_xlabel("Mean Variance of Divergence per City", fontsize=12)

    output_path = os.path.join(output_dir, "information_stats_analysis.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved information-theoretic analysis plot to {output_path}")
