"""
Core utility functions for the urbaninformation project.

This module contains self-contained, reusable functions for tasks such as
filename sanitization, data manipulation (padding, transformations), and
statistical calculations (divergence, trajectory stats). These functions form
the foundational toolkit for other modules in the analysis pipeline.
"""

import numpy as np


def sanitize_filename(filename: str) -> str:
    """
    Removes or replaces characters from a string to make it a valid filename.

    Origin: Found in `run_cbsa_analysis_dynamic_p.py` and `dynamic_p_utils.py`.
    """
    return "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_")).rstrip()


def pad_to_equal_length(seqs: list, align: str = "right") -> np.ndarray:
    """
    Pads a list of 1D arrays/lists with NaNs to equal length.

    Origin: `visualization_scripts/dynamic_p_utils.py`

    Args:
        seqs: List of 1D arrays/lists to pad.
        align: "right" keeps the tail aligned (useful for time series),
               "left" keeps the head aligned.

    Returns:
        A 2D numpy array with padded sequences.
    """
    if not seqs:
        return np.empty((0, 0))
    max_len = max(len(s) for s in seqs)
    padded = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=float)
        length = len(s)
        if length == 0:
            continue
        if align == "left":
            padded[i, :length] = s
        else:  # align right
            padded[i, -length:] = s
    return padded


def transform_x_to_gamma_dynamic(
    x_traj: np.ndarray, p_t: np.ndarray, l: int = 2
) -> np.ndarray:
    """
    Transforms an agent belief trajectory (x_t) into a growth rate (gamma)
    trajectory using a dynamic environmental predictability series (p_t).

    Origin: `visualization_scripts/dynamic_p_utils.py`

    Args:
        x_traj: Agent belief trajectory (numpy array).
        p_t: Dynamic environmental predictability series (numpy array).
        l: Model parameter (number of choices, default=2).

    Returns:
        The corresponding growth rate trajectory (numpy array).
    """
    x_traj = np.asarray(x_traj)
    p_t = np.asarray(p_t)

    if len(p_t) != len(x_traj):
        raise ValueError(
            f"Shape mismatch: p_t has length {len(p_t)} but x_traj has length {len(x_traj)}"
        )

    x_clipped = np.clip(x_traj, 1e-9, 1 - 1e-9)
    return np.log(l) + p_t * np.log(x_clipped) + (1 - p_t) * np.log(1 - x_clipped)


def calculate_divergence(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Computes the KL divergence between two Bernoulli distributions parameterized
    by p and x.

    Formula: D_KL(P || Q) = p*log(p/x) + (1-p)*log((1-p)/(1-x))

    Origin: `visualization_scripts/dynamic_p_utils.py`

    Args:
        p: The "true" probability distribution parameter (numpy array).
        x: The "approximating" probability distribution parameter (numpy array).

    Returns:
        The KL divergence (numpy array).
    """
    p = np.asarray(p)
    x = np.asarray(x)

    p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
    x_clipped = np.clip(x, 1e-9, 1 - 1e-9)

    term1 = p_clipped * np.log(p_clipped / x_clipped)
    term2 = (1 - p_clipped) * np.log((1 - p_clipped) / (1 - x_clipped))

    return term1 + term2


def compute_trajectory_statistics(trajectories: list, align: str = "right") -> tuple:
    """
    Computes mean, standard deviation, and confidence intervals for a list of
    trajectories of potentially different lengths.

    Origin: `visualization_scripts/dynamic_p_utils.py`

    Args:
        trajectories: A list of 1D numpy arrays or lists.
        align: Alignment for padding ("right" or "left").

    Returns:
        A tuple containing (mean_traj, std_traj, ci_low, ci_high) as numpy arrays.
    """
    if not trajectories:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    padded = pad_to_equal_length(trajectories, align=align)
    with np.warnings.catch_warnings():
        # Ignore warnings from slices with all NaNs
        np.warnings.filterwarnings("ignore", r"Mean of empty slice")
        np.warnings.filterwarnings("ignore", r"invalid value encountered in percentile")

        mean_traj = np.nanmean(padded, axis=0)
        std_traj = np.nanstd(padded, axis=0)
        ci_low = np.nanpercentile(padded, 2.5, axis=0)
        ci_high = np.nanpercentile(padded, 97.5, axis=0)

    return mean_traj, std_traj, ci_low, ci_high
