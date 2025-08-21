"""
Defines and fits the Bayesian state-space model for inferring agent learning.

This module contains the core statistical model of the project. The function
`fit_ssm_dynamic_p_model` uses PyMC to implement a state-space model where an
agent's latent belief state (x_t) evolves based on a measured, dynamic
environmental signal (p_t).
"""

import logging
import os
from contextlib import redirect_stdout
from typing import Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from arviz import InferenceData
from pytensor.scan import scan

# Configure PyMC logging to be less verbose
logging.getLogger("pymc").setLevel(logging.ERROR)


def fit_ssm_dynamic_p_model(
    y: np.ndarray,
    p_t_series: np.ndarray,
    l: int = 2,
    method: str = "vi",
    vi_steps: int = 15000,
    z0_mu: float = 0.0,
    z0_sigma: float = 1.0,
    k_prior_sigma: float = 0.5,
    tau_sigma: float = 0.1,
    obs_sigma_prior: float = 0.05,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.98,
    random_seed: int = 42,
) -> Tuple[InferenceData, Optional[float]]:
    """
    Fits a Bayesian state-space model to infer learning from economic growth data.

    This model treats an agent's belief (x_t) as a latent state that evolves
    towards the environment's predictability (p_t). The observed data (y) are
    log-income growth rates, modeled as a mixture of two normal distributions
    weighted by the dynamic p_t.

    Args:
        y: Time series of observed log-income growth rates.
        p_t_series: Time series of the measured environmental predictability.
        l: Number of choices available to the agent (default=2).
        method: Inference method, 'vi' for variational inference or 'mcmc' for MCMC sampling.
        vi_steps: Number of steps for variational inference.
        z0_mu: Mean of the prior for the initial belief state (logit-space).
        z0_sigma: Std dev of the prior for the initial belief state (logit-space).
        k_prior_sigma: Std dev of the HalfNormal prior for the learning rate `k`.
        tau_sigma: Std dev of the HalfNormal prior for the process noise `tau`.
        obs_sigma_prior: Std dev of the HalfNormal prior for the observation noise `sigma`.
        draws: Number of posterior samples to draw.
        tune: Number of tuning steps for MCMC.
        chains: Number of chains for MCMC.
        target_accept: Target acceptance rate for MCMC.
        random_seed: Seed for reproducibility.

    Returns:
        A tuple containing:
        - idata (arviz.InferenceData): The posterior samples and model data.
        - loss (float or None): The final loss value from VI, or None for MCMC.
    """
    y = np.asarray(y, dtype=float)
    p_t_series = np.asarray(p_t_series, dtype=float)
    T = len(y)

    coords = {"time": np.arange(T)}

    with pm.Model(coords=coords) as model:
        # --- Priors ---
        k = pm.HalfNormal("k", sigma=k_prior_sigma)
        z0 = pm.Normal("z0", mu=z0_mu, sigma=z0_sigma)
        tau = pm.HalfNormal("tau", sigma=tau_sigma)
        sigma = pm.HalfNormal("sigma", sigma=obs_sigma_prior)

        # Target for belief evolution is the logit-transformed environmental signal
        p_t_logit = pt.log(p_t_series / (1 - p_t_series))

        # --- Recurrent Evolution of Belief State (z_t) ---
        def step(p_target_logit_t, z_prev, k_rate, noise_t):
            """Defines the state transition for a single time step."""
            z_mean = (p_target_logit_t * (1 / (k_rate * l)) + z_prev) / (
                1 + 1 / (k_rate * l)
            )
            z_t = z_mean + noise_t
            return z_t

        process_noise = pm.Normal("process_noise", mu=0, sigma=tau, dims="time")

        z_full, _ = scan(
            fn=step,
            sequences=[p_t_logit, process_noise],
            outputs_info=[z0],
            non_sequences=[k],
            strict=True,
        )

        # --- Likelihood ---
        x = pm.Deterministic("x", pm.math.invlogit(z_full), dims="time")

        mu_correct = pm.Deterministic("mu_correct", pm.math.log(l * x), dims="time")
        mu_wrong = pm.Deterministic("mu_wrong", pm.math.log(l * (1 - x)), dims="time")

        comp_correct = pm.Normal.dist(mu=mu_correct, sigma=sigma)
        comp_wrong = pm.Normal.dist(mu=mu_wrong, sigma=sigma)

        # Mixture weights are the dynamic p_t data
        weights = pt.stack([p_t_series, 1 - p_t_series]).T
        pm.Mixture(
            "y_obs",
            w=weights,
            comp_dists=[comp_correct, comp_wrong],
            observed=y,
            dims="time",
        )

        # --- Sampling ---
        if method == "vi":
            # Suppress verbose output from pm.fit
            with open(os.devnull, "w") as f, redirect_stdout(f):
                approx = pm.fit(n=vi_steps, random_seed=random_seed, progressbar=False)
            idata = approx.sample(draws=draws, random_seed=random_seed)
            loss = approx.hist[-1]
        else:  # MCMC
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=False,
            )
            loss = None

    return idata, loss
