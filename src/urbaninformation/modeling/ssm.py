import logging
import os
from contextlib import redirect_stdout

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan

logging.getLogger("pymc").setLevel(logging.ERROR)


def fit_ssm_dynamic_p_model(
    y,
    p_t_series,
    l=2,
    method="vi",
    vi_steps=15000,
    z0_mu=0.0,
    z0_sigma=1.0,
    k_prior_sigma=0.5,
    tau_sigma=0.1,
    obs_sigma_prior=0.05,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.98,
    random_seed=42,
):
    y = np.asarray(y, dtype=float)
    p_t_series = np.asarray(p_t_series, dtype=float)
    T = len(y)

    coords = {"time": np.arange(T)}

    with pm.Model(coords=coords) as model:
        # --- Priors ---
        # Agent's learning rate
        k = pm.HalfNormal("k", sigma=k_prior_sigma)
        # Agent's initial belief state (in logit-space)
        z0 = pm.Normal("z0", mu=z0_mu, sigma=z0_sigma)
        # Standard deviation of the process noise
        tau = pm.HalfNormal("tau", sigma=tau_sigma)
        # Standard deviation for the likelihood
        sigma = pm.HalfNormal("sigma", sigma=obs_sigma_prior)

        # Convert p_t to logit space to be the target for z_t
        p_t_logit = pt.log(p_t_series / (1 - p_t_series))

        # --- Recurrent Evolution of Belief State (z_t) ---
        def step(t, z_prev, p_target_logit, k_rate, noise):
            # Your proposed evolution equation
            z_mean = (p_target_logit * (1 / (k_rate * l)) + z_prev) / (
                1 + 1 / (k_rate * l)
            )
            # The new state is the mean + noise
            z_t = z_mean + noise[t]
            return z_t

        # This is noise for the *entire* time series, pre-supplied to scan
        process_noise = pm.Normal("process_noise", mu=0, sigma=tau, dims="time")

        z_full, _ = scan(
            fn=step,
            sequences=[pt.arange(T), p_t_logit],
            outputs_info=[z0],
            non_sequences=[k, process_noise],
            strict=True,
        )

        # --- Likelihood ---
        x = pm.Deterministic("x", pm.math.invlogit(z_full), dims="time")

        mu_correct = pm.Deterministic("mu_correct", pm.math.log(l * x), dims="time")
        mu_wrong = pm.Deterministic("mu_wrong", pm.math.log(l * (1 - x)), dims="time")

        comp_correct = pm.Normal.dist(mu=mu_correct, sigma=sigma)
        comp_wrong = pm.Normal.dist(mu=mu_wrong, sigma=sigma)

        # The weight `w` is now the dynamic p_t_series data
        pm.Mixture(
            "y_obs",
            w=pt.stack([p_t_series, 1 - p_t_series]).T,
            comp_dists=[comp_correct, comp_wrong],
            observed=y,
            dims="time",
        )

        # --- Sampling ---
        if method == "vi":
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
