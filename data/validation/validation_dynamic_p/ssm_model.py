import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
import numpy as np
import os
from contextlib import redirect_stdout

import logging
logging.getLogger("pymc").setLevel(logging.ERROR)

def logit(p):
    return np.log(p) - np.log(1 - p)

def invlogit(x):
    return 1 / (1 + np.exp(-x))

def fit_ssm_random_walk(y_values, l_t_series, init_x_traj, n_samples=1000):
    """
    Fits a State-Space Model where the latent belief x follows a Gaussian Random Walk.

    Args:
        y_values (np.ndarray): Time series of observed log-income growth.
        l_t_series (np.ndarray): Time series of the number of outcomes, l.
        init_x_traj (np.ndarray): An initial guess for the x trajectory, used for initialization.
        n_samples (int): Number of samples for the VI fit.

    Returns:
        tuple: (arviz.InferenceData, float) containing the fitted model and loss.
    """
    with pm.Model() as model:
        # --- Priors ---
        # Volatility of the belief evolution. A larger sigma allows faster belief changes.
        # Increased from 0.1 to 0.3 to better capture agents with strong learning patterns.
        sigma = pm.HalfNormal("sigma", 0.3)
        
        # --- Latent State: Belief (x) ---
        # The agent's belief is modeled as a random walk in logit-space to keep it in (0,1).
        # We initialize it with a Normal distribution centered on our smart guess.
        z_init_dist = pm.Normal.dist(mu=logit(init_x_traj[0]), sigma=0.1)
        z = pm.GaussianRandomWalk("z", sigma=sigma, init_dist=z_init_dist, shape=len(y_values))
        
        # Transform the latent state z back to the probability space [0,1] for x
        x = pm.Deterministic("x", pm.math.invlogit(z))

        # --- Observation Model ---
        # The expected log-income growth (y) depends on the belief x.
        # We need to handle the two cases: wins (y > 0) and losses (y < 0).
        mu_win = pt.log(l_t_series * x)
        mu_loss = pt.log(l_t_series * (1 - x))

        # We use pt.switch to select the correct formula based on the sign of the observed y.
        # This is a robust way to handle the discontinuity.
        # Note: We can't use y_values directly in the switch condition inside the model.
        # Instead, we pass the sign of y_values as a constant.
        y_signs = np.sign(y_values)
        mu = pt.switch(pt.gt(y_signs, 0), mu_win, mu_loss)
        
        # Observation noise - how much the actual income deviates from the theoretical expectation.
        obs_sigma = pm.HalfNormal("obs_sigma", 0.5)

        # Likelihood of observing the income data given our model
        y_obs = pm.Normal("y_obs", mu=mu, sigma=obs_sigma, observed=y_values)

        # --- Inference ---
        approx = pm.fit(n=n_samples, method='advi', obj_optimizer=pm.adagrad(learning_rate=0.1))
        idata = approx.sample(1000)
        loss = -approx.hist[-1]

    return idata, loss

def fit_ssm_dynamic_p_model(y, p_t_series, l=2, method='vi', vi_steps=15000,
                            z0_mu=0.0, z0_sigma=1.0, k_prior_sigma=0.5, 
                            tau_sigma=0.1, obs_sigma_prior=0.05, 
                            draws=1000, tune=1000, chains=4, 
                            target_accept=0.98, random_seed=42):
    
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
            z_mean = (p_target_logit * (1 / (k_rate * l)) + z_prev) / (1 + 1 / (k_rate * l))
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
            strict=True
        )
        
        # --- Likelihood ---
        x = pm.Deterministic("x", pm.math.invlogit(z_full), dims="time")
        
        mu_correct = pm.Deterministic("mu_correct", pm.math.log(l * x), dims="time")
        mu_wrong = pm.Deterministic("mu_wrong", pm.math.log(l * (1 - x)), dims="time")
        
        comp_correct = pm.Normal.dist(mu=mu_correct, sigma=sigma)
        comp_wrong = pm.Normal.dist(mu=mu_wrong, sigma=sigma)
        
        # The weight `w` is now the dynamic p_t_series data
        pm.Mixture("y_obs", w=pt.stack([p_t_series, 1 - p_t_series]).T, 
                   comp_dists=[comp_correct, comp_wrong], observed=y, dims="time")
        
        # --- Sampling ---
        if method == 'vi':
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                approx = pm.fit(n=vi_steps, random_seed=random_seed, progressbar=False)
            idata = approx.sample(draws=draws, random_seed=random_seed)
            loss = approx.hist[-1]
        else: # MCMC
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, 
                              random_seed=random_seed, progressbar=False)
            loss = None
            
        return idata, loss 