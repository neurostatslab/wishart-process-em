"""Quickstart: fit a Gaussian Wishart process to synthetic data.

Simulates smoothly-varying condition-dependent covariance over a 1-D periodic
condition (e.g. grating orientation), fits the model, and compares its held-out
log-likelihood against classical covariance-estimation baselines.

Run with::

    python examples/quickstart.py
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

import wishart_process_em as wpe


def main() -> None:
    # --- A Fourier basis (nemos) over a 1-D condition in [0, 1] --------------
    basis = wpe.fourier_basis(max_freq=8, num_dims=1)

    # --- Ground-truth Gaussian Wishart process ------------------------------
    # The spectral density sets the GP smoothness across conditions.
    model = wpe.WishartProcessModel(
        basis, num_neurons=12, rank=3, likelihood="gaussian",
        spectral_density=wpe.squared_exponential(0.2, 1.0),
    )
    true_params = model.init_params(jxr.PRNGKey(0))

    # Simulate a few trials at each of many conditions (few trials / condition).
    conditions = jnp.linspace(0, 1, 40, endpoint=False)
    X = jnp.repeat(conditions, 8)  # 8 trials per condition
    Y, _ = model.sample(jxr.PRNGKey(1), true_params, X)
    print(f"simulated {Y.shape[0]} trials x {Y.shape[1]} neurons "
          f"({len(conditions)} conditions, 8 trials each)")

    data = wpe.NeuralDataset(np.asarray(Y), np.asarray(X))
    train, test = data.train_test_split(test_frac=0.25, seed=0)

    # --- Fit the Wishart process --------------------------------------------
    result = wpe.fit(
        model,
        jnp.asarray(train.Y),
        jnp.asarray(train.X),
        num_steps=500,
        learning_rate=5e-2,
        progress=True,
    )

    wp_ll = float(
        wpe.heldout_loglike(
            model, result.params, jnp.asarray(test.Y), jnp.asarray(test.X),
            per_trial=True,
        )
    )

    # --- Baselines ----------------------------------------------------------
    print("\nHeld-out log-likelihood per trial (higher is better):")
    print(f"  Wishart process     {wp_ll:8.3f}")
    for method in ["grand", "ledoit_wolf", "weighted_average"]:
        est = wpe.ConditionCovarianceEstimator(method, alpha=0.5).fit(train.Y, train.X)
        ll = np.mean(est.loglike(test.Y, test.X))
        print(f"  {method:18s} {ll:8.3f}")

    # --- Recover the covariance ---------------------------------------------
    grid = jnp.linspace(0, 1, 20)
    true_cov = model.predict_cov(true_params, grid)
    est_cov = model.predict_cov(result.params, grid)
    rel = jnp.linalg.norm(true_cov - est_cov, axis=(1, 2)) / jnp.linalg.norm(
        true_cov, axis=(1, 2)
    )
    print(f"\nmedian covariance relative error: {float(jnp.median(rel)):.3f}")


if __name__ == "__main__":
    main()
