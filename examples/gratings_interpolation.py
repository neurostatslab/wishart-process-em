"""Interpolating noise covariance to unseen conditions (2-D condition space).

Mirrors the drifting-gratings analysis of the paper: a 2-D condition space of
grating *orientation* (periodic) and *temporal frequency* (non-periodic).  We
fit the Wishart process on a subset of conditions and predict the mean and
covariance at entirely held-out conditions, exploiting the smoothness of the
process to interpolate where no trials were observed.

Run with::

    python examples/gratings_interpolation.py
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

import wishart_process_em as wpe


def main() -> None:
    # --- Build a 2-D grid of conditions: orientation x temporal frequency ---
    n_orient, n_tf, trials_per_cond = 8, 5, 12
    orient = np.linspace(0, 1, n_orient, endpoint=False)  # periodic in [0,1)
    tf = np.linspace(0.1, 0.9, n_tf)  # non-periodic
    grid = np.array([[o, f] for o in orient for f in tf])  # (40, 2)

    # --- Ground-truth Gaussian Wishart process ------------------------------
    basis = wpe.TruncatedFourierBasis(5, 2, wpe.squared_exponential(0.3, 1.0))
    model = wpe.WishartProcessModel(basis, num_neurons=10, rank=3, likelihood="gaussian")
    true_params = model.init_params(jxr.PRNGKey(0))

    X = jnp.asarray(np.repeat(grid, trials_per_cond, axis=0))
    Y, _ = model.sample(jxr.PRNGKey(1), true_params, X)
    print(f"{Y.shape[0]} trials, {Y.shape[1]} neurons, "
          f"{len(grid)} conditions ({trials_per_cond}/condition)")

    data = wpe.NeuralDataset(np.asarray(Y), np.asarray(X))

    # --- Hold out entire conditions (interpolation test) --------------------
    train, held = data.holdout_conditions(frac=0.25, seed=3)
    held_conditions = np.unique(held.X, axis=0)
    print(f"training on {len(np.unique(train.X, axis=0))} conditions, "
          f"holding out {len(held_conditions)} conditions entirely")

    result = wpe.fit(
        model, jnp.asarray(train.Y), jnp.asarray(train.X),
        num_steps=600, learning_rate=5e-2, progress=True,
    )

    # --- Predict covariance at the UNSEEN conditions ------------------------
    wp_ll = float(
        wpe.heldout_loglike(
            model, result.params, jnp.asarray(held.Y), jnp.asarray(held.X),
            per_trial=True,
        )
    )
    # Baseline can only fall back to the nearest observed condition.
    baseline = wpe.ConditionCovarianceEstimator("weighted_average", alpha=0.5)
    baseline.fit(train.Y, train.X)
    base_ll = float(np.mean(baseline.loglike(held.Y, held.X)))

    print("\nHeld-out-condition log-likelihood per trial:")
    print(f"  Wishart process (interpolates)   {wp_ll:8.3f}")
    print(f"  weighted-average (nearest cond.) {base_ll:8.3f}")

    # Covariance error at the unseen conditions.
    true_cov = model.predict_cov(true_params, jnp.asarray(held_conditions))
    est_cov = model.predict_cov(result.params, jnp.asarray(held_conditions))
    rel = jnp.linalg.norm(true_cov - est_cov, axis=(1, 2)) / jnp.linalg.norm(
        true_cov, axis=(1, 2)
    )
    print(f"\nmedian covariance error at unseen conditions: {float(jnp.median(rel)):.3f}")


if __name__ == "__main__":
    main()
