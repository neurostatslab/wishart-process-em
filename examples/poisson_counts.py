"""Poisson spike-count example with QMC + Laplace latent integration.

Demonstrates the non-Gaussian workflow: the per-trial latent is integrated with
a quasi-Monte-Carlo lattice during fitting and evaluated more precisely with the
Laplace-mixture importance sampler.

Note on identifiability: the noise covariance of Poisson counts is only
recoverable when the shared-latent signal produces meaningful over-dispersion
(Fano factor / (var/mean) noticeably above 1).  This example uses a strong
latent scale so the covariance is well identified; with near-Poisson data
(var/mean ~ 1) the covariance is only weakly constrained.

Run with::

    python examples/poisson_counts.py
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

import wishart_process_em as wpe


def main() -> None:
    basis = wpe.TruncatedFourierBasis(6, 1, wpe.squared_exponential(0.3, 1.0))

    # Strong latent scale -> clear over-dispersion -> identifiable covariance.
    model = wpe.WishartProcessModel(
        basis, num_neurons=8, rank=2, likelihood="poisson", init_weight_scale=2.5
    )
    true_params = model.init_params(jxr.PRNGKey(0), mean_bias=jnp.full(8, 2.5))

    X = jnp.linspace(0, 1, 1500)
    Y, _ = model.sample(jxr.PRNGKey(1), true_params, X)
    fano = np.var(np.asarray(Y), axis=0) / np.mean(np.asarray(Y), axis=0)
    print(f"mean count {float(jnp.mean(Y)):.2f}, "
          f"var/mean per neuron {np.round(fano, 2)}")

    # A shared QMC lattice (dimension = latent dimension of the model).
    lattice = wpe.QMCLattice(211, model.latent_dim, seed=0)

    result = wpe.fit(
        model, Y, X, num_steps=800, learning_rate=2e-2, lattice=lattice, progress=True
    )

    # Evaluate with the higher-accuracy Laplace estimator.
    ll = wpe.heldout_loglike(
        model, result.params, Y, X, key=jxr.PRNGKey(7), lattice=lattice,
        method="laplace", per_trial=True,
    )
    print(f"held-out log-likelihood per trial (Laplace): {float(ll):.3f}")

    # Covariance recovery (of the latent linear predictor).
    grid = jnp.linspace(0, 1, 20)
    true_cov = model.predict_cov(true_params, grid)
    est_cov = model.predict_cov(result.params, grid)
    rel = jnp.linalg.norm(true_cov - est_cov, axis=(1, 2)) / jnp.linalg.norm(
        true_cov, axis=(1, 2)
    )
    print(f"median latent-covariance relative error: {float(jnp.median(rel)):.3f}")


if __name__ == "__main__":
    main()
