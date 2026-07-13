"""Tests for the marginal-likelihood estimators and the fitting driver."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

from wishart_process_em import (
    QMCLattice,
    WishartProcessModel,
    dataset_marginal_loglike,
    fit,
    fourier_basis,
    squared_exponential,
)


def test_conjugate_matches_manual(gaussian_model, gaussian_data):
    Y, X, params = gaussian_data
    # Manual per-trial Gaussian logpdf summed.
    from jax.scipy.stats import multivariate_normal

    X2 = X[:, None]
    total = 0.0
    for i in range(Y.shape[0]):
        mu = gaussian_model.mean_at(params, X2[i])
        cov = gaussian_model.cov_at(params, X2[i])
        total += float(multivariate_normal.logpdf(Y[i], mu, cov))
    got = float(dataset_marginal_loglike(gaussian_model, params, Y, X))
    np.testing.assert_allclose(got, total, rtol=1e-6)


def test_auto_method_uses_conjugate_for_gaussian(gaussian_model, gaussian_data):
    Y, X, params = gaussian_data
    # No key/lattice needed -> confirms the conjugate path is taken.
    val = dataset_marginal_loglike(gaussian_model, params, Y, X, method="auto")
    assert jnp.isfinite(val)


def test_qmc_and_laplace_agree_for_counts(poisson_model):
    params = poisson_model.init_params(jxr.PRNGKey(0), mean_bias=jnp.full(5, 2.0))
    X = jnp.linspace(0, 1, 60)
    Y, _ = poisson_model.sample(jxr.PRNGKey(1), params, X)
    lat = QMCLattice(211, poisson_model.latent_dim, seed=0)
    key = jxr.PRNGKey(2)
    qmc = float(dataset_marginal_loglike(poisson_model, params, Y, X, key=key, lattice=lat, method="qmc"))
    lap = float(dataset_marginal_loglike(poisson_model, params, Y, X, key=key, lattice=lat, method="laplace"))
    # Both estimate the same integral; agree to within a few nats over 60 trials.
    assert abs(qmc - lap) / abs(lap) < 0.02


def test_dataset_marginal_requires_lattice_for_counts(poisson_model):
    params = poisson_model.init_params(jxr.PRNGKey(0))
    X = jnp.linspace(0, 1, 10)
    Y, _ = poisson_model.sample(jxr.PRNGKey(1), params, X)
    import pytest

    with pytest.raises(ValueError):
        dataset_marginal_loglike(poisson_model, params, Y, X, method="qmc")


def test_gaussian_fit_improves_and_recovers():
    basis = fourier_basis(6, num_dims=1)
    model = WishartProcessModel(
        basis, num_neurons=6, rank=2, likelihood="gaussian",
        spectral_density=squared_exponential(0.3, 1.0),
    )
    true = model.init_params(jxr.PRNGKey(0))
    X = jnp.linspace(0, 1, 600)
    Y, _ = model.sample(jxr.PRNGKey(1), true, X)

    result = fit(model, Y, X, num_steps=400, learning_rate=5e-2, seed=0, progress=False)
    ll_init = dataset_marginal_loglike(model, result.init_params, Y, X)
    ll_fit = dataset_marginal_loglike(model, result.params, Y, X)
    assert float(ll_fit) > float(ll_init)

    Xg = jnp.linspace(0, 1, 20)
    Ct = model.predict_cov(true, Xg)
    Cf = model.predict_cov(result.params, Xg)
    rel = jnp.linalg.norm(Ct - Cf, axis=(1, 2)) / jnp.linalg.norm(Ct, axis=(1, 2))
    assert float(jnp.median(rel)) < 0.25


def test_poisson_fit_improves_objective():
    basis = fourier_basis(5, num_dims=1)
    model = WishartProcessModel(
        basis, num_neurons=5, rank=2, likelihood="poisson", init_weight_scale=2.0,
        spectral_density=squared_exponential(0.3, 1.0),
    )
    true = model.init_params(jxr.PRNGKey(0), mean_bias=jnp.full(5, 2.5))
    X = jnp.linspace(0, 1, 400)
    Y, _ = model.sample(jxr.PRNGKey(1), true, X)
    result = fit(model, Y, X, num_steps=150, learning_rate=2e-2, seed=0, progress=False)
    assert result.loss_history[-1] < result.loss_history[0]
    assert np.all(np.isfinite(result.loss_history))
