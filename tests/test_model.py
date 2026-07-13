"""Tests for likelihoods, covariance assembly, and the model."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np
import pytest

from wishart_process_em import (
    Gaussian,
    NegativeBinomial,
    Poisson,
    WishartProcessModel,
    fourier_basis,
    get_likelihood,
)
from wishart_process_em.covariance import (
    raw_from_scale_tril,
    scale_tril_from_raw,
    softplus_inverse,
)


# --------------------------------------------------------------------------- #
# Likelihoods
# --------------------------------------------------------------------------- #
def test_get_likelihood_registry():
    assert isinstance(get_likelihood("gaussian"), Gaussian)
    assert isinstance(get_likelihood("poisson"), Poisson)
    assert isinstance(get_likelihood("nb"), NegativeBinomial)
    with pytest.raises(ValueError):
        get_likelihood("nonsense")


def test_gaussian_marginal_matches_scipy():
    from jax.scipy.stats import multivariate_normal

    lik = Gaussian()
    key = jxr.PRNGKey(0)
    a = jxr.normal(key, (4, 4))
    cov = a @ a.T + jnp.eye(4)
    mean = jnp.arange(4.0)
    y = jnp.ones(4)
    got = lik.marginal_log_prob(None, y, mean, cov)
    expected = multivariate_normal.logpdf(y, mean, cov)
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_poisson_log_prob_and_mean_positive():
    lik = Poisson()
    eta = jnp.array([-2.0, 0.0, 3.0])
    y = jnp.array([0.0, 1.0, 4.0])
    assert jnp.isfinite(lik.log_prob(None, y, eta))
    assert jnp.all(lik.mean(None, eta) > 0)


def test_nb_approaches_poisson_for_large_dispersion():
    nb = NegativeBinomial()
    pois = Poisson()
    eta = jnp.array([0.5, 1.0, 2.0])
    y = jnp.array([1.0, 2.0, 3.0])
    big_r = {"log_r": jnp.full(3, softplus_inverse(jnp.asarray(1e6)))}
    np.testing.assert_allclose(
        nb.log_prob(big_r, y, eta), pois.log_prob(None, y, eta), atol=1e-3
    )


@pytest.mark.parametrize("name", ["gaussian", "poisson", "negative_binomial"])
def test_observe_shapes(name):
    lik = get_likelihood(name)
    eta = jnp.array([0.5, 1.0, -0.5])
    params = lik.init_params(3, jxr.PRNGKey(0))
    y = lik.observe(jxr.PRNGKey(1), params, eta)
    assert y.shape == eta.shape


# --------------------------------------------------------------------------- #
# Covariance helpers
# --------------------------------------------------------------------------- #
def test_softplus_inverse_roundtrip():
    import jax

    y = jnp.array([0.1, 1.0, 5.0])
    np.testing.assert_allclose(jax.nn.softplus(softplus_inverse(y)), y, atol=1e-6)


def test_scale_tril_roundtrip_and_lower_triangular():
    a = jxr.normal(jxr.PRNGKey(0), (4, 4))
    tril = jnp.tril(a, -1) + jnp.diag(jnp.abs(jnp.diagonal(a)) + 0.5)
    reconstructed = scale_tril_from_raw(raw_from_scale_tril(tril))
    np.testing.assert_allclose(reconstructed, tril, atol=1e-6)
    L = scale_tril_from_raw(a)
    assert np.allclose(jnp.triu(L, 1), 0.0)  # lower-triangular
    assert jnp.all(jnp.diagonal(L) > 0)  # positive diagonal


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_model_rank0_requires_diagonal(basis):
    with pytest.raises(ValueError):
        WishartProcessModel(basis, num_neurons=4, rank=0, use_diagonal=False)


@pytest.mark.parametrize("use_diagonal,use_scale", [(False, False), (True, False), (False, True), (True, True)])
def test_covariance_is_symmetric_psd(basis, use_diagonal, use_scale):
    model = WishartProcessModel(
        basis, num_neurons=5, rank=2, use_diagonal=use_diagonal, use_scale=use_scale
    )
    params = model.init_params(jxr.PRNGKey(0))
    cov = model.cov_at(params, jnp.array([0.4]))
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    eigvals = jnp.linalg.eigvalsh(cov)
    assert float(eigvals.min()) > 0


def test_latent_dim():
    b = fourier_basis(4, num_dims=1)
    assert WishartProcessModel(b, 6, rank=3).latent_dim == 3
    assert WishartProcessModel(b, 6, rank=3, use_diagonal=True).latent_dim == 9


def test_predict_shapes(gaussian_model):
    params = gaussian_model.init_params(jxr.PRNGKey(0))
    X = jnp.linspace(0, 1, 7)
    assert gaussian_model.predict_mean(params, X).shape == (7, 5)
    assert gaussian_model.predict_cov(params, X).shape == (7, 5, 5)


def test_sample_covariance_matches_predict_cov(gaussian_model):
    # For a Gaussian model, empirical covariance at a fixed condition should
    # match predict_cov (statistically).
    params = gaussian_model.init_params(jxr.PRNGKey(0))
    x = jnp.array([0.3])
    X = jnp.repeat(x, 4000, axis=0)[:, None] if x.ndim else x
    X = jnp.full((4000, 1), 0.3)
    Y, _ = gaussian_model.sample(jxr.PRNGKey(2), params, X)
    emp = jnp.cov(Y, rowvar=False)
    true = gaussian_model.cov_at(params, jnp.array([0.3]))
    rel = jnp.linalg.norm(emp - true) / jnp.linalg.norm(true)
    assert float(rel) < 0.15


def test_multidim_conditions():
    b = fourier_basis(4, num_dims=2)
    model = WishartProcessModel(b, num_neurons=4, rank=2)
    params = model.init_params(jxr.PRNGKey(0))
    X = jxr.uniform(jxr.PRNGKey(1), (12, 2))
    Y, _ = model.sample(jxr.PRNGKey(2), params, X)
    assert Y.shape == (12, 4)
    assert model.predict_cov(params, X).shape == (12, 4, 4)
