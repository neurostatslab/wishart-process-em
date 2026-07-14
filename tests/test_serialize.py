"""Tests for saving and loading fitted models."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

from wishart_process_em import (
    WishartProcessModel,
    fourier_basis,
    squared_exponential,
)
from wishart_process_em.serialize import load_fit, save_fit


def _make_model(likelihood="gaussian", **kw):
    basis = fourier_basis(5, num_dims=1)
    return WishartProcessModel(
        basis, num_neurons=5, rank=2, likelihood=likelihood,
        spectral_density=squared_exponential(0.3, 1.0), **kw,
    )


def test_serialize_roundtrip_reproduces_covariance(tmp_path):
    model = _make_model(use_scale=True, use_diagonal=True)
    params = model.init_params(jxr.PRNGKey(0))
    kernel = {"type": "squared_exponential", "lengthscale": 0.3, "variance": 1.0}
    p = tmp_path / "fit.npz"
    save_fit(str(p), model, params, kernel)

    model2, params2, config = load_fit(str(p))
    X = jnp.linspace(0, 1, 10)
    np.testing.assert_allclose(
        model.predict_cov(params, X), model2.predict_cov(params2, X), atol=1e-8
    )
    assert config["model"]["use_scale"] and config["model"]["use_diagonal"]


def test_serialize_roundtrip_negative_binomial(tmp_path):
    model = _make_model(likelihood="negative_binomial")
    params = model.init_params(jxr.PRNGKey(0))
    kernel = {"type": "squared_exponential", "lengthscale": 0.3, "variance": 1.0}
    p = tmp_path / "nb.npz"
    save_fit(str(p), model, params, kernel)
    model2, params2, _ = load_fit(str(p))
    assert params2.lik is not None and "log_r" in params2.lik
    np.testing.assert_allclose(params.lik["log_r"], params2.lik["log_r"])
