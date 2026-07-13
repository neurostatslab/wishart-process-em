"""Tests for model serialisation and the command-line interface."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

from wishart_process_em import (
    WishartProcessModel,
    fourier_basis,
    squared_exponential,
)
from wishart_process_em.cli import main
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


def _write_data(path, model, seed=1, n=200):
    params = model.init_params(jxr.PRNGKey(0))
    X = jnp.linspace(0, 1, n)
    Y, _ = model.sample(jxr.PRNGKey(seed), params, X)
    np.savez(path, Y=np.asarray(Y), X=np.asarray(X))


def test_cli_fit_evaluate_sample_gaussian(tmp_path):
    model = _make_model()
    data = tmp_path / "train.npz"
    _write_data(data, model)
    fit_path = tmp_path / "fit.npz"

    main([
        "fit", str(data), "--likelihood", "gaussian", "--rank", "2",
        "--max-freq", "5", "--lengthscale", "0.3", "--steps", "50",
        "--lr", "0.05", "--no-progress", "-o", str(fit_path),
    ])
    assert fit_path.exists()

    # evaluate runs without error
    main(["evaluate", str(fit_path), str(data)])

    # sample produces a file with the right shapes
    samp = tmp_path / "samp.npz"
    main(["sample", str(fit_path), "--trials", "40", "-o", str(samp)])
    with np.load(samp) as d:
        assert d["Y"].shape == (40, 5)


def test_cli_fit_poisson(tmp_path):
    model = _make_model(likelihood="poisson", init_weight_scale=1.5)
    data = tmp_path / "counts.npz"
    _write_data(data, model, n=150)
    fit_path = tmp_path / "pfit.npz"
    main([
        "fit", str(data), "--likelihood", "poisson", "--rank", "2",
        "--max-freq", "5", "--lengthscale", "0.3", "--steps", "30",
        "--lr", "0.02", "--qmc-points", "101", "--no-progress", "-o", str(fit_path),
    ])
    assert fit_path.exists()
    model2, params2, _ = load_fit(str(fit_path))
    assert not model2.likelihood.conjugate_gaussian
