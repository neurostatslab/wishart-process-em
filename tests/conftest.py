"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import pytest

import wishart_process_em as wpe  # noqa: F401  (import enables x64)
from wishart_process_em import (
    WishartProcessModel,
    fourier_basis,
    squared_exponential,
)

SPECTRAL_DENSITY = squared_exponential(0.3, 1.0)


@pytest.fixture
def basis():
    return fourier_basis(6, num_dims=1)


@pytest.fixture
def gaussian_model(basis):
    return WishartProcessModel(
        basis, num_neurons=5, rank=2, likelihood="gaussian",
        spectral_density=SPECTRAL_DENSITY,
    )


@pytest.fixture
def poisson_model(basis):
    return WishartProcessModel(
        basis, num_neurons=5, rank=2, likelihood="poisson", init_weight_scale=2.0,
        spectral_density=SPECTRAL_DENSITY,
    )


@pytest.fixture
def gaussian_data(gaussian_model):
    """A small synthetic Gaussian dataset and its generating parameters."""
    params = gaussian_model.init_params(jxr.PRNGKey(0))
    X = jnp.linspace(0, 1, 300)
    Y, _ = gaussian_model.sample(jxr.PRNGKey(1), params, X)
    return Y, X, params
