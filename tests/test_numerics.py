"""Tests for the core numerics: basis, kernels, QMC, and the Newton optimiser."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jxr
import numpy as np
import pytest

from wishart_process_em import (
    QMCLattice,
    TruncatedFourierBasis,
    matern,
    newton_minimize,
    squared_exponential,
)
from wishart_process_em.qmc import (
    find_optimal_generator,
    korobov_points,
    star_discrepancy,
)


# --------------------------------------------------------------------------- #
# Basis / kernels
# --------------------------------------------------------------------------- #
def test_basis_feature_shape():
    b = TruncatedFourierBasis(6, 1, squared_exponential(0.3))
    phi = b.features(jnp.array([0.3]))
    assert phi.shape == (b.num_features,)
    assert b.num_features % 2 == 0  # sine + cosine pairs


def test_basis_is_periodic():
    # Fourier modes have integer frequencies -> period 1 in each dimension.
    b = TruncatedFourierBasis(5, 1, squared_exponential(0.25))
    x = jnp.array([0.2])
    np.testing.assert_allclose(b.features(x), b.features(x + 1.0), atol=1e-10)


def test_basis_vmap_batches():
    b = TruncatedFourierBasis(4, 2, squared_exponential(0.3))
    X = jxr.uniform(jxr.PRNGKey(0), (10, 2))
    feats = jax.vmap(b.features)(X)
    assert feats.shape == (10, b.num_features)


def test_basis_pruning_reduces_features():
    dense = TruncatedFourierBasis(8, 1, squared_exponential(0.5), tol=1e-8)
    sparse = TruncatedFourierBasis(8, 1, squared_exponential(0.5), tol=1e-1)
    assert sparse.num_features < dense.num_features


@pytest.mark.parametrize("density", [squared_exponential(0.3), matern(0.3, 1.5)])
def test_spectral_density_positive_decreasing(density):
    sq_omega = jnp.array([0.0, 1.0, 10.0, 100.0])
    vals = density(sq_omega)
    assert jnp.all(vals > 0)
    assert jnp.all(jnp.diff(vals) <= 0)  # non-increasing in frequency


# --------------------------------------------------------------------------- #
# QMC
# --------------------------------------------------------------------------- #
def test_korobov_points_in_unit_cube():
    pts = korobov_points(151, 13, 5)
    assert pts.shape == (151, 5)
    assert pts.min() >= 0.0 and pts.max() < 1.0


def test_lattice_gaussian_points_standard_normal():
    lat = QMCLattice(211, 3, seed=0)
    z = lat.gaussian_points(jxr.PRNGKey(0))
    assert z.shape == (211, 3)
    assert abs(float(jnp.mean(z))) < 0.1
    assert abs(float(jnp.std(z)) - 1.0) < 0.1


def test_lattice_randomization_differs_by_key():
    lat = QMCLattice(101, 2, generator=7)
    a = lat.gaussian_points(jxr.PRNGKey(0))
    b = lat.gaussian_points(jxr.PRNGKey(1))
    assert not np.allclose(a, b)


def test_find_optimal_generator_valid():
    # The discrepancy is a Monte-Carlo estimate, so only check it is a valid,
    # positive value and that the chosen generator is in range.
    a, disc = find_optimal_generator(53, 3, samples=500, rng=np.random.default_rng(0))
    assert 1 <= a < 53
    assert 0.0 < disc < 1.0


def test_star_discrepancy_deterministic_with_seed():
    d1 = star_discrepancy(53, 13, 3, num_samples=500, rng=np.random.default_rng(0))
    d2 = star_discrepancy(53, 13, 3, num_samples=500, rng=np.random.default_rng(0))
    assert d1 == d2


# --------------------------------------------------------------------------- #
# Newton optimiser
# --------------------------------------------------------------------------- #
def test_newton_quadratic_exact():
    A = jnp.array([3.0, 1.0])
    bvec = jnp.array([-1.0, 2.0])

    def f(x):
        return 0.5 * jnp.dot(x, A * x) + jnp.dot(bvec, x)

    res = newton_minimize(f, jnp.zeros(2))
    np.testing.assert_allclose(res.x, -bvec / A, atol=1e-4)
    assert bool(res.converged)
    # Hessian Cholesky: H = diag(A) -> L = diag(sqrt(A)).
    np.testing.assert_allclose(res.hess_chol @ res.hess_chol.T, jnp.diag(A), atol=1e-5)


def test_newton_nonquadratic_converges():
    def f(x):
        return jnp.sum(jnp.cosh(x)) + 0.5 * jnp.sum(x**2)

    res = newton_minimize(f, jnp.array([2.0, -3.0, 1.5]))
    np.testing.assert_allclose(res.x, jnp.zeros(3), atol=1e-4)


def test_newton_vmaps():
    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    solve = jax.vmap(lambda x0: newton_minimize(f, x0).x)
    xs = jxr.normal(jxr.PRNGKey(0), (5, 3))
    out = solve(xs)
    np.testing.assert_allclose(out, jnp.ones((5, 3)), atol=1e-5)
