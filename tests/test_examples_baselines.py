"""Smoke tests for the example baseline estimators (``examples/baselines.py``).

The baselines live outside the installed package, so we add ``examples/`` to the
import path before importing them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

baselines = pytest.importorskip("baselines")  # requires scikit-learn


def test_ledoit_wolf_is_positive_definite_for_tiny_samples():
    # Ledoit-Wolf shrinkage can collapse to ~0 for K < N; must stay PD.
    rng = np.random.default_rng(0)
    for k in (2, 3, 5):
        cov = baselines.ledoit_wolf_covariance(rng.normal(size=(k, 12)))
        np.linalg.cholesky(cov)  # raises if not positive definite
        assert np.linalg.eigvalsh(cov).min() > 0


def test_gaussian_loglike_matches_scipy():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T + np.eye(3)
    mean = np.array([0.5, -1.0, 2.0])
    y = np.array([1.0, 0.0, 1.5])
    got = baselines.gaussian_loglike(y, mean, cov)[0]
    assert np.isclose(got, multivariate_normal.logpdf(y, mean, cov))


def test_gaussian_loglike_singular_is_neginf():
    y = np.ones((2, 3))
    singular = np.outer(np.ones(3), np.ones(3))  # rank 1
    assert np.all(np.isneginf(baselines.gaussian_loglike(y, np.zeros(3), singular)))


@pytest.mark.parametrize("method", ["empirical", "grand", "ledoit_wolf", "weighted_average"])
def test_condition_estimator_fit_predict_loglike(method):
    rng = np.random.default_rng(1)
    C, K, N = 8, 15, 5
    X = np.repeat(np.linspace(0, 1, C, endpoint=False), K)
    Y = rng.normal(size=(C * K, N))
    est = baselines.ConditionCovarianceEstimator(method, alpha=0.5).fit(Y, X)
    means, covs = est.predict(np.linspace(0, 1, C, endpoint=False))
    assert means.shape == (C, N)
    assert covs.shape == (C, N, N)
    assert est.loglike(Y[:10], X[:10]).shape == (10,)
