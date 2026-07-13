"""Classical covariance-estimation baselines (example / comparison code).

Lightweight, self-contained implementations of the standard estimators the
Wishart process is compared against in the paper: the per-condition empirical
covariance, the grand empirical covariance pooled across conditions, the
Ledoit-Wolf shrinkage estimator, and a weighted average that shrinks the
per-condition estimate towards the grand covariance.

These live in ``examples/`` rather than the installed package: they are baselines
for benchmarking, not part of the Wishart process model.  Unlike the model, none
of them exploit the smooth parameterisation of conditions, so they cannot borrow
strength across neighbouring conditions or interpolate to unseen ones.

Requires ``numpy``, ``scipy`` and ``scikit-learn`` (``pip install
"wishart-process-em[viz]"``).  The only dependency on the package itself is
``group_by_condition``, a generic data utility.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import ledoit_wolf

from wishart_process_em import group_by_condition

__all__ = [
    "empirical_covariance",
    "grand_empirical_covariance",
    "ledoit_wolf_covariance",
    "gaussian_loglike",
    "ConditionCovarianceEstimator",
]


def _ensure_positive_definite(cov: np.ndarray, rel_floor: float = 1e-8) -> np.ndarray:
    """Symmetrise and guarantee a positive-definite covariance.

    Ledoit-Wolf's data-driven shrinkage can collapse to ~0 for very small
    samples (e.g. two trials in a condition), returning the singular empirical
    covariance.  Clip the smallest eigenvalue up to a tiny fraction of the
    largest (a no-op when the estimate is already well-conditioned).
    """
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    floor = rel_floor * max(float(eigvals[-1]), 1.0)
    if eigvals[0] < floor:
        cov = cov + (floor - eigvals[0]) * np.eye(n)
    return cov


def empirical_covariance(group: np.ndarray) -> np.ndarray:
    """Maximum-likelihood (biased) covariance of a ``(K, N)`` block of trials."""
    return np.cov(np.asarray(group), rowvar=False, bias=True)


def grand_empirical_covariance(groups: list[np.ndarray]) -> np.ndarray:
    """Grand empirical covariance: pool within-condition-centred trials."""
    centered = [g - g.mean(axis=0, keepdims=True) for g in groups]
    return np.cov(np.concatenate(centered, axis=0), rowvar=False, bias=True)


def ledoit_wolf_covariance(group: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance (shrinkage towards a scaled identity), kept PD."""
    cov, _ = ledoit_wolf(np.asarray(group, dtype=float))
    return _ensure_positive_definite(cov)


def gaussian_loglike(
    Y: np.ndarray, mean: np.ndarray, cov: np.ndarray, jitter: float = 0.0
) -> np.ndarray:
    """Per-trial Gaussian log-likelihood ``log N(y_i; mean, cov)``.

    Returns ``-inf`` for a trial when ``cov`` is not positive definite (e.g. the
    singular empirical covariance when ``K < N``).
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    mean = np.asarray(mean, dtype=float)
    n = mean.shape[0]
    cov = np.asarray(cov, dtype=float) + jitter * np.eye(n)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.full(Y.shape[0], -np.inf)
    sol = solve_triangular(chol, (Y - mean[None, :]).T, lower=True)
    quad = np.sum(sol**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (n * np.log(2 * np.pi) + log_det + quad)


class ConditionCovarianceEstimator:
    """Fit per-condition mean/covariance with a classical estimator.

    Parameters
    ----------
    method : {"empirical", "grand", "ledoit_wolf", "weighted_average"}
        Which estimator to use.
    alpha : float
        For ``"weighted_average"``, the weight on the per-condition empirical
        covariance: ``cov = alpha * emp + (1 - alpha) * grand``.
    """

    def __init__(self, method: str = "ledoit_wolf", alpha: float = 0.5) -> None:
        valid = {"empirical", "grand", "ledoit_wolf", "weighted_average"}
        if method not in valid:
            raise ValueError(f"method must be one of {sorted(valid)}")
        self.method = method
        self.alpha = alpha

    def fit(self, Y: np.ndarray, X: np.ndarray) -> ConditionCovarianceEstimator:
        """Estimate per-condition means and covariances from training trials."""
        uniq, groups = group_by_condition(Y, X)
        grand = grand_empirical_covariance(groups)
        self.conditions_ = uniq
        self.means_ = np.stack([g.mean(axis=0) for g in groups])

        covs = []
        for g in groups:
            if self.method == "empirical":
                c = empirical_covariance(g)
            elif self.method == "grand":
                c = grand
            elif self.method == "ledoit_wolf":
                c = ledoit_wolf_covariance(g)
            else:  # weighted_average
                c = self.alpha * empirical_covariance(g) + (1 - self.alpha) * grand
            covs.append(c)
        self.covs_ = np.stack(covs)
        return self

    def _match(self, X_query: np.ndarray) -> np.ndarray:
        """Nearest training condition for each query (baselines cannot interpolate)."""
        X_query = np.asarray(X_query, dtype=float)
        if X_query.ndim == 1:
            X_query = X_query[:, None]
        keys = X_query.reshape(X_query.shape[0], -1)
        dists = np.linalg.norm(keys[:, None, :] - self.conditions_[None], axis=2)
        return np.argmin(dists, axis=1)

    def predict(self, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(means, covs)`` for query conditions via nearest training match."""
        idx = self._match(X_query)
        return self.means_[idx], self.covs_[idx]

    def loglike(self, Y: np.ndarray, X: np.ndarray, jitter: float = 0.0) -> np.ndarray:
        """Per-trial Gaussian held-out log-likelihood under the fitted estimates."""
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        idx = self._match(X)
        return np.array(
            [
                gaussian_loglike(Y[i], self.means_[c], self.covs_[c], jitter=jitter)[0]
                for i, c in enumerate(idx)
            ]
        )
