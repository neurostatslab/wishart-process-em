r"""Classical covariance-estimation baselines.

These are the standard estimators the Wishart process is compared against in the
paper: the per-condition empirical covariance (eq. 1), the grand empirical
covariance pooled across conditions (eq. 2), the Ledoit-Wolf shrinkage
estimator, the graphical LASSO, and a weighted average that shrinks the
per-condition estimate towards the grand covariance.

Unlike the Wishart process, none of these exploit the smooth parameterisation of
conditions, so they cannot borrow statistical strength across neighbouring
conditions or interpolate covariance to unseen conditions.  They are provided
here for benchmarking on the same held-out log-likelihood metric.

All estimators operate on ``numpy`` arrays (they are not differentiable).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .data import group_by_condition

__all__ = [
    "empirical_covariance",
    "grand_empirical_covariance",
    "ledoit_wolf_covariance",
    "graphical_lasso_covariance",
    "gaussian_loglike",
    "ConditionCovarianceEstimator",
]


def empirical_covariance(group: np.ndarray) -> np.ndarray:
    """Maximum-likelihood (biased) covariance of a ``(K, N)`` block of trials."""
    return np.cov(np.asarray(group), rowvar=False, bias=True)


def grand_empirical_covariance(groups: list[np.ndarray]) -> np.ndarray:
    """Grand empirical covariance: pool within-condition-centred trials (eq. 2)."""
    centered = [g - g.mean(axis=0, keepdims=True) for g in groups]
    stacked = np.concatenate(centered, axis=0)
    return np.cov(stacked, rowvar=False, bias=True)


def ledoit_wolf_covariance(group: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance, shrinking towards a scaled identity target.

    Uses ``sklearn`` when available (the reference implementation); otherwise
    falls back to the closed-form Ledoit-Wolf (2004) shrinkage intensity.
    """
    group = np.asarray(group, dtype=float)
    try:
        from sklearn.covariance import ledoit_wolf

        cov, _ = ledoit_wolf(group)
    except ImportError:
        cov = _ledoit_wolf_fallback(group)
    return _ensure_positive_definite(cov)


def _ensure_positive_definite(cov: np.ndarray, rel_floor: float = 1e-8) -> np.ndarray:
    """Symmetrise and guarantee a positive-definite covariance.

    Ledoit-Wolf's data-driven shrinkage can collapse to ~0 for very small
    samples (e.g. two trials in a condition), returning the singular empirical
    covariance.  Since the point of a shrinkage estimator is to be
    well-conditioned, we clip its smallest eigenvalue up to a tiny fraction of
    the largest (a no-op when the estimate is already well-conditioned).
    """
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    floor = rel_floor * max(float(eigvals[-1]), 1.0)
    if eigvals[0] < floor:
        cov = cov + (floor - eigvals[0]) * np.eye(n)
    return cov


def _ledoit_wolf_fallback(X: np.ndarray) -> np.ndarray:
    k, n = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    s = (Xc.T @ Xc) / k
    mu = np.trace(s) / n
    target = mu * np.eye(n)
    d2 = np.sum((s - target) ** 2)
    # Expected error of the empirical covariance (b^2), clipped to [0, d2].
    b2 = np.sum(
        [np.sum((np.outer(x, x) - s) ** 2) for x in Xc]
    ) / (k**2)
    b2 = min(b2, d2)
    shrink = 0.0 if d2 == 0 else b2 / d2
    return shrink * target + (1.0 - shrink) * s


def graphical_lasso_covariance(group: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Graphical-LASSO covariance (sparse inverse covariance); needs ``sklearn``."""
    try:
        from sklearn.covariance import graphical_lasso
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "graphical_lasso_covariance requires scikit-learn; "
            "install with `pip install 'wishart-process-em[viz]'`"
        ) from exc
    group = np.asarray(group, dtype=float)
    emp = np.cov(group, rowvar=False, bias=True)
    cov, _ = graphical_lasso(emp, alpha=alpha)
    return cov


def gaussian_loglike(
    Y: np.ndarray, mean: np.ndarray, cov: np.ndarray, jitter: float = 0.0
) -> np.ndarray:
    """Per-trial Gaussian log-likelihood ``log N(y_i; mean, cov)``.

    Returns ``-inf`` for trials when ``cov`` is not positive definite (e.g. the
    singular empirical covariance in the ``K < N`` regime), matching the paper's
    treatment of degenerate baselines.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    mean = np.asarray(mean, dtype=float)
    n = mean.shape[0]
    cov = np.asarray(cov, dtype=float) + jitter * np.eye(n)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.full(Y.shape[0], -np.inf)
    diff = Y - mean[None, :]
    sol = solve_triangular(chol, diff.T, lower=True)
    quad = np.sum(sol**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (n * np.log(2 * np.pi) + log_det + quad)


class ConditionCovarianceEstimator:
    """Fit per-condition mean/covariance with a classical estimator.

    Parameters
    ----------
    method : {"empirical", "grand", "ledoit_wolf", "graphical_lasso", "weighted_average"}
        Which estimator to use.
    alpha : float
        For ``"weighted_average"``, the weight on the per-condition empirical
        covariance (``cov = alpha * emp + (1 - alpha) * grand``); for
        ``"graphical_lasso"``, the sparsity penalty.
    """

    def __init__(self, method: str = "ledoit_wolf", alpha: float = 0.5) -> None:
        valid = {"empirical", "grand", "ledoit_wolf", "graphical_lasso", "weighted_average"}
        if method not in valid:
            raise ValueError(f"method must be one of {sorted(valid)}")
        self.method = method
        self.alpha = alpha
        self.conditions_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covs_: np.ndarray | None = None

    def fit(self, Y: np.ndarray, X: np.ndarray) -> ConditionCovarianceEstimator:
        """Estimate per-condition means and covariances from training trials."""
        uniq, groups = group_by_condition(Y, X)
        grand = grand_empirical_covariance(groups)
        means = np.stack([g.mean(axis=0) for g in groups])

        covs = []
        for g in groups:
            if self.method == "empirical":
                c = empirical_covariance(g)
            elif self.method == "grand":
                c = grand
            elif self.method == "ledoit_wolf":
                c = ledoit_wolf_covariance(g)
            elif self.method == "graphical_lasso":
                c = graphical_lasso_covariance(g, alpha=self.alpha)
            else:  # weighted_average
                c = self.alpha * empirical_covariance(g) + (1 - self.alpha) * grand
            covs.append(c)

        self.conditions_ = uniq
        self.means_ = means
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
        out = np.empty(Y.shape[0])
        for i, c in enumerate(idx):
            out[i] = gaussian_loglike(
                Y[i], self.means_[c], self.covs_[c], jitter=jitter
            )[0]
        return out
