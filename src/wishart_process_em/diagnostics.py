r"""Evaluation metrics and downstream analyses for fitted models.

Includes the primary benchmark used in the paper (held-out log-likelihood), a
continuous estimator of **Fisher information** obtained by differentiating the
inferred mean and covariance through the condition space (a natural use of JAX
autodiff), and a **quadratic discriminant analysis** decoder that turns
condition-dependent covariance estimates into a maximum-likelihood classifier.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular

__all__ = [
    "heldout_loglike",
    "fisher_information",
    "fisher_information_curve",
    "qda_predict",
    "qda_accuracy",
    "covariance_operator_norm_error",
]


def heldout_loglike(
    model,
    params,
    Y: jnp.ndarray,
    X: jnp.ndarray,
    key: jax.Array | None = None,
    lattice=None,
    method: str = "auto",
    per_trial: bool = False,
) -> float | jnp.ndarray:
    """Held-out (marginal) log-likelihood under a fitted Wishart process.

    Wraps :func:`wishart_process_em.inference.dataset_marginal_loglike`.  For
    count models prefer ``method="laplace"`` for a lower-variance estimate.  With
    ``per_trial=True`` the mean per-trial log-likelihood is returned instead of
    the total.
    """
    from .inference import dataset_marginal_loglike

    total = dataset_marginal_loglike(
        model, params, Y, X, key=key, lattice=lattice, method=method
    )
    if per_trial:
        return total / jnp.asarray(Y).shape[0]
    return total


def fisher_information(model, params, x: jnp.ndarray) -> jnp.ndarray:
    r"""Fisher information about the condition ``x`` at a single point.

    For a Gaussian response ``N(mu(x), Sigma(x))`` the Fisher information matrix
    with respect to the (possibly multi-dimensional) condition is

    .. math::

        I_{ab}(x) = \partial_a\mu^\top \Sigma^{-1} \partial_b\mu
                    + \tfrac12 \operatorname{tr}\!\big(
                        \Sigma^{-1}\partial_a\Sigma\,
                        \Sigma^{-1}\partial_b\Sigma \big),

    computed by differentiating the inferred mean and covariance through ``x``.
    For count models this uses the latent Gaussian (linear-predictor) mean and
    covariance and is therefore an approximation.

    Returns a ``(D, D)`` matrix (a ``(1, 1)`` matrix for scalar conditions).
    """
    x = jnp.atleast_1d(jnp.asarray(x, dtype=float))
    sigma = model.cov_at(params, x)
    sinv = jnp.linalg.inv(sigma)

    mean_jac = jax.jacobian(lambda z: model.mean_at(params, z))(x)  # (N, D)
    cov_jac = jax.jacobian(lambda z: model.cov_at(params, z))(x)  # (N, N, D)

    # Mean term: (D, D).
    term1 = jnp.einsum("na,nm,mb->ab", mean_jac, sinv, mean_jac)
    # Covariance term: 0.5 * tr(Sinv dSig_a Sinv dSig_b).
    m = jnp.einsum("ij,jka->ika", sinv, cov_jac)  # Sinv @ dSigma_a, shape (N,N,D)
    term2 = 0.5 * jnp.einsum("ika,kib->ab", m, m)
    return term1 + term2


def fisher_information_curve(model, params, X: jnp.ndarray) -> jnp.ndarray:
    """Fisher information at each condition in ``X``.

    Returns shape ``(T, D, D)``.  For scalar conditions, squeeze to ``(T,)`` with
    ``curve[:, 0, 0]`` for a 1-D information curve.
    """
    X = jnp.asarray(X)
    X = X[:, None] if X.ndim == 1 else X
    return jax.vmap(lambda x: fisher_information(model, params, x))(X)


def _gaussian_loglike(Y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-trial ``log N(y_i; mean, cov)``; ``-inf`` if ``cov`` is not PD."""
    n = mean.shape[0]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.full(Y.shape[0], -np.inf)
    sol = solve_triangular(chol, (Y - mean[None, :]).T, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (n * np.log(2 * np.pi) + log_det + np.sum(sol**2, axis=0))


def qda_predict(
    means: np.ndarray, covs: np.ndarray, Y: np.ndarray, jitter: float = 0.0
) -> np.ndarray:
    """Quadratic-discriminant-analysis class predictions.

    Given candidate class means ``(C, N)`` and covariances ``(C, N, N)``, assign
    each trial in ``Y`` ``(T, N)`` to the class of highest Gaussian likelihood.
    Setting all ``covs`` equal recovers linear discriminant analysis (LDA).
    """
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    c, n = means.shape
    eye = jitter * np.eye(n)
    logliks = np.stack(
        [_gaussian_loglike(Y, means[k], covs[k] + eye) for k in range(c)],
        axis=1,
    )  # (T, C)
    return np.argmax(logliks, axis=1)


def qda_accuracy(
    means: np.ndarray,
    covs: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    jitter: float = 0.0,
) -> float:
    """Classification accuracy of the QDA decoder against known ``labels``."""
    pred = qda_predict(means, covs, Y, jitter=jitter)
    return float(np.mean(pred == np.asarray(labels)))


def covariance_operator_norm_error(
    true_covs: np.ndarray, est_covs: np.ndarray
) -> np.ndarray:
    """Per-condition operator-norm (spectral) error between covariance stacks.

    ``||Sigma_true - Sigma_est||_2`` for each condition; a scale-sensitive
    complement to log-likelihood used in the paper's synthetic experiments.
    """
    true_covs = np.asarray(true_covs)
    est_covs = np.asarray(est_covs)
    diff = true_covs - est_covs
    return np.array([np.linalg.norm(d, ord=2) for d in diff])
