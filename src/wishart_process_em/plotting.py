r"""Plotting helpers (optional; require ``matplotlib`` and ``scikit-learn``).

Visualise inferred means and condition-dependent covariances as ellipses in the
top-two principal-component subspace of the data, mirroring the figures in the
paper.  These helpers are imported lazily so the core package has no hard
dependency on ``matplotlib`` / ``scikit-learn``; install them with
``pip install 'wishart-process-em[viz]'``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["covariance_ellipse", "plot_covariance_pca"]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting requires matplotlib; install `pip install "
            "'wishart-process-em[viz]'`"
        ) from exc


def covariance_ellipse(mean, cov, ax=None, n_std: float = 1.0, num_pts: int = 100, **kwargs):
    """Plot a covariance ellipse for a 2-D Gaussian ``(mean, cov)``.

    Parameters
    ----------
    mean : array, shape (2,)
    cov : array, shape (2, 2)
    ax : matplotlib Axes, optional
    n_std : float
        Radius of the ellipse in standard deviations.
    """
    plt = _require_matplotlib()
    ax = ax or plt.gca()
    mean = np.asarray(mean)
    chol = np.linalg.cholesky(np.asarray(cov) + 1e-12 * np.eye(2))
    theta = np.linspace(0, 2 * np.pi, num_pts)
    circle = np.stack([np.cos(theta), np.sin(theta)])  # (2, num_pts)
    pts = mean[:, None] + n_std * chol @ circle
    return ax.plot(pts[0], pts[1], **kwargs)


def plot_covariance_pca(
    model,
    params,
    X_grid,
    data=None,
    ax=None,
    n_std: float = 1.0,
    pca=None,
    ellipse_kwargs: dict | None = None,
    mean_kwargs: dict | None = None,
):
    """Plot inferred mean path and covariance ellipses in top-2 PC space.

    Parameters
    ----------
    model, params : WishartProcessModel, WPParams
    X_grid : array
        Conditions at which to draw the mean and covariance ellipses.
    data : array, optional
        Observations ``(T, N)`` to scatter and to fit the PCA on.  If ``None``,
        a ``pca`` object must be supplied.
    pca : sklearn PCA, optional
        Pre-fit 2-component PCA; fit on ``data`` if not given.
    """
    plt = _require_matplotlib()
    ax = ax or plt.gca()
    ellipse_kwargs = ellipse_kwargs or {"color": "C0"}
    mean_kwargs = mean_kwargs or {"color": "r"}

    if pca is None:
        from sklearn.decomposition import PCA

        if data is None:
            raise ValueError("provide `data` to fit a PCA, or pass a fitted `pca`")
        pca = PCA(2).fit(np.asarray(data))

    if data is not None:
        proj = pca.transform(np.asarray(data))
        ax.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.25, lw=0, color="0.5")

    means = np.asarray(model.predict_mean(params, X_grid))
    covs = np.asarray(model.predict_cov(params, X_grid))
    u = pca.components_  # (2, N)
    proj_means = (means - pca.mean_[None, :]) @ u.T  # (T, 2)
    proj_covs = np.einsum("ij,tjk,lk->til", u, covs, u)  # (T, 2, 2)

    ax.plot(proj_means[:, 0], proj_means[:, 1], **mean_kwargs)
    ax.scatter(proj_means[:, 0], proj_means[:, 1], color="k", s=20, zorder=5)
    for m, c in zip(proj_means, proj_covs, strict=True):
        covariance_ellipse(m, c, ax=ax, n_std=n_std, **ellipse_kwargs)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    return ax
