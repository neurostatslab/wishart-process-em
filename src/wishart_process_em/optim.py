r"""A small, JAX-friendly Newton optimiser for Laplace approximations.

The Laplace approximation to a per-trial latent posterior requires finding the
posterior mode and the curvature (Hessian) there.  This module provides a
damped Newton method with backtracking (Armijo) line search that is fully
traceable, so it can be ``jit``-compiled and ``vmap``-ed over trials.

Unlike a generic optimiser, the routine also returns the Cholesky factor of the
Hessian at the solution, which is exactly what the Laplace proposal needs: the
posterior covariance is :math:`H^{-1}`, so a sample is
:math:`m + L^{-\top} \varepsilon` with :math:`\varepsilon \sim \mathcal N(0, I)`
and :math:`L L^\top = H`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["NewtonResult", "newton_minimize"]


class NewtonResult(NamedTuple):
    """Result of :func:`newton_minimize`.

    Attributes
    ----------
    x : jnp.ndarray
        The (approximate) minimiser / posterior mode.
    grad_norm : jnp.ndarray
        Euclidean norm of the gradient at ``x``.
    n_iter : jnp.ndarray
        Number of Newton iterations taken.
    hess_chol : jnp.ndarray
        Lower-triangular Cholesky factor ``L`` of the (damped) Hessian at ``x``,
        with ``L @ L.T`` approximately equal to the Hessian.
    converged : jnp.ndarray
        Boolean flag; ``True`` if the gradient-norm tolerance was met.
    """

    x: jnp.ndarray
    grad_norm: jnp.ndarray
    n_iter: jnp.ndarray
    hess_chol: jnp.ndarray
    converged: jnp.ndarray


def _line_search(fun, x, fx, g, p, alpha, beta, min_step):
    """Backtracking line search returning a step size satisfying Armijo."""
    gp = jnp.dot(g, p)

    def cond(state):
        t, done = state
        return jnp.logical_and(jnp.logical_not(done), t > min_step)

    def body(state):
        t, _ = state
        sufficient = fun(x + t * p) <= fx + alpha * t * gp
        return (jnp.where(sufficient, t, t * beta), sufficient)

    t, _ = jax.lax.while_loop(cond, body, (jnp.asarray(1.0), jnp.asarray(False)))
    return t


def newton_minimize(
    fun: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
    damping: float = 1e-6,
    ls_alpha: float = 0.3,
    ls_beta: float = 0.8,
    ls_min_step: float = 1e-12,
) -> NewtonResult:
    r"""Minimise a smooth scalar function with damped Newton + line search.

    Parameters
    ----------
    fun : Callable
        Scalar objective ``f : R^d -> R`` (traceable by JAX).
    x0 : jnp.ndarray
        Initial point of shape ``(d,)``.
    max_iter : int, optional
        Maximum Newton iterations.
    tol : float, optional
        Stop when ``||grad|| < sqrt(d) * tol``.
    damping : float, optional
        Levenberg-Marquardt damping added to the Hessian diagonal for positive
        definiteness (on top of any curvature from the prior).
    ls_alpha, ls_beta, ls_min_step : float, optional
        Armijo line-search parameters.

    Returns
    -------
    NewtonResult
    """
    grad_fun = jax.grad(fun)
    hess_fun = jax.hessian(fun)
    d = x0.shape[0]
    eye = jnp.eye(d)
    tol_scaled = jnp.sqrt(d) * tol

    def cond(state):
        x, g, i, _ = state
        return jnp.logical_and(i < max_iter, jnp.linalg.norm(g) >= tol_scaled)

    def body(state):
        x, g, i, _ = state
        fx = fun(x)
        H = hess_fun(x) + damping * eye
        L = jnp.linalg.cholesky(H)
        p = jax.scipy.linalg.cho_solve((L, True), -g)
        t = _line_search(fun, x, fx, g, p, ls_alpha, ls_beta, ls_min_step)
        x_new = x + t * p
        return (x_new, grad_fun(x_new), i + 1, L)

    g0 = grad_fun(x0)
    L0 = jnp.linalg.cholesky(hess_fun(x0) + damping * eye)
    x, g, n_iter, L = jax.lax.while_loop(cond, body, (x0, g0, jnp.asarray(0), L0))

    grad_norm = jnp.linalg.norm(g)
    return NewtonResult(
        x=x,
        grad_norm=grad_norm,
        n_iter=n_iter,
        hess_chol=L,
        converged=grad_norm < tol_scaled,
    )
