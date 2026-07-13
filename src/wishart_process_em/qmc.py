r"""Quasi-Monte-Carlo lattices for latent-variable integration.

Fitting the non-Gaussian (Poisson / negative-binomial) models requires
integrating over per-trial latent variables :math:`v \sim \mathcal N(0, I_Q)`.
We estimate these integrals with a *randomised rank-1 lattice rule* (a Korobov
lattice), which for smooth integrands converges markedly faster than plain
Monte-Carlo while remaining an unbiased estimator after randomisation.

The pipeline is:

1. Build a deterministic Korobov lattice on the unit cube
   :math:`\{ (k a^0, k a^1, \dots) / n \bmod 1 : k = 0, \dots, n-1 \}`.
2. At integration time, apply a random shift (mod 1) and a random rotation for
   unbiasedness / variance reduction across dimensions.
3. Map the shifted points to standard-normal samples with the inverse Gaussian
   CDF.

The generator ``a`` is chosen to (approximately) minimise the star discrepancy
of the lattice in the target dimension.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jxr
import numpy as np

__all__ = ["korobov_points", "star_discrepancy", "find_optimal_generator", "QMCLattice"]


def korobov_points(n: int, a: int, d: int) -> np.ndarray:
    """Generate ``n`` rank-1 Korobov lattice points in ``d`` dimensions.

    Returns an ``(n, d)`` array of points in ``[0, 1)`` with coordinates
    ``(k * a**i mod n) / n``.
    """
    powers = np.array([pow(int(a), i, int(n)) for i in range(d)], dtype=np.int64)
    k = np.arange(n, dtype=np.int64)[:, None]
    return (k * powers[None, :] % n) / n


def star_discrepancy(
    n: int, a: int, d: int, num_samples: int = 10_000, rng: np.random.Generator | None = None
) -> float:
    """Monte-Carlo estimate of the star discrepancy of a Korobov lattice.

    Lower is better.  A random shift is applied so the estimate reflects the
    randomised rule actually used at integration time.
    """
    rng = np.random.default_rng() if rng is None else rng
    points = (korobov_points(n, a, d) + rng.random((1, d))) % 1
    boxes = rng.random((num_samples, d))
    # Fraction of points inside each anchored box vs. its volume.
    inside = np.all(points[None, :, :] <= boxes[:, None, :], axis=2)
    empirical = inside.mean(axis=1)
    theoretical = np.prod(boxes, axis=1)
    return float(np.max(np.abs(empirical - theoretical)))


def find_optimal_generator(
    n: int, d: int, samples: int = 4_000, rng: np.random.Generator | None = None
) -> tuple[int, float]:
    """Search ``a in {1, ..., n-1}`` for the lowest-discrepancy lattice.

    ``n`` should be prime for the rank-1 lattice construction to have good
    coverage.  Returns ``(best_a, best_discrepancy)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    best_a, best_disc = 1, np.inf
    for a in range(1, n):
        disc = star_discrepancy(n, a, d, num_samples=samples, rng=rng)
        if disc < best_disc:
            best_a, best_disc = a, disc
    return best_a, best_disc


class QMCLattice:
    """A randomised Korobov lattice for integrating standard-normal latents.

    Parameters
    ----------
    num_points : int
        Number of lattice points ``n`` (ideally prime).
    dim : int
        Latent dimension ``Q``.
    generator : int, optional
        Korobov generator ``a``.  If ``None``, it is selected by
        :func:`find_optimal_generator`.
    seed : int, optional
        Seed for the (host-side) generator search, for reproducibility.
    """

    def __init__(
        self,
        num_points: int,
        dim: int,
        generator: int | None = None,
        seed: int = 0,
    ) -> None:
        self.num_points = int(num_points)
        self.dim = int(dim)
        rng = np.random.default_rng(seed)
        if generator is None:
            generator, disc = find_optimal_generator(self.num_points, self.dim, rng=rng)
            self.discrepancy = disc
        else:
            self.discrepancy = star_discrepancy(
                self.num_points, generator, self.dim, rng=rng
            )
        self.generator = int(generator)
        self.points = jnp.asarray(korobov_points(self.num_points, self.generator, self.dim))

    def gaussian_points(self, key: jax.Array) -> jnp.ndarray:
        """Randomised QMC samples from :math:`\\mathcal N(0, I_{\\text{dim}})`.

        Applies a random shift (mod 1) and a random orthogonal rotation, then
        maps through the inverse Gaussian CDF.  Returns an ``(num_points, dim)``
        array whose empirical distribution approximates a standard normal but
        with much lower integration error than i.i.d. sampling.
        """
        k_shift, k_rot = jxr.split(key)
        shift = jxr.uniform(k_shift, shape=(1, self.dim))
        rotation = jxr.orthogonal(k_rot, self.dim)
        shifted = (self.points + shift) % 1.0
        # ndtri = inverse standard-normal CDF.
        z = jax.scipy.special.ndtri(shifted)
        return z @ rotation

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"QMCLattice(num_points={self.num_points}, dim={self.dim}, "
            f"generator={self.generator})"
        )
