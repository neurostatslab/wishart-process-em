r"""Data containers and preprocessing for neural noise-covariance analysis.

A dataset is a collection of trials, each a neural response vector ``y`` at a
condition ``x`` (a point in a smoothly-parameterised condition space).  The
:class:`NeuralDataset` container standardises this layout and provides the
utilities most analyses need: scaling conditions into the unit box expected by
the Fourier basis, splitting trials into train / validation / test sets,
holding out whole conditions (to test interpolation), grouping trials by
condition (for the empirical baselines), and ``.npz`` serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NeuralDataset", "ConditionScaler", "scale_conditions", "group_by_condition"]


@dataclass
class ConditionScaler:
    """Affine map taking raw conditions into ``[0, 1]`` per dimension.

    ``periodic`` dimensions are scaled by their period so that wrap-around is
    respected by the periodic Fourier basis; non-periodic dimensions are min-max
    scaled with a margin so the (implicitly periodic) basis does not alias the
    endpoints together.
    """

    lower: np.ndarray
    span: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (X - self.lower[None, :]) / self.span[None, :]

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        return Xs * self.span[None, :] + self.lower[None, :]


def scale_conditions(
    X: np.ndarray,
    periods: dict[int, float] | None = None,
    margin: float = 0.05,
) -> tuple[np.ndarray, ConditionScaler]:
    """Scale raw conditions into ``[0, 1]^D`` for the Fourier basis.

    Parameters
    ----------
    X : array, shape (T,) or (T, D)
        Raw condition values.
    periods : dict[int, float], optional
        Maps a dimension index to its period (e.g. ``{0: 360.0}`` for degrees,
        ``{0: 2*pi}`` for radians).  Periodic dimensions are scaled by their
        period and wrapped into ``[0, 1)``.
    margin : float, optional
        Fractional padding for non-periodic dimensions, keeping data away from
        the wrap-around boundary of the periodic basis.

    Returns
    -------
    (Xs, scaler) : (np.ndarray, ConditionScaler)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    d = X.shape[1]
    periods = periods or {}

    lower = np.empty(d)
    span = np.empty(d)
    for j in range(d):
        if j in periods:
            lower[j] = 0.0
            span[j] = periods[j]
        else:
            lo, hi = X[:, j].min(), X[:, j].max()
            pad = margin * (hi - lo if hi > lo else 1.0)
            lower[j] = lo - pad
            span[j] = (hi - lo) + 2 * pad
    scaler = ConditionScaler(lower=lower, span=span)
    Xs = scaler.transform(X) % 1.0
    return Xs, scaler


def group_by_condition(
    Y: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group trials by their (exact) condition value.

    Returns ``(unique_conditions, groups)`` where ``groups[c]`` is the array of
    trials (shape ``(K_c, N)``) at ``unique_conditions[c]``.
    """
    Y = np.asarray(Y)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    keys = X.reshape(X.shape[0], -1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    groups = [Y[inv == c] for c in range(len(uniq))]
    return uniq, groups


@dataclass
class NeuralDataset:
    """A trial-by-neuron dataset with associated conditions.

    Attributes
    ----------
    Y : np.ndarray
        Responses, shape ``(T, N)``.
    X : np.ndarray
        Conditions, shape ``(T, D)`` (a 1-D input is promoted to ``D = 1``).
    scaler : ConditionScaler or None
        The scaler used to map raw conditions into ``[0, 1]``, if any.
    """

    Y: np.ndarray
    X: np.ndarray
    scaler: ConditionScaler | None = None

    def __post_init__(self) -> None:
        self.Y = np.asarray(self.Y)
        X = np.asarray(self.X, dtype=float)
        self.X = X[:, None] if X.ndim == 1 else X
        if self.Y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"Y and X must have the same number of trials; got "
                f"{self.Y.shape[0]} and {self.X.shape[0]}"
            )

    # -- basic properties -------------------------------------------------
    @property
    def num_trials(self) -> int:
        return self.Y.shape[0]

    @property
    def num_neurons(self) -> int:
        return self.Y.shape[1]

    @property
    def condition_dim(self) -> int:
        return self.X.shape[1]

    def __len__(self) -> int:
        return self.num_trials

    def subset(self, idx: np.ndarray) -> NeuralDataset:
        """Return the dataset restricted to trial indices ``idx``."""
        return NeuralDataset(self.Y[idx], self.X[idx], scaler=self.scaler)

    # -- construction / preprocessing ------------------------------------
    @classmethod
    def from_arrays(
        cls,
        Y: np.ndarray,
        X: np.ndarray,
        periods: dict[int, float] | None = None,
        scale: bool = True,
        transform: str | None = None,
    ) -> NeuralDataset:
        """Build a dataset, optionally scaling conditions and transforming ``Y``.

        Parameters
        ----------
        transform : {"sqrt", "log1p", None}, optional
            Variance-stabilising transform of the responses, commonly applied to
            spike counts / calcium traces before Gaussian modelling.
        """
        Y = np.asarray(Y, dtype=float)
        if transform == "sqrt":
            Y = np.sqrt(Y)
        elif transform == "log1p":
            Y = np.log1p(Y)
        elif transform is not None:
            raise ValueError(f"unknown transform {transform!r}")

        scaler = None
        if scale:
            X, scaler = scale_conditions(X, periods=periods)
        return cls(Y=Y, X=np.asarray(X, dtype=float), scaler=scaler)

    # -- splitting --------------------------------------------------------
    def train_test_split(
        self, test_frac: float = 0.2, seed: int = 0
    ) -> tuple[NeuralDataset, NeuralDataset]:
        """Randomly split *trials* into train and test sets."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.num_trials)
        n_test = int(round(test_frac * self.num_trials))
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        return self.subset(train_idx), self.subset(test_idx)

    def holdout_conditions(
        self, frac: float = 0.2, seed: int = 0
    ) -> tuple[NeuralDataset, NeuralDataset]:
        """Hold out *entire conditions* (for testing interpolation).

        Returns ``(train, heldout)`` where every trial of a held-out condition is
        placed in ``heldout`` and no held-out condition appears in ``train``.
        """
        rng = np.random.default_rng(seed)
        uniq, _ = group_by_condition(self.Y, self.X)
        n_hold = max(1, int(round(frac * len(uniq))))
        hold_conditions = uniq[rng.permutation(len(uniq))[:n_hold]]
        keys = self.X.reshape(self.num_trials, -1)
        is_held = np.any(np.all(keys[:, None, :] == hold_conditions[None], axis=2), axis=1)
        return self.subset(~is_held), self.subset(is_held)

    def grouped(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Group trials by condition (see :func:`group_by_condition`)."""
        return group_by_condition(self.Y, self.X)

    # -- serialisation ----------------------------------------------------
    def save(self, path: str) -> None:
        """Save to a ``.npz`` with arrays ``Y`` and ``X`` (and scaler if set)."""
        arrays = {"Y": self.Y, "X": self.X}
        if self.scaler is not None:
            arrays["scaler_lower"] = self.scaler.lower
            arrays["scaler_span"] = self.scaler.span
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> NeuralDataset:
        """Load a dataset saved with :meth:`save` (or any npz with ``Y``, ``X``)."""
        with np.load(path) as d:
            scaler = None
            if "scaler_lower" in d:
                scaler = ConditionScaler(d["scaler_lower"], d["scaler_span"])
            return cls(Y=d["Y"], X=d["X"], scaler=scaler)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NeuralDataset(trials={self.num_trials}, neurons={self.num_neurons}, "
            f"condition_dim={self.condition_dim})"
        )
