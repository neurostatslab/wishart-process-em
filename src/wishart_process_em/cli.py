r"""Command-line interface: ``wishart-em``.

Three subcommands cover the common workflow:

* ``fit``      -- fit a model to data in an ``.npz`` (arrays ``Y`` and ``X``).
* ``evaluate`` -- held-out (marginal) log-likelihood of a saved fit on new data.
* ``sample``   -- draw synthetic data from a saved fit.

Run ``wishart-em <command> --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import sys

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

from .data import scale_conditions
from .diagnostics import heldout_loglike
from .fit import fit as fit_model
from .kernels import matern, squared_exponential
from .qmc import QMCLattice


def _load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        if "Y" not in d.files or "X" not in d.files:
            raise SystemExit(f"{path}: expected arrays 'Y' and 'X' in the npz")
        X = d["X"]
        return np.asarray(d["Y"], dtype=float), np.asarray(X, dtype=float)


def _kernel_spec(args) -> dict:
    if args.kernel == "squared_exponential":
        return {
            "type": "squared_exponential",
            "lengthscale": args.lengthscale,
            "variance": args.variance,
        }
    return {
        "type": "matern",
        "lengthscale": args.lengthscale,
        "nu": args.nu,
        "variance": args.variance,
    }


def _density(spec: dict):
    spec = dict(spec)
    t = spec.pop("type")
    return (squared_exponential if t == "squared_exponential" else matern)(**spec)


def cmd_fit(args) -> None:
    from .basis import fourier_basis
    from .model import WishartProcessModel
    from .serialize import save_fit

    Y, X = _load_xy(args.data)
    if args.transform == "sqrt":
        Y = np.sqrt(Y)
    elif args.transform == "log1p":
        Y = np.log1p(Y)
    if X.ndim == 1:
        X = X[:, None]
    if args.scale_conditions:
        X, _ = scale_conditions(X)

    num_dims = X.shape[1] if args.num_dims is None else args.num_dims
    spec = _kernel_spec(args)
    basis = fourier_basis(args.max_freq, num_dims)
    model = WishartProcessModel(
        basis,
        num_neurons=Y.shape[1],
        rank=args.rank,
        likelihood=args.likelihood,
        use_diagonal=args.use_diagonal,
        use_scale=args.use_scale,
        spectral_density=_density(spec),
    )
    print(f"Fitting {model} on {Y.shape[0]} trials ...", file=sys.stderr)
    result = fit_model(
        model,
        jnp.asarray(Y),
        jnp.asarray(X),
        num_steps=args.steps,
        learning_rate=args.lr,
        seed=args.seed,
        num_qmc_points=args.qmc_points,
        progress=not args.no_progress,
    )
    save_fit(args.output, model, result.params, spec)
    print(
        f"final objective = {result.loss_history[-1]:.2f}  ->  saved to {args.output}",
        file=sys.stderr,
    )


def cmd_evaluate(args) -> None:
    from .serialize import load_fit

    model, params, _ = load_fit(args.fit)
    Y, X = _load_xy(args.data)
    lattice = None
    key = None
    if not model.likelihood.conjugate_gaussian:
        lattice = QMCLattice(args.qmc_points, model.latent_dim, seed=args.seed)
        key = jxr.PRNGKey(args.seed)
    total = heldout_loglike(model, params, jnp.asarray(Y), jnp.asarray(X),
                            key=key, lattice=lattice, method=args.method)
    total = float(total)
    print(f"held-out log-likelihood (total)     = {total:.3f}")
    print(f"held-out log-likelihood (per trial) = {total / Y.shape[0]:.4f}")


def cmd_sample(args) -> None:
    from .serialize import load_fit

    model, params, _ = load_fit(args.fit)
    if args.conditions is not None:
        with np.load(args.conditions, allow_pickle=False) as d:
            X = np.asarray(d["X"], dtype=float)
    else:
        X = np.linspace(0.0, 1.0, args.trials)[:, None]
    Y, eta = model.sample(jxr.PRNGKey(args.seed), params, jnp.asarray(X))
    np.savez(args.output, Y=np.asarray(Y), X=np.asarray(X), eta=np.asarray(eta))
    print(f"sampled {Y.shape[0]} x {Y.shape[1]} responses -> {args.output}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wishart-em", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    # ---- fit --------------------------------------------------------------
    f = sub.add_parser("fit", help="fit a model to data (.npz with Y, X)")
    f.add_argument("data", help="input .npz with arrays Y (T,N) and X (T,D)")
    f.add_argument("-o", "--output", default="fit.npz", help="output .npz path")
    f.add_argument("--likelihood", default="gaussian",
                   choices=["gaussian", "poisson", "negative_binomial"])
    f.add_argument("--rank", type=int, default=2, help="covariance factor rank P")
    f.add_argument("--kernel", default="squared_exponential",
                   choices=["squared_exponential", "matern"])
    f.add_argument("--lengthscale", type=float, default=0.2)
    f.add_argument("--variance", type=float, default=1.0)
    f.add_argument("--nu", type=float, default=1.5, help="Matern smoothness")
    f.add_argument("--max-freq", type=int, default=8)
    f.add_argument("--num-dims", type=int, default=None,
                   help="condition dimensionality (default: infer from X)")
    f.add_argument("--use-diagonal", action="store_true")
    f.add_argument("--use-scale", action="store_true")
    f.add_argument("--steps", type=int, default=1000)
    f.add_argument("--lr", type=float, default=0.1)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--qmc-points", type=int, default=151)
    f.add_argument("--scale-conditions", action="store_true",
                   help="min-max scale X into [0,1] before fitting")
    f.add_argument("--transform", default=None, choices=["sqrt", "log1p"],
                   help="variance-stabilising transform of Y")
    f.add_argument("--no-progress", action="store_true")
    f.set_defaults(func=cmd_fit)

    # ---- evaluate ---------------------------------------------------------
    e = sub.add_parser("evaluate", help="held-out log-likelihood of a saved fit")
    e.add_argument("fit", help="saved fit .npz (from `wishart-em fit`)")
    e.add_argument("data", help="evaluation data .npz with Y, X")
    e.add_argument("--method", default="auto", choices=["auto", "qmc", "laplace"])
    e.add_argument("--qmc-points", type=int, default=211)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_evaluate)

    # ---- sample -----------------------------------------------------------
    s = sub.add_parser("sample", help="sample synthetic data from a saved fit")
    s.add_argument("fit", help="saved fit .npz")
    s.add_argument("-o", "--output", default="samples.npz")
    s.add_argument("--trials", type=int, default=500,
                   help="number of trials (uses a linspace grid over [0,1])")
    s.add_argument("--conditions", default=None,
                   help="optional .npz providing X to sample at")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)

    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``wishart-em`` console script."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
