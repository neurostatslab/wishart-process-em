r"""Save and load fitted models to a single ``.npz`` file.

A saved fit bundles (a) a JSON description of the model architecture and kernel,
so the model can be reconstructed, and (b) the fitted parameter arrays.  No
pickling is used, so files are safe to share.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from .basis import fourier_basis
from .covariance import WPParams
from .kernels import matern, squared_exponential
from .model import WishartProcessModel

__all__ = ["model_config", "build_model", "save_fit", "load_fit"]

_KERNELS = {"squared_exponential": squared_exponential, "matern": matern}


def _basis_spec(basis) -> dict:
    """Recover the ``fourier_basis`` arguments from a nemos Fourier basis."""
    freqs = np.asarray(basis.frequencies)  # (num_dims, max_freq)
    num_dims = int(freqs.shape[0])
    raw_bounds = basis.bounds or (0.0, 1.0)
    per_dim = raw_bounds if num_dims == 1 else raw_bounds[0]
    return {
        "max_freq": int(freqs.shape[1]),
        "num_dims": num_dims,
        "bounds": [float(per_dim[0]), float(per_dim[1])],
    }


def model_config(model: WishartProcessModel, kernel: dict | None) -> dict:
    """Build a JSON-serialisable config describing ``model`` and its ``kernel``.

    ``kernel`` is a dict like ``{"type": "squared_exponential", "lengthscale":
    0.2, "variance": 1.0}`` recording the GP spectral density (the closure itself
    cannot be serialised), or ``None`` for a model with no spectral scaling.
    """
    return {
        "kernel": kernel,
        "basis": _basis_spec(model.basis),
        "model": {
            "num_neurons": model.num_neurons,
            "rank": model.rank,
            "likelihood": type(model.likelihood).__name__.lower(),
            "use_diagonal": model.use_diagonal,
            "use_scale": model.use_scale,
            "jitter": model.jitter,
            "init_weight_scale": model.init_weight_scale,
            "prior_weight_scale": model.prior_weight_scale,
        },
    }


def build_model(config: dict) -> WishartProcessModel:
    """Reconstruct a :class:`WishartProcessModel` from a config dict."""
    density = None
    kspec = config.get("kernel")
    if kspec:
        kspec = dict(kspec)
        ktype = kspec.pop("type")
        if ktype not in _KERNELS:
            raise ValueError(f"unknown kernel type {ktype!r}")
        density = _KERNELS[ktype](**kspec)

    b = config["basis"]
    basis = fourier_basis(b["max_freq"], b["num_dims"], bounds=tuple(b["bounds"]))

    m = config["model"]
    lik = m["likelihood"]
    lik = "negative_binomial" if lik in ("negativebinomial", "negative_binomial") else lik
    return WishartProcessModel(
        basis,
        num_neurons=m["num_neurons"],
        rank=m["rank"],
        likelihood=lik,
        use_diagonal=m["use_diagonal"],
        use_scale=m["use_scale"],
        jitter=m["jitter"],
        init_weight_scale=m["init_weight_scale"],
        prior_weight_scale=m["prior_weight_scale"],
        spectral_density=density,
    )


def save_fit(path: str, model: WishartProcessModel, params: WPParams, kernel: dict) -> None:
    """Save ``model`` architecture + fitted ``params`` to ``path`` (``.npz``)."""
    config = model_config(model, kernel)
    arrays = {
        "_config": np.array(json.dumps(config)),
        "mean_w": np.asarray(params.mean_w),
        "mean_b": np.asarray(params.mean_b),
        "factor_w": np.asarray(params.factor_w),
        "log_latent_scale": np.asarray(params.log_latent_scale),
    }
    if params.diag_w is not None:
        arrays["diag_w"] = np.asarray(params.diag_w)
    if params.scale_raw is not None:
        arrays["scale_raw"] = np.asarray(params.scale_raw)
    if params.lik is not None:
        for k, v in params.lik.items():
            arrays[f"lik__{k}"] = np.asarray(v)
    np.savez(path, **arrays)


def load_fit(path: str) -> tuple[WishartProcessModel, WPParams, dict]:
    """Load a fit saved by :func:`save_fit`; returns ``(model, params, config)``."""
    with np.load(path, allow_pickle=False) as d:
        config = json.loads(str(d["_config"]))
        model = build_model(config)
        lik_keys = [k for k in d.files if k.startswith("lik__")]
        lik = {k[len("lik__") :]: jnp.asarray(d[k]) for k in lik_keys} or None
        params = WPParams(
            mean_w=jnp.asarray(d["mean_w"]),
            mean_b=jnp.asarray(d["mean_b"]),
            factor_w=jnp.asarray(d["factor_w"]),
            log_latent_scale=jnp.asarray(d["log_latent_scale"]),
            diag_w=jnp.asarray(d["diag_w"]) if "diag_w" in d.files else None,
            scale_raw=jnp.asarray(d["scale_raw"]) if "scale_raw" in d.files else None,
            lik=lik,
        )
    return model, params, config
