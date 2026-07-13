# Command-line interface

Installing the package provides the `wishart-em` command-line tool for fitting,
evaluating, and sampling Wishart process models without writing Python. It wraps
the same [`fit`](api.md#wishart_process_em.fit) driver and model API used from
code.

!!! note "Illustrative flags"
    The commands and flags below are illustrative. The authoritative, always
    up-to-date list of subcommands and options is:

    ```bash
    wishart-em --help
    wishart-em fit --help          # per-subcommand help
    ```

## Data format

All subcommands read and write NumPy `.npz` archives containing two arrays:

| Array | Shape   | Meaning |
|-------|---------|---------|
| `Y`   | `(T, N)` | Trial-by-neuron observations (real-valued for Gaussian, counts for Poisson / negative binomial). |
| `X`   | `(T, D)` | Per-trial condition values, scaled into `[0, 1]^D`. A 1-D condition may be stored as `(T,)`. |

Create one from Python with `numpy.savez`:

```python
import numpy as np
np.savez("data.npz", Y=Y, X=X)
```

## `fit` — fit a model to data

Fit a Wishart process to a dataset and save the resulting parameters.

```bash
wishart-em fit data.npz --likelihood poisson --rank 3 --steps 1000 -o fit.npz
```

Common options:

| Flag | Meaning |
|------|---------|
| `--likelihood {gaussian,poisson,negative_binomial}` | Observation model. Gaussian uses the exact (conjugate) marginal likelihood; count models use QMC. |
| `--rank P` | Rank $P$ of the low-rank covariance factor $U$. |
| `--steps N` | Number of optimisation steps. |
| `-o, --output fit.npz` | Where to write the fitted model / parameters. |

The saved `fit.npz` bundles the optimised parameters together with the model
configuration (basis, likelihood, rank, …) so the `evaluate` and `sample`
subcommands can reconstruct the model.

## `evaluate` — held-out log-likelihood

Score a saved fit on a held-out dataset, reporting the marginal
log-likelihood (using the more accurate Laplace estimator for count models):

```bash
wishart-em evaluate fit.npz heldout.npz
```

This is the primary quantitative check that the model generalises across
conditions rather than overfitting the training trials.

## `sample` — generate synthetic data

Draw synthetic trials from a fitted model, e.g. for posterior predictive checks
or downstream simulation:

```bash
wishart-em sample fit.npz --trials 500 -o synthetic.npz
```

The output `synthetic.npz` again holds `Y` (the sampled observations) and `X`
(the conditions they were sampled at), matching the input data format.

## Typical workflow

```bash
# 1. Fit on the training split.
wishart-em fit train.npz --likelihood negative_binomial --rank 3 --steps 1500 -o fit.npz

# 2. Check held-out generalisation.
wishart-em evaluate fit.npz test.npz

# 3. Generate synthetic data for a predictive check.
wishart-em sample fit.npz --trials 1000 -o synthetic.npz
```

See `wishart-em --help` for the complete and current option list.
