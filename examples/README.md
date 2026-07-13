# Examples

Runnable scripts demonstrating the package. Each is self-contained and uses
synthetic data, so no downloads are required.

| File | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | Fit a Gaussian Wishart process over a 1-D periodic condition; compare held-out log-likelihood to classical baselines. |
| [`poisson_counts.py`](poisson_counts.py) | Spike-count (Poisson) model with QMC + Laplace latent integration; identifiability of noise covariance under over-dispersion. |
| [`baselines.py`](baselines.py) | Lightweight classical covariance estimators (empirical, grand, Ledoit-Wolf, weighted-average) used for comparison. Imported by the examples above; not part of the installed package. |

Run any of them with, e.g.:

```bash
python examples/quickstart.py
```

## Using your own / real data

The model expects two arrays:

- `Y` of shape `(T, N)` — responses on each of `T` trials for `N` neurons;
- `X` of shape `(T, D)` — the condition of each trial in a `D`-dimensional
  parameter space.

Scale conditions into `[0, 1]` first (periodic dimensions by their period):

```python
from wishart_process_em import scale_conditions, NeuralDataset
Xs, scaler = scale_conditions(X_raw, periods={0: 360.0})   # dim 0 is degrees
data = NeuralDataset(Y, Xs)
```

For raw spike counts you can either use `likelihood="poisson"` /
`"negative_binomial"` directly, or variance-stabilise and use the Gaussian
model:

```python
data = NeuralDataset.from_arrays(counts, X_raw, periods={0: 360.0}, transform="sqrt")
```

### Allen Brain Observatory (Neuropixels)

To reproduce the drifting-gratings analysis on real data, install the AllenSDK
(`pip install allensdk`) and extract, for one session, a spike-count matrix in a
post-stimulus window together with the `(orientation, temporal_frequency)` of
each trial. Build a 2-D condition space, pass `periods={0: 360.0}` (orientation
is periodic) to `scale_conditions`, and use `NeuralDataset.holdout_conditions`
to test covariance prediction at entirely unseen conditions.
