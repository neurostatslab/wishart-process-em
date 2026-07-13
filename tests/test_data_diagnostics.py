"""Tests for data utilities and diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

from wishart_process_em import (
    NeuralDataset,
    covariance_operator_norm_error,
    fisher_information,
    group_by_condition,
    heldout_loglike,
    qda_accuracy,
    scale_conditions,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _toy_dataset(seed=0):
    rng = np.random.default_rng(seed)
    C, K, N = 8, 15, 5
    X = np.repeat(np.linspace(0, 1, C, endpoint=False), K)
    Y = rng.normal(size=(C * K, N))
    return NeuralDataset(Y, X)


def test_dataset_basic_props():
    ds = _toy_dataset()
    assert ds.num_trials == 8 * 15
    assert ds.num_neurons == 5
    assert ds.condition_dim == 1


def test_train_test_split_global_partitions():
    ds = _toy_dataset()
    tr, te = ds.train_test_split(0.25, seed=1, stratify=False)
    assert tr.num_trials + te.num_trials == ds.num_trials
    assert te.num_trials == round(0.25 * ds.num_trials)


def test_train_test_split_stratified_balances_conditions():
    ds = _toy_dataset()  # 8 conditions x 15 trials
    tr, te = ds.train_test_split(0.25, seed=1)  # stratify=True by default
    assert tr.num_trials + te.num_trials == ds.num_trials
    # every condition contributes round(0.25 * 15) = 4 test trials
    _, test_groups = group_by_condition(te.Y, te.X)
    assert all(g.shape[0] == round(0.25 * 15) for g in test_groups)
    # and every condition is present in training
    assert len(group_by_condition(tr.Y, tr.X)[0]) == 8


def test_holdout_conditions_are_disjoint():
    ds = _toy_dataset()
    tr, held = ds.holdout_conditions(0.25, seed=2)
    tr_conditions = set(map(tuple, tr.X.tolist()))
    held_conditions = set(map(tuple, held.X.tolist()))
    assert tr_conditions.isdisjoint(held_conditions)


def test_group_by_condition():
    ds = _toy_dataset()
    uniq, groups = group_by_condition(ds.Y, ds.X)
    assert len(uniq) == 8
    assert all(g.shape[0] == 15 for g in groups)


def test_save_load_roundtrip(tmp_path):
    ds = _toy_dataset()
    p = tmp_path / "ds.npz"
    ds.save(str(p))
    loaded = NeuralDataset.load(str(p))
    np.testing.assert_allclose(loaded.Y, ds.Y)
    np.testing.assert_allclose(loaded.X, ds.X)


def test_scale_conditions_unit_box():
    X = np.array([[10.0, -5.0], [20.0, 5.0], [15.0, 0.0]])
    Xs, scaler = scale_conditions(X)
    assert Xs.min() >= 0.0 and Xs.max() <= 1.0
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X, atol=1e-8)


def test_scale_conditions_periodic():
    X = np.array([0.0, 90.0, 180.0, 270.0])[:, None]
    Xs, _ = scale_conditions(X, periods={0: 360.0})
    np.testing.assert_allclose(Xs[:, 0], [0.0, 0.25, 0.5, 0.75], atol=1e-8)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_heldout_loglike_finite(gaussian_model, gaussian_data):
    Y, X, params = gaussian_data
    val = heldout_loglike(gaussian_model, params, Y, X, per_trial=True)
    assert jnp.isfinite(val)


def test_fisher_information_symmetric_psd(gaussian_model):
    params = gaussian_model.init_params(jxr.PRNGKey(0))
    fi = fisher_information(gaussian_model, params, jnp.array([0.3]))
    assert fi.shape == (1, 1)
    assert float(fi[0, 0]) >= 0


def test_fisher_information_multidim():
    from wishart_process_em import (
        WishartProcessModel,
        fourier_basis,
        squared_exponential,
    )

    b = fourier_basis(4, num_dims=2)
    model = WishartProcessModel(
        b, num_neurons=4, rank=2, spectral_density=squared_exponential(0.3)
    )
    params = model.init_params(jxr.PRNGKey(0))
    fi = fisher_information(model, params, jnp.array([0.3, 0.6]))
    assert fi.shape == (2, 2)
    np.testing.assert_allclose(fi, fi.T, atol=1e-8)


def test_qda_accuracy_separable():
    # Two well-separated classes -> perfect classification.
    means = np.array([[-5.0, 0.0], [5.0, 0.0]])
    covs = np.stack([np.eye(2), np.eye(2)])
    rng = np.random.default_rng(0)
    Y = np.concatenate([rng.normal(means[0], 1, (20, 2)), rng.normal(means[1], 1, (20, 2))])
    labels = np.array([0] * 20 + [1] * 20)
    assert qda_accuracy(means, covs, Y, labels) > 0.95


def test_operator_norm_error():
    a = np.stack([np.eye(3), 2 * np.eye(3)])
    b = np.stack([np.eye(3), np.eye(3)])
    err = covariance_operator_norm_error(a, b)
    np.testing.assert_allclose(err, [0.0, 1.0], atol=1e-8)
