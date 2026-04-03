"""
Test fit_H_online_hals_DL — the DataLoader-based HALS solver.
Adapted from Dylan Kotliar's original cNMF; torch-based version by Alexandra Mo.

Results saved to: Test/results/test_fit_H_hals_dl/
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pytest

from torch_cnmf.cnmf import fit_H_online_hals_DL, fit_H_online_hals


RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "test_fit_H_hals_dl")


@pytest.fixture(autouse=True)
def setup_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #
def reconstruction_error(X, H, W):
    return float(np.sqrt(((X - H @ W) ** 2).sum()))


# ------------------------------------------------------------------ #
# 1. Basic output shape and non-negativity (numpy inputs)
# ------------------------------------------------------------------ #
def test_basic_shape_and_nonneg(nmf_problem, device):
    """fit_H_online_hals_DL should return (n_samples, n_components), all >= 0."""
    X, W, _ = nmf_problem
    H = fit_H_online_hals_DL(X, W, chunk_size=20, chunk_max_iter=50, device=device)

    assert H.shape == (X.shape[0], W.shape[0]), f"Shape mismatch: {H.shape}"
    assert np.all(H >= 0), "H contains negative values"

    with open(os.path.join(RESULT_DIR, "basic_shape_nonneg.json"), "w") as f:
        json.dump({
            "H_shape": list(H.shape),
            "min_val": float(H.min()),
            "max_val": float(H.max()),
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 2. Reconstruction error decreases vs random init
# ------------------------------------------------------------------ #
def test_reconstruction_improves(nmf_problem, device):
    """Fitted H should give lower reconstruction error than random H."""
    X, W, _ = nmf_problem
    rng = np.random.default_rng(99)
    H_rand = rng.exponential(1.0, size=(X.shape[0], W.shape[0])).astype(np.float32)

    H_fit = fit_H_online_hals_DL(X, W, H_init=H_rand.copy(),
                                  chunk_size=20, chunk_max_iter=100, device=device)

    err_rand = reconstruction_error(X, H_rand, W)
    err_fit = reconstruction_error(X, H_fit, W)

    assert err_fit < err_rand, (
        f"Fitted error ({err_fit:.4f}) should be less than random ({err_rand:.4f})"
    )

    with open(os.path.join(RESULT_DIR, "reconstruction_improves.json"), "w") as f:
        json.dump({
            "error_random": err_rand,
            "error_fitted": err_fit,
            "improvement_pct": (err_rand - err_fit) / err_rand * 100,
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 3. DL version matches non-DL HALS version
# ------------------------------------------------------------------ #
def test_dl_matches_plain_hals(nmf_problem, device):
    """fit_H_online_hals_DL should produce similar results to fit_H_online_hals."""
    X, W, _ = nmf_problem
    rng = np.random.default_rng(7)
    H_init = rng.exponential(1.0, size=(X.shape[0], W.shape[0])).astype(np.float32)

    common = dict(chunk_size=X.shape[0], chunk_max_iter=200, h_tol=1e-6)

    H_plain = fit_H_online_hals(X, W, H_init=H_init.copy(), device=device, **common)
    H_dl = fit_H_online_hals_DL(X, W, H_init=H_init.copy(), device=device, **common)

    err_plain = reconstruction_error(X, H_plain, W)
    err_dl = reconstruction_error(X, H_dl, W)

    # Both should reach similar reconstruction quality
    assert abs(err_plain - err_dl) / (err_plain + 1e-12) < 0.05, (
        f"Errors diverge: plain={err_plain:.6f}, DL={err_dl:.6f}"
    )

    with open(os.path.join(RESULT_DIR, "dl_vs_plain_hals.json"), "w") as f:
        json.dump({
            "error_plain_hals": err_plain,
            "error_dl_hals": err_dl,
            "relative_diff": abs(err_plain - err_dl) / (err_plain + 1e-12),
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 4. DataFrame inputs — labels preserved
# ------------------------------------------------------------------ #
def test_dataframe_inputs(nmf_problem_df, device):
    """When passed DataFrames, output should still be a valid numpy array."""
    X_df, W_df, H_df = nmf_problem_df
    H_fit = fit_H_online_hals_DL(X_df, W_df, H_init=H_df,
                                  chunk_size=20, chunk_max_iter=50, device=device)

    assert isinstance(H_fit, np.ndarray), "Output should be a numpy array"
    assert H_fit.shape == (len(X_df), len(W_df)), f"Shape mismatch: {H_fit.shape}"
    assert np.all(H_fit >= 0), "H contains negative values"

    with open(os.path.join(RESULT_DIR, "dataframe_inputs.json"), "w") as f:
        json.dump({
            "H_shape": list(H_fit.shape),
            "input_type": "DataFrame",
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 5. Sparse matrix input
# ------------------------------------------------------------------ #
def test_sparse_input(nmf_problem, device):
    """fit_H_online_hals_DL should handle scipy sparse X."""
    X, W, _ = nmf_problem
    X_sparse = sp.csr_matrix(X)

    H_sparse = fit_H_online_hals_DL(X_sparse, W, chunk_size=20, chunk_max_iter=50, device=device)
    H_dense = fit_H_online_hals_DL(X, W, chunk_size=20, chunk_max_iter=50, device=device)

    # Sparse and dense should give the same result (same random init via None)
    assert H_sparse.shape == H_dense.shape
    assert np.all(H_sparse >= 0)

    with open(os.path.join(RESULT_DIR, "sparse_input.json"), "w") as f:
        json.dump({
            "H_shape": list(H_sparse.shape),
            "error_sparse": reconstruction_error(X, H_sparse, W),
            "error_dense": reconstruction_error(X, H_dense, W),
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 6. Different chunk sizes all produce valid results
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("chunk_size", [10, 30, 60, 100])
def test_different_chunk_sizes(nmf_problem, chunk_size, device):
    """Different batch sizes for DataLoader should all produce valid results."""
    X, W, _ = nmf_problem
    rng = np.random.default_rng(0)
    H_init = rng.exponential(1.0, size=(X.shape[0], W.shape[0])).astype(np.float32)

    H_fit = fit_H_online_hals_DL(X, W, H_init=H_init.copy(),
                                  chunk_size=chunk_size, chunk_max_iter=100, device=device)

    assert H_fit.shape == (X.shape[0], W.shape[0])
    assert np.all(H_fit >= 0)
    err = reconstruction_error(X, H_fit, W)

    tag = f"chunk_{chunk_size}"
    with open(os.path.join(RESULT_DIR, f"chunk_size_{chunk_size}.json"), "w") as f:
        json.dump({
            "chunk_size": chunk_size,
            "H_shape": list(H_fit.shape),
            "reconstruction_error": err,
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 7. Regularization params accepted without error
# ------------------------------------------------------------------ #
def test_regularization(nmf_problem, device):
    """L1 and L2 regularization should run without error and stay non-negative."""
    X, W, _ = nmf_problem
    H_fit = fit_H_online_hals_DL(X, W, chunk_size=20, chunk_max_iter=50,
                                  l1_reg_H=0.1, l2_reg_H=0.1, device=device)

    assert H_fit.shape == (X.shape[0], W.shape[0])
    assert np.all(H_fit >= 0)
    err = reconstruction_error(X, H_fit, W)

    with open(os.path.join(RESULT_DIR, "regularization.json"), "w") as f:
        json.dump({
            "l1_reg_H": 0.1,
            "l2_reg_H": 0.1,
            "reconstruction_error": err,
            "H_sparsity": float((H_fit < 1e-8).mean()),
            "status": "PASSED",
        }, f)
