"""
Shared fixtures and helpers for DataLoader-related tests.
Adapted from Dylan Kotliar's original cNMF; torch-based version by Alexandra Mo.
"""

import pytest
import numpy as np
import pandas as pd
import os
import torch


# Small synthetic dimensions for fast tests
N_SAMPLES = 60
N_FEATURES = 40
N_COMPONENTS = 3
SEED = 42

# GPU device selection — used by all tests
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_nmf_problem(n_samples=N_SAMPLES, n_features=N_FEATURES,
                     n_components=N_COMPONENTS, seed=SEED):
    """
    Generate a synthetic NMF problem: X ≈ H @ W where W is (k, features)
    and H is (samples, k).  Returns (X, W_true, H_true) as numpy arrays.
    """
    rng = np.random.default_rng(seed)
    W_true = rng.exponential(1.0, size=(n_components, n_features)).astype(np.float32)
    H_true = rng.exponential(1.0, size=(n_samples, n_components)).astype(np.float32)
    X = (H_true @ W_true).astype(np.float32)
    # Add small noise so it's not trivially exact
    X += rng.exponential(0.05, size=X.shape).astype(np.float32)
    return X, W_true, H_true


@pytest.fixture
def nmf_problem():
    """Return (X, W_true, H_true) numpy arrays."""
    return make_nmf_problem()


@pytest.fixture
def nmf_problem_df():
    """Return (X, W, H_init) as pandas DataFrames with labelled axes."""
    X, W, H = make_nmf_problem()
    cell_names = [f"cell_{i}" for i in range(X.shape[0])]
    gene_names = [f"gene_{j}" for j in range(X.shape[1])]
    comp_names = [f"prog_{k}" for k in range(W.shape[0])]

    X_df = pd.DataFrame(X, index=cell_names, columns=gene_names)
    W_df = pd.DataFrame(W, index=comp_names, columns=gene_names)
    H_df = pd.DataFrame(H, index=cell_names, columns=comp_names)
    return X_df, W_df, H_df


@pytest.fixture
def device():
    """Return 'cuda' if available, else 'cpu'."""
    return DEVICE


@pytest.fixture
def results_base():
    """Return the base results directory; create it if needed."""
    base = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(base, exist_ok=True)
    return base
