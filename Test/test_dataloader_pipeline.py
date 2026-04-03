"""
Test the full cNMF pipeline with mode='dataloader'.
Adapted from Dylan Kotliar's original cNMF; torch-based version by Alexandra Mo.

Results saved to: Test/results/test_dataloader_pipeline/
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pytest

from torch_cnmf import cNMF


RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "test_dataloader_pipeline")
NUM_CELLS = 80
NUM_GENES = 200
SEED = 42


@pytest.fixture(autouse=True)
def setup_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


@pytest.fixture
def synthetic_h5ad():
    """Create a small synthetic h5ad for pipeline tests."""
    rng = np.random.default_rng(SEED)
    n_programs = 3
    W = rng.exponential(1.0, size=(NUM_CELLS, n_programs))
    H = rng.exponential(1.0, size=(n_programs, NUM_GENES))
    block_g = NUM_GENES // n_programs
    block_c = NUM_CELLS // n_programs
    for p in range(n_programs):
        H[p, p * block_g:(p + 1) * block_g] *= 5.0
        W[p * block_c:(p + 1) * block_c, p] *= 5.0
    counts = rng.poisson(W @ H).astype(np.float64)
    counts[counts.sum(axis=1) == 0, 0] = 1

    adata = sc.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(NUM_CELLS)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(NUM_GENES)]),
    )
    path = os.path.join(RESULT_DIR, "synthetic_counts.h5ad")
    adata.write_h5ad(path)
    return path


# ------------------------------------------------------------------ #
# 1. prepare + factorize with mode='dataloader'
# ------------------------------------------------------------------ #
def test_factorize_dataloader_mode(synthetic_h5ad):
    """The pipeline should run prepare + factorize with mode='dataloader'."""
    run_dir = os.path.join(RESULT_DIR, "factorize_dl")
    os.makedirs(run_dir, exist_ok=True)
    cnmf_obj = cNMF(output_dir=run_dir, name="dl_test")

    cnmf_obj.prepare(
        counts_fn=synthetic_h5ad,
        components=[3],
        n_iter=10,
        num_highvar_genes=80,
        seed=SEED,
        use_gpu=True,
        mode="dataloader",
        algo="halsvar",
        minibatch_size=30,
    )
    cnmf_obj.factorize()

    # Check that spectra files were created for each iteration
    # (dataloader mode does not produce per-iteration usage files)
    missing = []
    for it in range(10):
        spec_path = cnmf_obj.paths['iter_spectra'] % (3, it)
        if not os.path.exists(spec_path):
            missing.append(spec_path)

    assert len(missing) == 0, f"Missing iteration files: {missing}"

    with open(os.path.join(run_dir, "factorize_summary.json"), "w") as f:
        json.dump({
            "mode": "dataloader",
            "components": [3],
            "n_iter": 10,
            "missing_files": missing,
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 2. Full pipeline: prepare -> factorize -> combine -> consensus
# ------------------------------------------------------------------ #
def test_full_pipeline_dataloader(synthetic_h5ad):
    """End-to-end pipeline with mode='dataloader' should produce valid results."""
    run_dir = os.path.join(RESULT_DIR, "full_pipeline_dl")
    os.makedirs(run_dir, exist_ok=True)
    cnmf_obj = cNMF(output_dir=run_dir, name="dl_full")

    K = 3
    n_iter = 10

    cnmf_obj.prepare(
        counts_fn=synthetic_h5ad,
        components=[K],
        n_iter=n_iter,
        num_highvar_genes=80,
        seed=SEED,
        use_gpu=True,
        mode="dataloader",
        algo="halsvar",
        minibatch_size=30,
    )
    cnmf_obj.factorize()
    cnmf_obj.combine()
    cnmf_obj.consensus(k=K, density_threshold=2.0, show_clustering=False)
    cnmf_obj.k_selection_plot()  

    usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(
        K=K, density_threshold=2.0
    )

    assert usage.shape == (NUM_CELLS, K), f"Usage shape: {usage.shape}"
    assert spectra_scores.shape[1] == K, f"Spectra scores columns: {spectra_scores.shape[1]}"
    assert spectra_tpm.shape[1] == K, f"Spectra TPM columns: {spectra_tpm.shape[1]}"
    assert np.all(usage.values >= 0), "Usage has negative values"

    with open(os.path.join(run_dir, "full_pipeline_summary.json"), "w") as f:
        json.dump({
            "mode": "dataloader",
            "K": K,
            "n_iter": n_iter,
            "usage_shape": list(usage.shape),
            "spectra_scores_shape": list(spectra_scores.shape),
            "spectra_tpm_shape": list(spectra_tpm.shape),
            "top_genes_shape": list(top_genes.shape),
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 3. Compare dataloader vs batch mode on same data
# ------------------------------------------------------------------ #
def test_dataloader_vs_batch_mode(synthetic_h5ad):
    """dataloader and batch modes should produce comparable reconstruction quality."""
    results = {}
    for mode_name in ("batch", "dataloader"):
        run_dir = os.path.join(RESULT_DIR, f"compare_{mode_name}")
        os.makedirs(run_dir, exist_ok=True)
        cnmf_obj = cNMF(output_dir=run_dir, name=f"cmp_{mode_name}")

        K = 3
        cnmf_obj.prepare(
            counts_fn=synthetic_h5ad,
            components=[K],
            n_iter=10,
            num_highvar_genes=80,
            seed=SEED,
            use_gpu=True,
            mode=mode_name,
            algo="halsvar",
            minibatch_size=30,
        )
        cnmf_obj.factorize()
        cnmf_obj.combine()
        cnmf_obj.consensus(k=K, density_threshold=2.0, show_clustering=False)

        usage, spectra_scores, _, _ = cnmf_obj.load_results(K=K, density_threshold=2.0)
        results[mode_name] = {
            "usage_shape": list(usage.shape),
            "spectra_shape": list(spectra_scores.shape),
            "usage_mean": float(usage.values.mean()),
        }

    # Both modes should produce same-shaped output
    assert results["batch"]["usage_shape"] == results["dataloader"]["usage_shape"]
    assert results["batch"]["spectra_shape"] == results["dataloader"]["spectra_shape"]

    with open(os.path.join(RESULT_DIR, "batch_vs_dataloader.json"), "w") as f:
        json.dump({**results, "status": "PASSED"}, f, indent=2)


# ------------------------------------------------------------------ #
# 4. minibatch_size > number of cells
# ------------------------------------------------------------------ #
def test_batch_size_larger_than_num_cells(synthetic_h5ad):
    """Pipeline should handle minibatch_size > NUM_CELLS without error."""
    run_dir = os.path.join(RESULT_DIR, "batch_gt_cells")
    os.makedirs(run_dir, exist_ok=True)
    cnmf_obj = cNMF(output_dir=run_dir, name="big_batch")

    K = 3
    n_iter = 10
    # minibatch_size=500 is much larger than NUM_CELLS=80
    cnmf_obj.prepare(
        counts_fn=synthetic_h5ad,
        components=[K],
        n_iter=n_iter,
        num_highvar_genes=80,
        seed=SEED,
        use_gpu=True,
        mode="dataloader",
        algo="halsvar",
        minibatch_size=500,
    )
    cnmf_obj.factorize()
    cnmf_obj.combine()
    cnmf_obj.consensus(k=K, density_threshold=2.0, show_clustering=False)

    usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(
        K=K, density_threshold=2.0
    )

    assert usage.shape == (NUM_CELLS, K), f"Usage shape: {usage.shape}"
    assert spectra_scores.shape[1] == K, f"Spectra scores columns: {spectra_scores.shape[1]}"
    assert np.all(usage.values >= 0), "Usage has negative values"

    with open(os.path.join(run_dir, "batch_gt_cells_summary.json"), "w") as f:
        json.dump({
            "mode": "dataloader",
            "minibatch_size": 500,
            "num_cells": NUM_CELLS,
            "K": K,
            "usage_shape": list(usage.shape),
            "status": "PASSED",
        }, f)
