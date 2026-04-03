"""
Test NMFDataset class and its integration with PyTorch DataLoader.
Adapted from Dylan Kotliar's original cNMF; torch-based version by Alexandra Mo.

Results saved to: Test/results/test_nmf_dataset/
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytest

from torch_cnmf.cnmf import NMFDataset


RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "test_nmf_dataset")


@pytest.fixture(autouse=True)
def setup_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
# 1. Construction from numpy
# ------------------------------------------------------------------ #
def test_dataset_from_numpy(nmf_problem):
    X, _, _ = nmf_problem
    ds = NMFDataset(X=X, dtype=torch.float32)

    assert len(ds) == X.shape[0], "Dataset length should equal n_samples"
    row, idx = ds[0]
    assert row.shape == (X.shape[1],), "Row shape mismatch"
    assert idx == 0

    with open(os.path.join(RESULT_DIR, "from_numpy.json"), "w") as f:
        json.dump({"len": len(ds), "row_shape": list(row.shape), "status": "PASSED"}, f)


# ------------------------------------------------------------------ #
# 2. Construction from torch tensor
# ------------------------------------------------------------------ #
def test_dataset_from_tensor(nmf_problem):
    X, _, _ = nmf_problem
    X_t = torch.from_numpy(X).to(dtype=torch.float64)
    ds = NMFDataset(X=X_t, dtype=torch.float32)

    assert len(ds) == X.shape[0]
    row, idx = ds[3]
    np.testing.assert_allclose(row.numpy(), X[3], atol=1e-5,
                               err_msg="Row content mismatch after dtype conversion")

    with open(os.path.join(RESULT_DIR, "from_tensor.json"), "w") as f:
        json.dump({"len": len(ds), "dtype": str(ds.X_cpu.dtype), "status": "PASSED"}, f)


# ------------------------------------------------------------------ #
# 3. DataLoader integration — batching and full coverage
# ------------------------------------------------------------------ #
def test_dataloader_batching(nmf_problem):
    X, _, _ = nmf_problem
    ds = NMFDataset(X=X, dtype=torch.float32)
    batch_size = 16
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    total_rows = 0
    all_indices = []
    for batch_x, batch_idx in loader:
        total_rows += batch_x.shape[0]
        all_indices.extend(batch_idx.tolist())

    assert total_rows == X.shape[0], "DataLoader should iterate over all rows"
    assert sorted(all_indices) == list(range(X.shape[0])), "All indices should appear exactly once"

    with open(os.path.join(RESULT_DIR, "dataloader_batching.json"), "w") as f:
        json.dump({
            "total_rows": total_rows,
            "n_batches": len(list(DataLoader(ds, batch_size=batch_size))),
            "batch_size": batch_size,
            "status": "PASSED",
        }, f)


# ------------------------------------------------------------------ #
# 4. Data content preserved through DataLoader
# ------------------------------------------------------------------ #
def test_dataloader_content_preserved(nmf_problem):
    X, _, _ = nmf_problem
    ds = NMFDataset(X=X, dtype=torch.float32)
    loader = DataLoader(ds, batch_size=len(X), shuffle=False)

    batch_x, batch_idx = next(iter(loader))
    np.testing.assert_allclose(
        batch_x.numpy(), X, atol=1e-6,
        err_msg="DataLoader should return the same data as the original array"
    )

    with open(os.path.join(RESULT_DIR, "content_preserved.json"), "w") as f:
        json.dump({"max_abs_diff": float(np.abs(batch_x.numpy() - X).max()),
                    "status": "PASSED"}, f)


# ------------------------------------------------------------------ #
# 5. Batches transfer to GPU correctly
# ------------------------------------------------------------------ #
def test_dataloader_gpu_transfer(nmf_problem, device):
    """DataLoader batches should be movable to the target device (GPU if available)."""
    X, _, _ = nmf_problem
    ds = NMFDataset(X=X, dtype=torch.float32)
    loader = DataLoader(ds, batch_size=16, shuffle=False, pin_memory=(device == "cuda"))

    dev = torch.device(device)
    for batch_x, batch_idx in loader:
        batch_x = batch_x.to(dev)
        assert batch_x.device.type == dev.type, f"Batch not on {device}"
        break  # one batch is enough to verify

    with open(os.path.join(RESULT_DIR, "gpu_transfer.json"), "w") as f:
        json.dump({"device": device, "status": "PASSED"}, f)
