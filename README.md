### <code style="color : red">Notice: This is a fork of the cNMF repo with the core scikit-learn NMF implementation switched out for nmf-torch.</code>
* Singularity Container can be found in https://hub.docker.com/r/igvf/torch-cnmf/tags
* Make sure to install the right nmf-torch in https://github.com/ymo6/nmf-torch.git for modified cNMF (the modified nmf-torch use dataloader to train online/mini-batch mode)

# torch-cnmf: GPU-Accelerated Consensus NMF

Based on the cNMF method described in [Kotliar et al., eLife 2019](https://elifesciences.org/articles/43803).

## Key Features

- **GPU acceleration** via PyTorch with automatic fallback to CPU
- **Multiple NMF algorithms**: Multiplicative Update (MU), HALS, HALS-variant, and Block Principal Pivoting (BPP)
- **Three learning modes**: batch, mini-batch, and DataLoader-based online learning for memory-efficient processing of large datasets
- **Flexible loss functions**: Frobenius, Kullback-Leibler, and Itakura-Saito divergence
- **L1/L2 regularization** on both usage (W) and spectra (H) matrices
- **Batch correction** via Harmony integration for multi-sample and multi-batch experiments
- **CITE-Seq support** with separate ADT and RNA handling in preprocessing
- **starCAT compatibility** for annotating new datasets with learned GEPs
- **Efficient sparse matrix handling** with batched OLS computation for reduced memory usage

# Installation

Requires Python >= 3.9, nmf-torch, scikit-learn>=1.0, scanpy>=1.8, and AnnData>=0.9.

Install from pip:
```bash
pip install torch-cnmf
```

Or install from source (editable):
```bash
git clone https://github.com/ymo6/torch_based_cNMF.git
cd torch_based_cNMF
pip install -e .
```

# Running cNMF

cNMF can be run from the command line without any parallelization using the example commands below:

```bash
cnmf prepare --output-dir ./example_data --name example_cNMF -c ./example_data/counts_prefiltered.txt -k 5 6 7 8 9 10 11 12 13 --n-iter 100 --use_gpu --batch_size 5000 --seed 14

cnmf factorize --output-dir ./example_data --name example_cNMF

cnmf combine --output-dir ./example_data --name example_cNMF

cnmf k_selection_plot --output-dir ./example_data --name example_cNMF

cnmf consensus --output-dir ./example_data --name example_cNMF --components 10 --local-density-threshold 0.01 --show-clustering
```

Or alternatively, the same steps can be run from within a Python environment using the commands below:

```python
from torch_cnmf import cNMF
cnmf_obj = cNMF(output_dir="./example_data", name="example_cNMF")
cnmf_obj.prepare(counts_fn="./example_data/counts_prefiltered.txt", components=np.arange(5,14), n_iter=100, seed=14, use_gpu=True)
cnmf_obj.factorize()
cnmf_obj.combine()
cnmf_obj.k_selection_plot()
cnmf_obj.consensus(k=10, density_threshold=0.01)
usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=10, density_threshold=0.01)
```

For the Python environment approach, `usage` will contain the usage matrix with each cell normalized to sum to 1. `spectra_scores` contains the gene_spectra_scores output (aka Z-score unit gene expression matrix), `spectra_tpm` contains the GEP spectra in units of TPM and `top_genes` contains an ordered list of the top 100 associated genes for each program.

Output data files will all be available in the ./example_data/example_cNMF directory including:

   - Z-score unit gene expression program matrix - `example_data/example_cNMF/example_cNMF.gene_spectra_score.k_10.dt_0_01.txt`
   - TPM unit gene expression program matrix - `example_data/example_cNMF/example_cNMF.gene_spectra_tpm.k_10.dt_0_01.txt`
   - Usage matrix - `example_data/example_cNMF/example_cNMF.usages.k_10.dt_0_01.consensus.txt`
   - K selection plot - `example_data/example_cNMF/example_cNMF.k_selection.png`
   - Clustergram diagnostic plot - `example_data/example_cNMF/example_cNMF.clustering.k_10.dt_0_01.pdf`

Some usage notes:
 - __Input data__: Input data can be provided in multiple formats:
    - 1. as a scanpy file ending in .h5ad containing counts as the data feature. See the PBMC dataset tutorial for an example of how to generate the Scanpy object from the data provided by 10X. Because Scanpy uses sparse matrices by default, the .h5ad data structure can take up much less memory than the raw counts matrix and can be much faster to load.
    - 2. as a raw tab-delimited text file containing row labels with cell IDs (barcodes) and column labels as gene IDs
    - 3. as a 10x-Genomics-formatted mtx directory. You provide the path to the counts.mtx file or counts.mtx.gz file to counts_fn. It expects there to be barcodes.tsv and genes.tsv in the directory as well
    - 4. as a .npz sparse matrix file

See the tutorials or Stepwise_Guide.md for more details

# GPU Acceleration

Enable GPU acceleration by passing `--use_gpu` on the command line or `use_gpu=True` in the Python API. torch-cnmf automatically detects GPU availability and falls back to CPU if no GPU is found.

```bash
cnmf prepare --output-dir ./results --name mydata -c counts.h5ad \
  -k 5 6 7 8 9 10 --n-iter 100 --use_gpu --seed 14
```

# NMF Algorithms and Learning Modes

## Algorithms

torch-cnmf supports four NMF algorithms via the `--algo` parameter:

| Algorithm | Flag | Description |
|-----------|------|-------------|
| HALS-variant | `halsvar` (default) | HALS variant that mimics BPP for potentially better convergence |
| HALS | `hals` | Hierarchical Alternating Least Squares |
| Multiplicative Update | `mu` | Classic multiplicative update rules |
| Block Principal Pivoting | `bpp` | BPP method for non-negative least squares |

## Learning Modes

Three learning modes are available via the `--mode` parameter:

| Mode | Flag | Description |
|------|------|-------------|
| Batch | `batch` (default) | Standard batch learning with full dataset passes |
| Mini-batch | `minibatch` | Mini-batch SGD-like training for memory efficiency |
| DataLoader | `dataloader` | PyTorch DataLoader-based training with chunked processing, recommended for very large datasets |

For mini-batch and DataLoader modes, control the batch size with `--batch_size` (default: 5000).

## Loss Functions

Specify the loss function with `--beta-loss`:

- `frobenius` (default) - L2 distance, fastest
- `kullback-leibler` - KL divergence
- `itakura-saito` - IS divergence

## Regularization

L1 and L2 regularization can be applied independently to both the usage (W) and spectra (H) matrices:

```bash
cnmf prepare --output-dir ./results --name mydata -c counts.h5ad \
  -k 10 --n-iter 100 --use_gpu \
  --alpha-usage 0.1 --alpha-spectra 0.1 \
  --l1-ratio-usage 0.5 --l1-ratio-spectra 0.5
```

- `--alpha-usage` / `--alpha-spectra`: Overall regularization strength (default: 0)
- `--l1-ratio-usage` / `--l1-ratio-spectra`: Balance between L1 and L2 (0 = pure L2, 1 = pure L1; default: 0)

## Advanced Parameters

Additional parameters for fine-tuning NMF behavior:

- `--init`: Initialization method (`random`, `nndsvd`, `nndsvda`, `nndsvdar`)
- `--tol`: Convergence tolerance (default: 1e-4)
- `--fp-precision`: Floating point precision (`float` or `double`, default: `float`)
- `--batch-max-epoch`: Max epochs for batch mode (default: 500)
- `--minibatch-max-epoch`: Max epochs for mini-batch mode (default: 20)
- `--shuffle`: Enable shuffling across mini-batches
- `--sk-cd-refit`: Use scikit-learn coordinate descent for refitting (default: False)

# Change log

### Version 1.0.0
- Initial release of torch-cnmf, forked from cNMF with PyTorch-based NMF backend
- GPU acceleration via nmf-torch with automatic CPU fallback
- Multiple NMF algorithms: MU, HALS, HALS-variant, BPP
- Three learning modes: batch, mini-batch, and DataLoader-based online learning
- Flexible loss functions: Frobenius, KL divergence, Itakura-Saito
- L1/L2 regularization on usage and spectra matrices
- Efficient sparse + batched OLS computation
- Basic testing suite
