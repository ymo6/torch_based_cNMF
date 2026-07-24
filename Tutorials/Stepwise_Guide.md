# Step by step guide (torch-cNMF)

torch-cNMF is a GPU-accelerated PyTorch implementation of consensus NMF. It replaces the scikit-learn NMF backend with `nmf-torch`, enabling GPU acceleration and additional algorithms and learning modes.

You can see all possible command line options by running
```
cnmf --help
```

and see the [PBMC dataset tutorial](torch_cnmf_inference_tutorial.ipynb) for a step by step walkthrough with example data. We also describe some key ideas and parameters for each step below.

### Step 1 - normalize the input matrix and prepare the run parameters

Example command:

```
cnmf prepare --output-dir ./example_data --name example_cNMF \
    -c ./example_data/counts_prefiltered.h5ad \
    -k 5 6 7 8 9 10 11 12 13 --n-iter 100 --seed 14 --numgenes 2000 \
    --use_gpu --algo halsvar --mode batch
```

Path structure
  - --output-dir - the output directory into which all results will be placed. Default: `.`
  - --name - a subdirectory output_dir/name will be created and all output files will have name as their prefix. Default: `cNMF`

Input data
  - -c - path to the cell x gene counts matrix. Supported formats: `.h5ad`, `.mtx`, `.mtx.gz`, `.npz`, or tab-delimited text
  - --tpm [Optional] - Pre-computed Cell x Gene data in transcripts per million or other per-cell normalized data. If none is provided, TPM will be calculated automatically. This can be helpful if a particular normalization is desired. These can be loaded in the same formats as the counts file. Default: `None`
  - --genes-file [Optional] - List of over-dispersed genes to be used for the factorization steps. If not provided, over-dispersed genes will be calculated automatically and the number of genes to use can be set by the --numgenes parameter below. Default: `None`

Parameters
  - -k - space separated list of K values that will be tested for cNMF
  - --n-iter - number of NMF iterations to run for each K. Default: `100`
  - --seed - the master seed that will be used to generate the individual seed for each NMF replicate. Default: `None`
  - --numgenes - the number of highest variance genes that will be used for running the factorization. Removing low variance genes helps amplify the signal and is an important factor in correctly inferring programs in the data. The final spectra is re-fit to include estimates for all genes, even those not in the high-variance set. Default: `2000`
  - --beta-loss - Loss function for NMF, from one of `frobenius`, `kullback-leibler`, `itakura-saito`. Default: `frobenius`
  - --densify - Treat the input data as non-sparse. Not recommended for most single-cell RNA-Seq data. Default: `False`

GPU / torch parameters
  - --use_gpu - Enable GPU acceleration via PyTorch. Default: `False`
  - --algo - NMF algorithm. One of `mu` (Multiplicative Update), `hals` (Hierarchical ALS), `halsvar` (HALS variant mimicking BPP), `bpp` (Block Principal Pivoting). Default: `halsvar`
  - --mode - Learning mode. One of `batch`, `minibatch`, `dataloader`. Note: `minibatch` and `dataloader` only work with `frobenius` loss. Default: `batch`
  - --init - Initialization method: `random` or `nndsvd`. Default: `random`
  - --tol - Convergence tolerance. Default: `1e-4`
  - --n-jobs - Number of CPU threads for PyTorch. `-1` uses PyTorch default. Default: `-1`
  - --fp-precision - Numeric precision: `float` (float32) or `double` (float64). Default: `float`

Regularization parameters
  - --alpha-usage - Regularization strength on the usage matrix W. Default: `0.0`
  - --alpha-spectra - Regularization strength on the spectra matrix H. Default: `0.0`
  - --l1-ratio-usage - L1 penalty ratio on W (between 0 and 1; L2 ratio is `1 - l1_ratio`). Default: `0.0`
  - --l1-ratio-spectra - L1 penalty ratio on H (between 0 and 1). Default: `0.0`

Batch mode parameters
  - --batch-max-epoch - Maximum epochs for batch learning. Default: `500`
  - --batch-hals-tol - HALS tolerance for the halsvar algorithm. Default: `0.05`
  - --batch-hals-max-iter - Maximum HALS iterations per H/W update. Default: `200`

Minibatch mode parameters
  - --batch_size - Size of mini-batches. Default: `5000`
  - --shuffle - Enable shuffling of samples across mini-batches each epoch. Default: `False`
  - --minibatch-max-epoch - Maximum passes over all data. Default: `20`
  - --minibatch-max-iter - Maximum iterations for H/W update per mini-batch. Default: `200`
  - --minibatch-usage-tol - Convergence tolerance for usage updates in minibatch mode. Default: `1e-7`
  - --minibatch-spectra-tol - Convergence tolerance for spectra updates in minibatch mode. Default: `1e-7`

Refit parameters
  - --sk-cd-refit - Use the sklearn coordinate descent solver for the refit step. Default: `False`

This command generates a filtered and normalized matrix for running the factorizations on. It first subsets the data down to a set of over-dispersed genes that can be provided as an input file or calculated here. While the final spectra will be computed for all of the genes in the input counts file, the factorization is much faster and can find better patterns if it only runs on a set of high-variance genes. A per-cell normalized input file may be provided as well so that the final gene expression programs can be computed with respect to that normalization.

__Please note that the input matrix should not include any cells or genes with 0 total counts. Furthermore if any of the cells end up having 0 counts for the over-dispersed genes, that can cause an error. Please filter out cells and genes with low counts prior to running cNMF.__

### Step 2 - factorize the matrix

NMF is run for all of the replicates specified in the previous command. All jobs run sequentially in a single process (GPU-accelerated when `--use_gpu` was set during prepare):

```
cnmf factorize --output-dir ./example_data --name example_cNMF
```

To resume and skip already-completed replicates:
```
cnmf factorize --output-dir ./example_data --name example_cNMF --skip-completed-runs
```

### Step 3 - combine the individual spectra results files for each K into a merged file
Since a separate file has been created for each replicate for each K, we combine the replicates for each K as below:
```
cnmf combine --output-dir ./example_data --name example_cNMF
```
After this, you can optionally delete the individual spectra files like so:
```
rm ./example_data/example_cNMF/cnmf_tmp/example_cNMF.spectra.k_*.iter_*.df.npz
```

### Step 4 - select an optimal K by considering the trade-off between stability and error
This will iterate through all of the values of K that have been run and will calculate the stability and error.
It then outputs a PNG image file plotting this relationship into the output_dir/name directory.

```
cnmf k_selection_plot --output-dir ./example_data --name example_cNMF
```

This outputs a K selection plot to example_data/example_cNMF/example_cNMF.k_selection.png. There is no universally definitive criteria for choosing K but we will typically use the largest value that is reasonably stable and/or a local maximum in stability. See the discussion and methods section and the response to reviewer comments in [the manuscript](https://elifesciences.org/articles/43803) for more discussion about selecting K.

### Step 5 - obtain consensus estimates for the programs and their usages at the desired value of K
The last step is to cluster the spectra after first optionally filtering out outliers. This step ultimately outputs 4 files:
  - GEP estimate in units of TPM
  - GEP estimate in units of TPM Z-scores, reflecting whether having a higher usage of a program would be expected to decrease or increase gene expression
  - Unnormalized GEP usage estimate
  - Clustergram diagnostic plot, showing how much consensus there is amongst the replicates and a histogram of distances between each spectra and its K nearest neighbors

We recommend that you use the diagnostic plot to determine the threshold to filter outliers. By default cNMF sets the number of neighbors to use for this filtering as 30% of the number of iterations done. But this can be modified from the command line.

In practice, we tend to run this command twice, once with --local-density-threshold 2.00 to see what the distribution of average distances looks like, and then a second time with --local-density-threshold set to a smaller value determined based on this histogram to filter out outliers. See the tutorials for examples of this.

```
cnmf consensus --output-dir ./example_data --name example_cNMF --components 10 --local-density-threshold 0.2 --show-clustering
```
  - --components - value of K to compute consensus clusters for. Must be among the options provided to the prepare step
  - --local-density-threshold - the threshold on average distance to K nearest neighbors to use. 2.0 or above means that nothing will be filtered out. Default: `0.5`
  - --local-neighborhood-size - Percentage of replicates to consider as nearest neighbors for local density filtering. E.g. if you run 100 replicates, and set this to .3, 30 nearest neighbors will be used for outlier detection. Default: `0.3`
  - --show-clustering - Controls whether or not the clustergram image is output. Default: `False`

By the end of this step, you should have the following results files in your directory:
  - Z-score unit gene expression program matrix - `example_data/example_cNMF/example_cNMF.gene_spectra_score.k_10.dt_0_01.txt`
  - TPM unit gene expression program matrix - `example_data/example_cNMF/example_cNMF.gene_spectra_tpm.k_10.dt_0_01.txt`
  - Usage matrix - `example_data/example_cNMF/example_cNMF.usages.k_10.dt_0_01.consensus.txt`
  - Diagnostic plot - `example_data/example_cNMF/example_cNMF.clustering.k_10.dt_0_01.png`

See the tutorials for subsequent analysis steps that can be used to analyze these results files once they are created.
