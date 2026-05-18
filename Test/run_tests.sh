#!/bin/bash
# Run all DataLoader-related tests on a GPU node.
# Each test saves results to Test/results/<test_name>/
#
# Usage:
#   cd /oak/stanford/groups/engreitz/Users/ymo/Tools/torch-cnmf
#   sbatch Test/run_tests.sh            # submit to SLURM
#   bash   Test/run_tests.sh            # run interactively (if on a GPU node)
#   bash   Test/run_tests.sh -k dataset # run only NMFDataset tests

#SBATCH --job-name=cnmf_test
#SBATCH --partition=gpu,owners
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=Test/results/test_slurm_%j.out
#SBATCH --error=Test/results/test_slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ymo@stanford.edu


set -e

# Under SLURM, $0 points to a copy in /var/spool/slurmd, so derive paths
# from SLURM_SUBMIT_DIR when available; fall back to $0 for interactive runs.
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi

cd "$PROJECT_DIR"

# ── Activate conda environment ──
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-nmf-dl

echo "=== Device info ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo "==================="

python -m pytest /oak/stanford/groups/engreitz/Users/ymo/Tools/torch-cnmf/Test \
    --tb=short \
    -v \
    "$@"
