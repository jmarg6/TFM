#!/bin/bash

#SBATCH --job-name=TFM_Prelim
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH -w titan
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Stop on undefined variables and pipeline failures.
# The experiment exit code is handled explicitly below.
set -uo pipefail

# ---------------------------------------------------------------------------
# Command-line arguments
#
# Usage:
#   sbatch run_preliminary_slurm.sh MODEL DATASET ALGORITHM PERCENTAGE
#
# Example:
#   sbatch run_preliminary_slurm.sh alexnet MNIST RS 10
#
# The five experimental seeds are executed automatically inside the same job.
# ---------------------------------------------------------------------------

if [ "$#" -ne 4 ]; then
    echo "ERROR: Exactly four arguments are required."
    echo
    echo "Usage:"
    echo "  sbatch $0 MODEL DATASET ALGORITHM PERCENTAGE"
    echo
    echo "Valid models:"
    echo "  alexnet"
    echo "  resnext"
    echo "  efficientnetv2"
    echo "  swintransformer"
    echo
    echo "Valid datasets:"
    echo "  MNIST"
    echo "  CIFAR10"
    echo "  TINYIMAGENET"
    echo
    echo "Valid algorithms:"
    echo "  RS"
    echo "  LS"
    echo "  GEN"
    echo "  MEM"
    echo
    echo "Valid percentages:"
    echo "  10, 25, 50, 75, 90"
    echo
    echo "Example:"
    echo "  sbatch $0 alexnet MNIST RS 10"
    exit 2
fi

# Normalize arguments to the names expected by run_experiments.py.
MODEL="${1,,}"
DATASET="${2^^}"
ALGORITHM="${3^^}"
PERCENTAGE_INPUT="$4"

# ---------------------------------------------------------------------------
# Validate the selected model.
# ---------------------------------------------------------------------------

case "$MODEL" in
    alexnet|resnext|efficientnetv2|swintransformer)
        ;;
    *)
        echo "ERROR: Unsupported model: $MODEL"
        echo "Valid values: alexnet, resnext, efficientnetv2, swintransformer"
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Validate the selected dataset.
# ---------------------------------------------------------------------------

case "$DATASET" in
    MNIST|CIFAR10|TINYIMAGENET)
        ;;
    *)
        echo "ERROR: Unsupported dataset: $DATASET"
        echo "Valid values: MNIST, CIFAR10, TINYIMAGENET"
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Validate the selected instance-selection algorithm.
# ---------------------------------------------------------------------------

case "$ALGORITHM" in
    RS|LS|GEN|MEM)
        ;;
    *)
        echo "ERROR: Unsupported algorithm: $ALGORITHM"
        echo "Valid values: RS, LS, GEN, MEM"
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Convert the percentage to the decimal value expected by Python.
# ---------------------------------------------------------------------------

case "$PERCENTAGE_INPUT" in
    10|0.10)
        KEEP_PERCENTAGE="0.10"
        PERCENTAGE_TAG="10"
        ;;
    25|0.25)
        KEEP_PERCENTAGE="0.25"
        PERCENTAGE_TAG="25"
        ;;
    50|0.50)
        KEEP_PERCENTAGE="0.50"
        PERCENTAGE_TAG="50"
        ;;
    75|0.75)
        KEEP_PERCENTAGE="0.75"
        PERCENTAGE_TAG="75"
        ;;
    90|0.90)
        KEEP_PERCENTAGE="0.90"
        PERCENTAGE_TAG="90"
        ;;
    *)
        echo "ERROR: Unsupported percentage: $PERCENTAGE_INPUT"
        echo "Valid values: 10, 25, 50, 75, 90"
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

# Load the cluster Anaconda installation.
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

# Activate the TFM environment.
conda activate /mnt/homeGPU1/jmrodriguez/venv_tfm

# Configure deterministic CUDA matrix operations.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Limit CPU libraries to the resources allocated by Slurm.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Write Python logs immediately.
export PYTHONUNBUFFERED=1

# Project paths.
PROJECT_DIR="/mnt/homeGPU1/jmrodriguez"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/results"

cd "$PROJECT_DIR" || {
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
}

# Build one experiment directory containing the five seed runs.
DATASET_TAG="${DATASET,,}"

EXPERIMENT_NAME=$(
    printf \
        "%s_%s_%s_kp%s" \
        "$DATASET_TAG" \
        "$MODEL" \
        "$ALGORITHM" \
        "$PERCENTAGE_TAG"
)

echo "============================================================"
echo "TFM preliminary experiment"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "Start time: $(date)"
echo
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Algorithm: ${ALGORITHM}"
echo "Keep percentage: ${KEEP_PERCENTAGE}"
echo "Seeds: 42 123 456 789 1024"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "============================================================"

echo
echo "GPU assigned to this job:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv

echo
echo "Checking CUDA environment..."

python - <<'PYTHON_CHECK'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"Visible GPU count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm job.")

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "This job must see exactly one GPU, but "
        f"{torch.cuda.device_count()} GPUs are visible."
    )

print(f"Assigned GPU: {torch.cuda.get_device_name(0)}")
PYTHON_CHECK

CUDA_CHECK_STATUS=$?

if [ "$CUDA_CHECK_STATUS" -ne 0 ]; then
    echo "ERROR: CUDA validation failed."
    exit "$CUDA_CHECK_STATUS"
fi

echo
echo "Starting preliminary experiment..."
echo

# The five seeds are executed sequentially inside this same Python process.
#
# Test evaluation is deliberately skipped during the preliminary phase.
# The winning configuration must be selected using validation results only.
python -u run_experiments.py \
    --algorithms "$ALGORITHM" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --metrics accuracy \
    --metric-averaging macro \
    --keep-percentages "$KEEP_PERCENTAGE" \
    --seeds 42 123 456 789 1024 \
    --split-seed 42 \
    --max-evaluations 100 \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --weight-decay 0.0 \
    --validation-fraction 0.2 \
    --population-size 10 \
    --tournament-size 3 \
    --mutation-probability 0.1 \
    --mutation-fraction 0.05 \
    --local-search-num-swaps 10 \
    --memetic-local-search-probability 0.2 \
    --memetic-local-search-steps 10 \
    --memetic-local-search-num-swaps 5 \
    --device cuda \
    --num-workers 4 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --empty-cuda-cache \
    --skip-test \
    --fail-fast \
    --experiment-name "$EXPERIMENT_NAME"

STATUS=$?

echo
echo "============================================================"
echo "Preliminary experiment finished"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "End time: $(date)"
echo "Exit code: ${STATUS}"
echo "============================================================"

exit "$STATUS"