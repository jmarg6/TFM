#!/bin/bash

#SBATCH --job-name=TFM_B1000
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH -w zeus
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Stop on undefined variables and pipeline failures.
# The Python exit status is handled explicitly below.
set -uo pipefail

# ============================================================
# Argument validation
# ============================================================

if [ "$#" -ne 5 ]; then
    echo "Usage: sbatch $0 MODEL DATASET ALGORITHM PERCENTAGE SEED"
    echo "Example: sbatch $0 alexnet CIFAR10 GEN 50 42"
    echo "Example: sbatch $0 alexnet TINYIMAGENET RS 75 123"
    exit 2
fi

MODEL="${1,,}"
DATASET="${2^^}"
ALGORITHM="${3^^}"
PERCENTAGE_INPUT="$4"
SEED="$5"

# ============================================================
# Main-stage configuration validation
# ============================================================

# The definitive main stage uses only the frozen AlexNet representation.
case "$MODEL" in
    alexnet)
        ;;
    *)
        echo "ERROR: Unsupported main-stage model: $MODEL"
        echo "The definitive main stage uses only: alexnet"
        exit 2
        ;;
esac

# The definitive main stage uses CIFAR-10 and Tiny ImageNet.
case "$DATASET" in
    CIFAR10|TINYIMAGENET)
        ;;
    *)
        echo "ERROR: Unsupported main-stage dataset: $DATASET"
        echo "Valid values: CIFAR10, TINYIMAGENET"
        exit 2
        ;;
esac

# The enlarged-budget comparison is restricted to GA and RS.
case "$ALGORITHM" in
    RS|GEN)
        ;;
    *)
        echo "ERROR: Unsupported main-stage algorithm: $ALGORITHM"
        echo "Valid values: RS, GEN"
        exit 2
        ;;
esac

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

# Only the five predefined paired experimental seeds are accepted.
case "$SEED" in
    42|123|456|789|1024)
        ;;
    *)
        echo "ERROR: Unsupported seed: $SEED"
        echo "Valid values: 42, 123, 456, 789, 1024"
        exit 2
        ;;
esac

# ============================================================
# Conda environment
# ============================================================

# Load the cluster Anaconda installation.
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

# Conda may access interactive-shell variables such as PS1.
# Temporarily disable nounset during Conda initialization.
set +u

eval "$(conda shell.bash hook)"
CONDA_HOOK_STATUS=$?

if [ "$CONDA_HOOK_STATUS" -ne 0 ]; then
    echo "ERROR: Conda shell initialization failed."
    exit "$CONDA_HOOK_STATUS"
fi

conda activate /mnt/homeGPU1/jmrodriguez/venv_tfm
CONDA_ACTIVATE_STATUS=$?

set -u

if [ "$CONDA_ACTIVATE_STATUS" -ne 0 ]; then
    echo "ERROR: Could not activate the TFM Conda environment."
    exit "$CONDA_ACTIVATE_STATUS"
fi

# ============================================================
# Deterministic execution configuration
# ============================================================

# Configure deterministic CUDA matrix operations.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Match CPU-side numerical libraries to the Slurm allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Flush Python output immediately so that Slurm logs can be monitored.
export PYTHONUNBUFFERED=1

# ============================================================
# Project paths
# ============================================================

PROJECT_DIR="/mnt/homeGPU1/jmrodriguez"
DATA_DIR="${PROJECT_DIR}/data"
FEATURES_DIR="${DATA_DIR}/features"
OUTPUT_DIR="${PROJECT_DIR}/results"

cd "$PROJECT_DIR" || {
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
}

FEATURE_PATH="${FEATURES_DIR}/${DATASET}/${MODEL}"

# ============================================================
# Verify cached feature artifacts
# ============================================================

for REQUIRED_FILE in \
    train_features.pt \
    train_labels.pt \
    test_features.pt \
    test_labels.pt \
    metadata.json
do
    if [ ! -f "${FEATURE_PATH}/${REQUIRED_FILE}" ]; then
        echo "ERROR: Missing feature artifact: ${FEATURE_PATH}/${REQUIRED_FILE}"
        echo "Run feature extraction for this dataset/model first."
        exit 1
    fi
done

# ============================================================
# Experiment naming
# ============================================================

DATASET_TAG="${DATASET,,}"

# Each seed uses a different experiment directory. This is essential because
# jobs belonging to different seeds may execute concurrently.
EXPERIMENT_NAME="main_b1000_${DATASET_TAG}_${MODEL}_${ALGORITHM}_kp${PERCENTAGE_TAG}_seed${SEED}"

# ============================================================
# Experiment information
# ============================================================

echo "============================================================"
echo "TFM main-stage B=1000 experiment"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Algorithm: ${ALGORITHM}"
echo "Keep percentage: ${KEEP_PERCENTAGE}"
echo "Fitness evaluation budget: 1000"
echo "Seed: ${SEED}"
echo "Split seed: 42"
echo "Epochs per candidate: 10"
echo "Batch size: 32"
echo "Maximum Slurm wall time: 4 days"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Feature path: ${FEATURE_PATH}"
echo "Output path: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================
# GPU validation
# ============================================================

nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total \
    --format=csv

python - <<'PYTHON_CHECK'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPU count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available inside the Slurm job."
    )

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "This job must see exactly one GPU, but "
        f"{torch.cuda.device_count()} GPUs are visible."
    )

print(f"GPU name: {torch.cuda.get_device_name(0)}")
PYTHON_CHECK

CUDA_CHECK_STATUS=$?

if [ "$CUDA_CHECK_STATUS" -ne 0 ]; then
    echo "ERROR: CUDA validation failed."
    exit "$CUDA_CHECK_STATUS"
fi

echo
echo "Starting B=1000 main-stage search..."
echo

# ============================================================
# Main experiment
# ============================================================

# The full-data baseline is intentionally excluded. It will be executed
# separately on Zeus so that Stage-2 timing measurements use the same node.

python -u run_experiments.py \
    --algorithms "$ALGORITHM" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --metrics accuracy \
    --metric-averaging macro \
    --keep-percentages "$KEEP_PERCENTAGE" \
    --seeds "$SEED" \
    --split-seed 42 \
    --max-evaluations 1000 \
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
    --num-workers 0 \
    --data-dir "$DATA_DIR" \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-features \
    --skip-full-baseline \
    --fail-fast \
    --experiment-name "$EXPERIMENT_NAME"

STATUS=$?

# ============================================================
# Final status
# ============================================================

echo
echo "============================================================"
echo "TFM B=1000 main-stage experiment finished"
echo "Dataset: ${DATASET}"
echo "Algorithm: ${ALGORITHM}"
echo "Keep percentage: ${KEEP_PERCENTAGE}"
echo "Seed: ${SEED}"
echo "End time: $(date)"
echo "Exit code: ${STATUS}"
echo "Results: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "============================================================"

exit "$STATUS"