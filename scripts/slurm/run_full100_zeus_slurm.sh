#!/bin/bash

#SBATCH --job-name=TFM_FULL100
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH -w zeus
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Stop on undefined variables and pipeline failures.
set -uo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 DATASET"
    echo "Example: sbatch $0 CIFAR10"
    echo "Example: sbatch $0 TINYIMAGENET"
    exit 2
fi

DATASET="${1^^}"
MODEL="alexnet"

case "$DATASET" in
    CIFAR10|TINYIMAGENET)
        ;;
    *)
        echo "ERROR: Unsupported dataset: $DATASET"
        exit 2
        ;;
esac

# ============================================================
# Conda environment
# ============================================================

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

set +u
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU1/jmrodriguez/venv_tfm
set -u

# ============================================================
# Deterministic execution
# ============================================================

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

# ============================================================
# Paths
# ============================================================

PROJECT_DIR="/mnt/homeGPU1/jmrodriguez"
DATA_DIR="${PROJECT_DIR}/data"
FEATURES_DIR="${DATA_DIR}/features"
OUTPUT_DIR="${PROJECT_DIR}/results"

cd "$PROJECT_DIR" || {
    echo "ERROR: Project directory not found."
    exit 1
}

FEATURE_PATH="${FEATURES_DIR}/${DATASET}/${MODEL}"

for REQUIRED_FILE in \
    train_features.pt \
    train_labels.pt \
    test_features.pt \
    test_labels.pt \
    metadata.json
do
    if [ ! -f "${FEATURE_PATH}/${REQUIRED_FILE}" ]; then
        echo "ERROR: Missing feature artifact: ${FEATURE_PATH}/${REQUIRED_FILE}"
        exit 1
    fi
done

DATASET_TAG="${DATASET,,}"

# Use a completely new directory so the old Hera baseline is never reused.
EXPERIMENT_NAME="main_full100_zeus_${DATASET_TAG}_${MODEL}"

echo "============================================================"
echo "TFM Stage-2 100% full-data baseline"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Seeds: 42 123 456 789 1024"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Start time: $(date)"
echo "============================================================"

nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total \
    --format=csv

python - <<'PYTHON_CHECK'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPU count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the Slurm job.")

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "This job must see exactly one GPU, but "
        f"{torch.cuda.device_count()} GPUs are visible."
    )

print(f"GPU name: {torch.cuda.get_device_name(0)}")
PYTHON_CHECK

STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    exit "$STATUS"
fi

echo
echo "Starting Zeus 100% baseline evaluation..."
echo

# SRS is included only because the current experiment runner expects one
# selection method. The 100% baseline itself is evaluated independently
# using the complete base-training partition.

python -u run_experiments.py \
    --algorithms SRS \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --metrics accuracy \
    --metric-averaging macro \
    --keep-percentages 0.10 \
    --seeds 42 123 456 789 1024 \
    --split-seed 42 \
    --max-evaluations 0 \
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
    --fail-fast \
    --experiment-name "$EXPERIMENT_NAME"

STATUS=$?

echo
echo "============================================================"
echo "Zeus 100% baseline finished"
echo "Dataset: ${DATASET}"
echo "End time: $(date)"
echo "Exit code: ${STATUS}"
echo "Baseline results:"
echo "${OUTPUT_DIR}/${EXPERIMENT_NAME}/baselines"
echo "============================================================"

exit "$STATUS"