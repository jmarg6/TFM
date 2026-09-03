#!/bin/bash

#SBATCH --job-name=TFM_GKCG
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH -w zeus
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Stop immediately on command failures, undefined variables, and pipeline
# failures.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 DATASET"
    echo "Example: sbatch $0 CIFAR10"
    exit 2
fi

DATASET="${1^^}"
MODEL="alexnet"
ALGORITHM="KCG"
KEEP_PERCENTAGES=(0.10 0.25 0.50 0.75 0.90)

case "$DATASET" in
    CIFAR10|TINYIMAGENET)
        ;;
    *)
        echo "ERROR: Unsupported main-stage dataset: $DATASET"
        echo "Valid values: CIFAR10, TINYIMAGENET"
        exit 2
        ;;
esac

# Load and activate the TFM Conda environment.
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

# Conda may access interactive-shell variables that are undefined in a Slurm
# batch shell. Disable nounset only during Conda initialization.
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

# Configure deterministic CUDA matrix operations.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

PROJECT_DIR="/mnt/homeGPU1/jmrodriguez"
DATA_DIR="${PROJECT_DIR}/data"
FEATURES_DIR="${DATA_DIR}/features"
K_CENTER_CACHE_ROOT="${FEATURES_DIR}/k_center_cache"
OUTPUT_DIR="${PROJECT_DIR}/results"

cd "$PROJECT_DIR" || {
    echo "ERROR: Project directory not found: $PROJECT_DIR"
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
        echo "Run extract_features_slurm.sh for this dataset/model first."
        exit 1
    fi
done

DATASET_TAG="${DATASET,,}"
EXPERIMENT_NAME="main_global_kcg_${DATASET_TAG}_${MODEL}_all"

if [ "$DATASET" = "CIFAR10" ]; then
    EXPECTED_BASE_TRAIN=40000
    EXPECTED_DISTANCE_CACHE_GIB="approximately 5.96 GiB"
else
    EXPECTED_BASE_TRAIN=80000
    EXPECTED_DISTANCE_CACHE_GIB="approximately 23.84 GiB"
fi

echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python)"
echo "Python version: $(python --version 2>&1)"
echo "============================================================"
echo "TFM global k-center greedy coreset baseline"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Method: ${ALGORITHM}"
echo "Keep percentages: ${KEEP_PERCENTAGES[*]}"
echo "Selection seeds: 42 123 456 789 1024"
echo "Selection scope: global"
echo "Class labels used for selection: no"
echo "Validation used for selection: no"
echo "Feature normalization: none"
echo "Distance: squared Euclidean"
echo "Initial center: one seeded random global instance"
echo "Traversal: farthest-first"
echo "Retention masks: nested prefixes of one order per seed"
echo "Expected base-training size: ${EXPECTED_BASE_TRAIN}"
echo "Expected float32 distance cache: ${EXPECTED_DISTANCE_CACHE_GIB}"
echo "Cache root: ${K_CENTER_CACHE_ROOT}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Feature path: ${FEATURE_PATH}"
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

echo
echo "Building/reusing the global distance cache, constructing one nested"
echo "farthest-first order per seed, and evaluating every retention level..."
echo

python -u run_global_kcenter_experiments.py \
    --algorithms "$ALGORITHM" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --metrics accuracy \
    --metric-averaging macro \
    --keep-percentages "${KEEP_PERCENTAGES[@]}" \
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
    --k-center-cache-dir "$K_CENTER_CACHE_ROOT" \
    --k-center-distance-block-size 512 \
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

echo
echo "============================================================"
echo "Global KCG baseline finished"
echo "End time: $(date)"
echo "Exit code: ${STATUS}"
echo "Results: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "Cache: ${K_CENTER_CACHE_ROOT}/${DATASET}/${MODEL}/split_seed_42"
echo "============================================================"

exit "$STATUS"
