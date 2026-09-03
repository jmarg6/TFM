#!/bin/bash

#SBATCH --job-name=TFM_SRS
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH -w zeus
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Stop on undefined variables and pipeline failures.
# The Python exit status is handled explicitly below.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 DATASET PERCENTAGE"
    echo "Example: sbatch $0 CIFAR10 10"
    exit 2
fi

DATASET="${1^^}"
PERCENTAGE_INPUT="$2"
MODEL="alexnet"
ALGORITHM="SRS"

case "$DATASET" in
    CIFAR10|TINYIMAGENET)
        ;;
    *)
        echo "ERROR: Unsupported main-stage dataset: $DATASET"
        echo "Valid values: CIFAR10, TINYIMAGENET"
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

# Load and activate the TFM Conda environment.
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

# Conda may access variables such as PS1 that are undefined in a
# non-interactive Slurm shell. Disable nounset only during activation.
set +u

eval "$(conda shell.bash hook)"
CONDA_HOOK_STATUS=$?

if [ "$CONDA_HOOK_STATUS" -ne 0 ]; then
    echo "ERROR: Conda shell initialization failed."
    exit "$CONDA_HOOK_STATUS"
fi

conda activate /mnt/homeGPU1/jmrodriguez/venv_tfm
CONDA_ACTIVATE_STATUS=$?

# Restore undefined-variable checking.
set -u

if [ "$CONDA_ACTIVATE_STATUS" -ne 0 ]; then
    echo "ERROR: Could not activate the TFM Conda environment."
    exit "$CONDA_ACTIVATE_STATUS"
fi

echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: $(command -v python)"
echo "Python version: $(python --version 2>&1)"

# Configure deterministic CUDA matrix operations.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

PROJECT_DIR="/mnt/homeGPU1/jmrodriguez"
DATA_DIR="${PROJECT_DIR}/data"
FEATURES_DIR="${DATA_DIR}/features"
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
EXPERIMENT_NAME="main_srs_${DATASET_TAG}_${MODEL}_kp${PERCENTAGE_TAG}"

echo "============================================================"
echo "TFM single stratified random sample baseline"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Method: ${ALGORITHM}"
echo "Keep percentage: ${KEEP_PERCENTAGE}"
echo "Sampling seeds: 42 123 456 789 1024"
echo "Fitness evaluations per seed: 0"
echo "Generated subsets per seed: 1"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Feature path: ${FEATURE_PATH}"
echo "Start time: $(date)"
echo "============================================================"

nvidia-smi

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

CUDA_CHECK_STATUS=$?

if [ "$CUDA_CHECK_STATUS" -ne 0 ]; then
    exit "$CUDA_CHECK_STATUS"
fi

echo
echo "Generating one stratified mask per seed and evaluating it on test..."
echo

python -u run_experiments.py \
    --algorithms "$ALGORITHM" \
    --datasets "$DATASET" \
    --models "$MODEL" \
    --metrics accuracy \
    --metric-averaging macro \
    --keep-percentages "$KEEP_PERCENTAGE" \
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
    --skip-full-baseline \
    --fail-fast \
    --experiment-name "$EXPERIMENT_NAME"

STATUS=$?

echo
echo "============================================================"
echo "SRS baseline finished"
echo "End time: $(date)"
echo "Exit code: ${STATUS}"
echo "Results: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "============================================================"

exit "$STATUS"
