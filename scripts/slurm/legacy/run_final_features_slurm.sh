#!/bin/bash

#SBATCH --job-name=TFM_FinalTest
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH -w zeus
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Stop on undefined variables and pipeline failures.
# The Python exit status is handled explicitly below.
set -uo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: sbatch $0 MODEL DATASET ALGORITHM PERCENTAGE"
    echo "Example: sbatch $0 alexnet MNIST RS 10"
    exit 2
fi

MODEL="${1,,}"
DATASET="${2^^}"
ALGORITHM="${3^^}"
PERCENTAGE_INPUT="$4"

case "$MODEL" in
    alexnet|resnext|efficientnetv2|swintransformer)
        ;;
    *)
        echo "ERROR: Unsupported model: $MODEL"
        exit 2
        ;;
esac

case "$DATASET" in
    MNIST|CIFAR10|TINYIMAGENET)
        ;;
    *)
        echo "ERROR: Unsupported dataset: $DATASET"
        exit 2
        ;;
esac

case "$ALGORITHM" in
    RS|LS|GEN|MEM)
        ;;
    *)
        echo "ERROR: Unsupported algorithm: $ALGORITHM"
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
        exit 2
        ;;
esac

# Load the cluster Anaconda installation.
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

# Activate the TFM environment.
conda activate /mnt/homeGPU1/jmrodriguez/venv_tfm

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
EXPERIMENT_NAME="prelim_features_${DATASET_TAG}_${MODEL}_${ALGORITHM}_kp${PERCENTAGE_TAG}"

echo "============================================================"
echo "TFM final test evaluation using cached features"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Algorithm: ${ALGORITHM}"
echo "Keep percentage: ${KEEP_PERCENTAGE}"
echo "Seeds: 42 123 456 789 1024"
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
echo "Starting final test evaluation..."
echo

# Evaluate the full-data baseline only when explicitly requested.
FINAL_EXTRA_ARGS=()

if [ "${RUN_FULL_BASELINE:-0}" != "1" ]; then
    FINAL_EXTRA_ARGS+=(--skip-full-baseline)
fi

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
    --num-workers 0 \
    --data-dir "$DATA_DIR" \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-features \
    "${FINAL_EXTRA_ARGS[@]}" \
    --fail-fast \
    --experiment-name "$EXPERIMENT_NAME"

STATUS=$?

echo
echo "============================================================"
echo "Final test evaluation finished"
echo "End time: $(date)"
echo "Exit code: ${STATUS}"
echo "============================================================"

exit "$STATUS"
