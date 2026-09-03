#!/bin/bash

#SBATCH --job-name=TFM_Extract
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH -w hera
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-extract-%x-%j.out
#SBATCH --error=slurm-extract-%x-%j.err

# Stop on command failures, undefined variables, and pipeline failures.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 DATASET MODEL"
    echo "Example: sbatch $0 MNIST alexnet"
    exit 2
fi

DATASET="${1^^}"
MODEL="${2,,}"

case "$DATASET" in
    MNIST|CIFAR10|TINYIMAGENET)
        ;;
    *)
        echo "ERROR: Unsupported dataset: $DATASET"
        exit 2
        ;;
esac

case "$MODEL" in
    alexnet)
        EXTRACTION_BATCH_SIZE=64
        ;;
    resnext)
        EXTRACTION_BATCH_SIZE=32
        ;;
    efficientnetv2)
        EXTRACTION_BATCH_SIZE=16
        ;;
    swintransformer)
        EXTRACTION_BATCH_SIZE=32
        ;;
    *)
        echo "ERROR: Unsupported model: $MODEL"
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

cd "$PROJECT_DIR"

echo "============================================================"
echo "TFM frozen-feature extraction"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Extraction batch size: ${EXTRACTION_BATCH_SIZE}"
echo "Start time: $(date)"
echo "============================================================"

nvidia-smi

python -u extract_features.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --features-dir "$FEATURES_DIR" \
    --batch-size "$EXTRACTION_BATCH_SIZE" \
    --num-workers 4 \
    --device cuda

echo
echo "============================================================"
echo "Feature extraction completed"
echo "End time: $(date)"
echo "Output: ${FEATURES_DIR}/${DATASET}/${MODEL}"
echo "============================================================"
