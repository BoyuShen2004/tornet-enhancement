#!/bin/bash
#SBATCH --job-name=tornet-enhanced-eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

echo ">>> Starting Enhanced TorNet Evaluation..."
echo ">>> Start time: $(date)"
START_TIME=$SECONDS

source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate tornet

cd /projects/weilab/shenb/csci3370/tornet_enhanced

export TORNET_ROOT=/projects/weilab/shenb/csci3370/data
export KERAS_BACKEND=torch

# Optional model path (defaults to epoch 8 checkpoint under an example run)
MODEL_PATH_ARG="$1"
DEFAULT_MODEL_PATH="/projects/weilab/shenb/csci3370/tornet_enhanced/tornado_enhanced_latest/checkpoints/tornadoDetector_008.keras"
MODEL_PATH="${MODEL_PATH_ARG:-$DEFAULT_MODEL_PATH}"

echo "=========================================="
echo "Keras backend: $KERAS_BACKEND"
echo "TORNET_ROOT:   $TORNET_ROOT"
echo "MODEL_PATH:    $MODEL_PATH"
echo "GPU info:"
nvidia-smi || true
echo "=========================================="

python scripts/tornado_detection/test_enhanced_tornet_keras.py \
       "$MODEL_PATH"

DURATION=$((SECONDS - START_TIME))
echo "Job completed at $(date)"
echo "Total runtime: $((DURATION / 60)) min $((DURATION % 60)) sec"
echo "Evaluation completed successfully!"
echo "Results saved alongside: $MODEL_PATH"


