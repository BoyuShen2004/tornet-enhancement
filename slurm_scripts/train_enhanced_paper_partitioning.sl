#!/bin/bash
#SBATCH --job-name=tornet-enhanced-paper-partitioning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --partition=medium
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shenb@bc.edu

# Enhanced TorNet Training Script - Building on Baseline Success

echo ">>> Starting Enhanced TorNet Training (Building on Baseline Success)..."
echo ">>> Job ID: $SLURM_JOB_ID"
echo ">>> Start time: $(date)"

# Set up environment
source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate tornet

# Set working directory (enhanced copy)
cd /projects/weilab/shenb/csci3370/tornet_enhanced

# Set environment variables
export TORNET_ROOT=/projects/weilab/shenb/csci3370/data
export KERAS_BACKEND=torch
export EXP_DIR=/projects/weilab/shenb/csci3370/tornet_enhanced/tornado_enhanced_$(date +%Y%m%d%H%M%S)-$SLURM_JOB_ID-None

# Create experiment directory
mkdir -p $EXP_DIR

echo ">>> TORNET_ROOT: $TORNET_ROOT"
echo ">>> KERAS_BACKEND: $KERAS_BACKEND"
echo ">>> EXP_DIR: $EXP_DIR"

# Create enhanced configuration
cat > $EXP_DIR/params_enhanced_tornet.json << 'EOF'
{
  "config": {
    "epochs": 20,
    "input_variables": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"],
    "train_years": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "val_years": [2021, 2022],
    "batch_size": 64,
    "model": "simple_enhanced_cnn",
    "start_filters": 16,
    "learning_rate": 0.001,
    "decay_steps": 1386,
    "decay_rate": 0.958,
    "l2_reg": 1e-05,
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
    "label_smooth": 0.0,
    "loss": "focal_imbalanced",
    "head": "maxpool",
    "exp_name": "tornado_enhanced_tornet",
    "exp_dir": ".",
    "dataloader": "keras",
    "dataloader_kwargs": {
      "select_keys": ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH", "range_folded_mask", "coordinates"]
    },
    "use_attention": false,
    "use_residual": true,
    "use_multiscale": false,
    "focal_alpha": 0.5,
    "focal_gamma": 1.5,
    "tversky_alpha": 0.7,
    "tversky_beta": 0.3,
    "use_augmentation": true,
    "use_class_balancing": true,
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.5,
    "use_cosine_annealing": true,
    "warmup_epochs": 3,
    "use_mixup": false,
    "use_cutmix": false,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "class_weight_method": "balanced",
    "oversample_ratio": 1.2,
    "focal_weight": 0.8,
    "dice_weight": 0.2,
    "class_balanced_beta": 0.9999,
    "data_partitioning": "julian_day_modulo",
    "julian_modulo": 20,
    "training_threshold": 17,
    "overlap_removal": true,
    "overlap_time_minutes": 30,
    "overlap_distance_degrees": 0.25,
    "rotation_range": 10.0,
    "translation_range": 0.05,
    "scale_range": [0.95, 1.05],
    "noise_std": 0.005,
    "brightness_range": 0.05,
    "contrast_range": [0.95, 1.05],
    "flip_probability": 0.3,
    "weight_decay": 1e-4,
    "use_multiscale": true,
    "use_attention": true,
    "use_residual": true
  }
}
EOF

echo ">>> Starting enhanced TorNet training ..."
python scripts/tornado_detection/train_enhanced_tornet_keras.py \
       $EXP_DIR/params_enhanced_tornet.json

echo ">>> Training completed"
echo ">>> End time: $(date)"


