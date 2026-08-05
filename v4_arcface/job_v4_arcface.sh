#!/bin/bash -l
#SBATCH --job-name=v4-arcface
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=v4_arcface_%j.out
#SBATCH --error=v4_arcface_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval
export TORCH_HOME=/home/hpc/iwi5/iwi5419h/torch_cache

# Project root on the cluster — edit here if the deployment moves. All versions
# must use the same directory so they share one train/test split.
PROJECT_DIR=/home/hpc/iwi5/iwi5419h/vase_project/vase_urn_project
export VASE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "=== V4: ResNet50 + ArcFace + SAM Crop ==="

echo "=== STEP 1: Detect & Crop Figures (SAM) ==="
python v4_arcface/detect_crop.py

echo "=== STEP 2: Extract Features (from crops) ==="
python v4_arcface/extract_features.py

echo "=== STEP 3: Train (ArcFace Loss) ==="
python v4_arcface/train.py

echo "=== STEP 4: Retrieve ==="
python v4_arcface/retrieve.py --input-dir runs/v4 --model-path runs/v4/model.pth

echo "=== STEP 5: Evaluate ==="
python v4_arcface/evaluate.py

echo "=== V4 DONE ==="
