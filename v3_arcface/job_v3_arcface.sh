#!/bin/bash -l
#SBATCH --job-name=v3-arcface
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=v3_arcface_%j.out
#SBATCH --error=v3_arcface_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval

cd /home/woody/iwi5/iwi5419h/vase_project

echo "=== V3: ResNet50 + ArcFace + SAM Crop ==="

echo "=== STEP 1: Detect & Crop Figures (SAM) ==="
python v3_arcface/detect_crop.py

echo "=== STEP 2: Extract Features (from crops) ==="
python v3_arcface/extract_features.py

echo "=== STEP 3: Train (ArcFace Loss) ==="
python v3_arcface/train.py

echo "=== STEP 4: Retrieve ==="
python v3_arcface/retrieve.py

echo "=== STEP 5: Evaluate ==="
python v3_arcface/evaluate.py

echo "=== V3 DONE ==="
