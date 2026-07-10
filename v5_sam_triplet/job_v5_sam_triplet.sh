#!/bin/bash -l
#SBATCH --job-name=v5-sam-triplet
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=v5_sam_triplet_%j.out
#SBATCH --error=v5_sam_triplet_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval
export TORCH_HOME=/home/hpc/iwi5/iwi5419h/torch_cache

cd /home/hpc/iwi5/iwi5419h/vase_project/vase_urn_project

echo "=== V5: ResNet50 + SAM Crop + Triplet Loss ==="

echo "=== STEP 1: Detect & Crop Figures (SAM) ==="
python v5_sam_triplet/detect_crop.py

echo "=== STEP 2: Extract Features (from crops) ==="
python v5_sam_triplet/extract_features.py

echo "=== STEP 3: Train (Triplet Loss) ==="
python v5_sam_triplet/train.py

echo "=== STEP 4: Retrieve ==="
python v5_sam_triplet/retrieve.py

echo "=== STEP 5: Evaluate ==="
python v5_sam_triplet/evaluate.py

echo "=== V5 DONE ==="
