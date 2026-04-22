#!/bin/bash -l
#SBATCH --job-name=vase-retrieval
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=vase_%j.out
#SBATCH --error=vase_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval

cd /home/woody/iwi5/iwi5419h/vase_project

echo "=== STEP 1: Split Data ==="
python split_data.py

echo "=== STEP 2: Detect & Crop Figures (SAM) ==="
python detect_crop.py

echo "=== STEP 3: Extract Features (from crops) ==="
python extract_features.py

echo "=== STEP 4: Train (Triplet Loss) ==="
python train.py

echo "=== STEP 4: Retrieve ==="
python retrieve.py

echo "=== STEP 5: Evaluate ==="
python evaluate.py

echo "=== ALL DONE ==="
