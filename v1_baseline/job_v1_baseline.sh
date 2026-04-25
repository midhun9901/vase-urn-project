#!/bin/bash -l
#SBATCH --job-name=v1-baseline
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=1:00:00
#SBATCH --output=v1_baseline_%j.out
#SBATCH --error=v1_baseline_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval

cd /home/woody/iwi5/iwi5419h/vase_project

echo "=== V1: Raw ResNet50 Baseline ==="

echo "=== STEP 1: Extract Features ==="
python v1_baseline/extract_features.py

echo "=== STEP 2: Retrieve (baseline) ==="
python v1_baseline/retrieve_baseline.py

echo "=== STEP 3: Evaluate ==="
python v1_baseline/evaluate_baseline.py

echo "=== V1 DONE ==="
