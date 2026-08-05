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
export TORCH_HOME=/home/hpc/iwi5/iwi5419h/torch_cache

# Project root on the cluster — edit here if the deployment moves. All versions
# must use the same directory so they share one train/test split.
PROJECT_DIR=/home/hpc/iwi5/iwi5419h/vase_project/vase_urn_project
export VASE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "=== V1: Raw ResNet50 Baseline ==="

echo "=== STEP 1: Extract Features ==="
python v1_baseline/extract_features.py --output-dir runs/v1

echo "=== STEP 2: Retrieve (full ranking, no model) ==="
python v1_baseline/retrieve.py --input-dir runs/v1

echo "=== STEP 3: Evaluate ==="
python v1_baseline/evaluate.py --input-dir runs/v1

echo "=== V1 DONE ==="
