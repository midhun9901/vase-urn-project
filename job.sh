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

echo "=== STEP 2: Extract Features ==="
python extract_features.py

echo "=== BASELINE (No Metric Learning) ==="
python pipeline_baseline/retrieve.py
python pipeline_baseline/evaluate.py

echo "=== METRIC LEARNING (Triplet Loss) ==="
python pipeline_metric_learning/train.py
python pipeline_metric_learning/retrieve.py
python pipeline_metric_learning/evaluate.py

echo "=== ALL DONE ==="
