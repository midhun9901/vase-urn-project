#!/bin/bash -l
#SBATCH --job-name=v2-resnet-triplet
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=v2_resnet_triplet_%j.out
#SBATCH --error=v2_resnet_triplet_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval

cd /home/woody/iwi5/iwi5419h/vase_project

echo "=== V2: ResNet50 + Triplet Loss ==="

echo "=== STEP 1: Extract Features ==="
python v2_resnet_triplet/extract_features.py --output-dir runs/v2

echo "=== STEP 2: Train (Triplet Loss) ==="
python v2_resnet_triplet/train.py --input-dir runs/v2 --output-dir runs/v2

echo "=== STEP 3: Retrieve ==="
python v2_resnet_triplet/retrieve.py --input-dir runs/v2 --model-path runs/v2/model.pth

echo "=== STEP 4: Evaluate ==="
python v2_resnet_triplet/evaluate.py --input-dir runs/v2

echo "=== V2 DONE ==="
