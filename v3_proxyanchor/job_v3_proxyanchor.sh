#!/bin/bash -l
#SBATCH --job-name=v3-proxyanchor
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=v3_proxyanchor_%j.out
#SBATCH --error=v3_proxyanchor_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval
export TORCH_HOME=/home/hpc/iwi5/iwi5419h/torch_cache

cd /home/hpc/iwi5/iwi5419h/vase_project/vase_urn_project

echo "=== V3: ResNet50 + ProxyAnchor Loss ==="

echo "=== STEP 1: Extract Features (ResNet50) ==="
python v3_proxyanchor/extract_features.py --output-dir runs/proxyanchor

echo "=== STEP 2: Train (ProxyAnchor Loss) ==="
python v3_proxyanchor/train.py --input-dir runs/proxyanchor --output-dir runs/proxyanchor

echo "=== STEP 3: Retrieve ==="
python v3_proxyanchor/retrieve.py --input-dir runs/proxyanchor --model-path runs/proxyanchor/model.pth

echo "=== STEP 4: Evaluate ==="
python v3_proxyanchor/evaluate.py --input-dir runs/proxyanchor

echo "=== V3 DONE ==="
