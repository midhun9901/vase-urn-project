#!/bin/bash -l
#SBATCH --job-name=v6-dinov2
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=v6_dinov2_%j.out
#SBATCH --error=v6_dinov2_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval
export TORCH_HOME=/home/woody/iwi5/iwi5419h/torch_cache

cd /home/woody/iwi5/iwi5419h/vase_project

echo "=== V6A: DINOv2 full-image retrieval ==="
python v6_dinov2/extract_features.py --output-dir runs/dinov2_full --model-name dinov2_vits14
python v6_dinov2/retrieve.py --input-dir runs/dinov2_full
python v6_dinov2/evaluate.py --input-dir runs/dinov2_full

echo "=== V6B: DINOv2 + Triplet retrieval ==="
mkdir -p runs/dinov2_triplet
cp runs/dinov2_full/train_embeddings.npy runs/dinov2_triplet/
cp runs/dinov2_full/train_labels.npy runs/dinov2_triplet/
cp runs/dinov2_full/test_embeddings.npy runs/dinov2_triplet/
cp runs/dinov2_full/test_labels.npy runs/dinov2_triplet/
cp runs/dinov2_full/train_image_paths.txt runs/dinov2_triplet/
cp runs/dinov2_full/test_image_paths.txt runs/dinov2_triplet/
cp runs/dinov2_full/model_name.txt runs/dinov2_triplet/
python v6_dinov2/train.py --input-dir runs/dinov2_triplet --output-dir runs/dinov2_triplet
python v6_dinov2/retrieve.py --input-dir runs/dinov2_triplet --model-path runs/dinov2_triplet/model.pth
python v6_dinov2/evaluate.py --input-dir runs/dinov2_triplet

echo "=== V6 DINOv2 DONE ==="
