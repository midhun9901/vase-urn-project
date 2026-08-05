#!/bin/bash -l
#SBATCH --job-name=v7-pose
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=4:00:00
#SBATCH --output=v7_pose_%j.out
#SBATCH --error=v7_pose_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval
export TORCH_HOME=/home/hpc/iwi5/iwi5419h/torch_cache

cd /home/hpc/iwi5/iwi5419h/vase_project/vase_urn_project

echo "=== V7: vase Figure-Pose keypoints + Triplet Loss ==="

# 1. Fine-tune a YOLO pose model on the vase keypoint annotations
python v7_pose/train_pose.py --output-dir runs/pose_model

# 2. Extract real figure keypoints (add --crop-field to focus on the image field)
python v7_pose/extract_features.py --output-dir runs/pose_triplet \
    --pose-weights runs/pose_model/pose_best.pt

# 3. Train the MLP + Triplet projection on the keypoint features
python v7_pose/train.py --input-dir runs/pose_triplet --output-dir runs/pose_triplet

# 4. Retrieve and evaluate
python v7_pose/retrieve.py --input-dir runs/pose_triplet --model-path runs/pose_triplet/model.pth
python v7_pose/evaluate.py --input-dir runs/pose_triplet

echo "=== V7 Pose DONE ==="
