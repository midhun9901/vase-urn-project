#!/bin/bash -l
#SBATCH --job-name=v7-pose
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=v7_pose_%j.out
#SBATCH --error=v7_pose_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval
export TORCH_HOME=/home/woody/iwi5/iwi5419h/torch_cache

# Project root on the cluster — edit here if the deployment moves. All versions
# must use the same directory so they share one train/test split.
PROJECT_DIR=/home/hpc/iwi5/iwi5419h/vase_project/vase_urn_project
export VASE_PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "=== V7: YOLOv8 pose keypoints + Triplet Loss ==="
python v7_pose/extract_features.py --output-dir runs/pose_triplet --model-name yolov8n-pose.pt
python v7_pose/train.py --input-dir runs/pose_triplet --output-dir runs/pose_triplet
python v7_pose/retrieve.py --input-dir runs/pose_triplet --model-path runs/pose_triplet/model.pth
python v7_pose/evaluate.py --input-dir runs/pose_triplet

echo "=== V7 Pose DONE ==="
