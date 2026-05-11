#!/bin/bash -l
#SBATCH --job-name=v7-pose
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=3:00:00
#SBATCH --output=v7_pose_%j.out
#SBATCH --error=v7_pose_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval

cd /home/woody/iwi5/iwi5419h/vase_project

echo "=== V7: Pose Estimation (YOLOv8 + Skeleton Keypoints + Triplet) ==="

# Download YOLOv8x-pose weights if not already present
YOLO_WEIGHTS=/home/woody/iwi5/iwi5419h/vase_project/yolov8x-pose.pt
if [ ! -f "$YOLO_WEIGHTS" ]; then
    echo "Downloading YOLOv8x-pose weights..."
    python -c "from ultralytics import YOLO; YOLO('yolov8x-pose.pt')"
    cp ~/.config/Ultralytics/yolov8x-pose.pt "$YOLO_WEIGHTS" 2>/dev/null || \
    find ~/.ultralytics -name 'yolov8x-pose.pt' -exec cp {} "$YOLO_WEIGHTS" \; 2>/dev/null || \
    echo "Warning: could not cache weights — will re-download next run"
fi

echo "=== STEP 1: Detect Poses ==="
python v7_pose/detect_pose.py --output-dir runs/v7 --model-path "$YOLO_WEIGHTS"

echo "=== STEP 2: Extract Normalized Keypoint Features ==="
python v7_pose/extract_features.py --input-dir runs/v7 --output-dir runs/v7

echo "=== STEP 3: Train (Triplet Loss on Pose Features) ==="
python v7_pose/train.py --input-dir runs/v7 --output-dir runs/v7

echo "=== STEP 4: Retrieve ==="
python v7_pose/retrieve.py --input-dir runs/v7 --model-path runs/v7/model.pth

echo "=== STEP 5: Evaluate ==="
python v7_pose/evaluate.py --input-dir runs/v7

echo "=== V7 DONE ==="
echo "Results saved to: runs/v7/metrics.json"
