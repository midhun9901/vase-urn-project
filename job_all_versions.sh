#!/bin/bash -l
#SBATCH --job-name=vase-all-versions
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=16:00:00
#SBATCH --output=all_versions_%j.out
#SBATCH --error=all_versions_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval
export TORCH_HOME=/home/woody/iwi5/iwi5419h/torch_cache

cd /home/woody/iwi5/iwi5419h/vase_project

# ─────────────────────────────────────────────
# V1 — Raw ResNet50, no metric learning
# ─────────────────────────────────────────────
echo "=== V1: Raw ResNet50 Baseline ==="

echo "--- V1 STEP 1: Extract Features ---"
python v1_baseline/extract_features.py --output-dir runs/v1

echo "--- V1 STEP 2: Retrieve ---"
python v1_baseline/retrieve.py --input-dir runs/v1

echo "--- V1 STEP 3: Evaluate ---"
python v1_baseline/evaluate.py --input-dir runs/v1

echo "=== V1 DONE ==="

# ─────────────────────────────────────────────
# V2 — ResNet50 + Triplet Loss
# ─────────────────────────────────────────────
echo "=== V2: ResNet50 + Triplet Loss ==="

echo "--- V2 STEP 1: Extract Features ---"
python v2_resnet_triplet/extract_features.py --output-dir runs/v2

echo "--- V2 STEP 2: Train (Triplet Loss) ---"
python v2_resnet_triplet/train.py --input-dir runs/v2 --output-dir runs/v2

echo "--- V2 STEP 3: Retrieve ---"
python v2_resnet_triplet/retrieve.py --input-dir runs/v2 --model-path runs/v2/model.pth

echo "--- V2 STEP 4: Evaluate ---"
python v2_resnet_triplet/evaluate.py --input-dir runs/v2

echo "=== V2 DONE ==="

# ─────────────────────────────────────────────
# V3 — ResNet50 + ProxyAnchor Loss
# ─────────────────────────────────────────────
echo "=== V3: ResNet50 + ProxyAnchor Loss ==="

echo "--- V3 STEP 1: Extract Features ---"
python v3_proxyanchor/extract_features.py --output-dir runs/v3

echo "--- V3 STEP 2: Train (ProxyAnchor Loss) ---"
python v3_proxyanchor/train.py --input-dir runs/v3 --output-dir runs/v3

echo "--- V3 STEP 3: Retrieve ---"
python v3_proxyanchor/retrieve.py --input-dir runs/v3 --model-path runs/v3/model.pth

echo "--- V3 STEP 4: Evaluate ---"
python v3_proxyanchor/evaluate.py --input-dir runs/v3

echo "=== V3 DONE ==="

# ─────────────────────────────────────────────
# V4 — ResNet50 + SAM Crop + ArcFace Loss
# ─────────────────────────────────────────────
echo "=== V4: ResNet50 + SAM Crop + ArcFace Loss ==="

echo "--- V4 STEP 1: Detect & Crop Figures (SAM) ---"
python v4_arcface/detect_crop.py --crops-dir runs/v4/crops

echo "--- V4 STEP 2: Extract Features (from crops) ---"
python v4_arcface/extract_features.py --output-dir runs/v4 --crops-dir runs/v4/crops

echo "--- V4 STEP 3: Train (ArcFace Loss) ---"
python v4_arcface/train.py --input-dir runs/v4 --output-dir runs/v4

echo "--- V4 STEP 4: Retrieve ---"
python v4_arcface/retrieve.py --input-dir runs/v4 --model-path runs/v4/model.pth

echo "--- V4 STEP 5: Evaluate ---"
python v4_arcface/evaluate.py --input-dir runs/v4

echo "=== V4 DONE ==="

# ─────────────────────────────────────────────
# V5 — ResNet50 + SAM Crop + Triplet Loss
# ─────────────────────────────────────────────
echo "=== V5: ResNet50 + SAM Crop + Triplet Loss ==="

echo "--- V5 STEP 1: Detect & Crop Figures (SAM) ---"
python v5_sam_triplet/detect_crop.py --crops-dir runs/v5/crops

echo "--- V5 STEP 2: Extract Features (from crops) ---"
python v5_sam_triplet/extract_features.py --output-dir runs/v5 --crops-dir runs/v5/crops

echo "--- V5 STEP 3: Train (Triplet Loss) ---"
python v5_sam_triplet/train.py --input-dir runs/v5 --output-dir runs/v5

echo "--- V5 STEP 4: Retrieve ---"
python v5_sam_triplet/retrieve.py --input-dir runs/v5 --model-path runs/v5/model.pth

echo "--- V5 STEP 5: Evaluate ---"
python v5_sam_triplet/evaluate.py --input-dir runs/v5

echo "=== V5 DONE ==="

# ─────────────────────────────────────────────
# V6 — DINOv2 (full-image + triplet fine-tune)
# ─────────────────────────────────────────────
echo "=== V6A: DINOv2 full-image retrieval ==="
python v6_dinov2/extract_features.py --output-dir runs/dinov2_full --model-name dinov2_vits14
python v6_dinov2/retrieve.py --input-dir runs/dinov2_full
python v6_dinov2/evaluate.py --input-dir runs/dinov2_full

echo "=== V6B: DINOv2 + Triplet fine-tune ==="
mkdir -p runs/dinov2_triplet
cp runs/dinov2_full/train_embeddings.npy  runs/dinov2_triplet/
cp runs/dinov2_full/train_labels.npy      runs/dinov2_triplet/
cp runs/dinov2_full/test_embeddings.npy   runs/dinov2_triplet/
cp runs/dinov2_full/test_labels.npy       runs/dinov2_triplet/
cp runs/dinov2_full/train_image_paths.txt runs/dinov2_triplet/
cp runs/dinov2_full/test_image_paths.txt  runs/dinov2_triplet/
cp runs/dinov2_full/model_name.txt        runs/dinov2_triplet/
python v6_dinov2/train.py    --input-dir runs/dinov2_triplet --output-dir runs/dinov2_triplet
python v6_dinov2/retrieve.py --input-dir runs/dinov2_triplet --model-path runs/dinov2_triplet/model.pth
python v6_dinov2/evaluate.py --input-dir runs/dinov2_triplet

echo "=== V6 DONE ==="

# ─────────────────────────────────────────────
# V7 — YOLOv8 Pose + Skeleton Keypoints + Triplet
# ─────────────────────────────────────────────
echo "=== V7: Pose Estimation (YOLOv8 + Skeleton Keypoints + Triplet) ==="

YOLO_WEIGHTS=/home/woody/iwi5/iwi5419h/vase_project/yolov8x-pose.pt
if [ ! -f "$YOLO_WEIGHTS" ]; then
    echo "Downloading YOLOv8x-pose weights..."
    python -c "from ultralytics import YOLO; YOLO('yolov8x-pose.pt')"
    cp ~/.config/Ultralytics/yolov8x-pose.pt "$YOLO_WEIGHTS" 2>/dev/null || \
    find ~/.ultralytics -name 'yolov8x-pose.pt' -exec cp {} "$YOLO_WEIGHTS" \; 2>/dev/null || \
    echo "Warning: could not cache weights — will re-download next run"
fi

echo "--- V7 STEP 1: Detect Poses ---"
python v7_pose/detect_pose.py --output-dir runs/v7 --model-path "$YOLO_WEIGHTS"

echo "--- V7 STEP 2: Extract Normalized Keypoint Features ---"
python v7_pose/extract_features.py --input-dir runs/v7 --output-dir runs/v7

echo "--- V7 STEP 3: Train (Triplet Loss on Pose Features) ---"
python v7_pose/train.py --input-dir runs/v7 --output-dir runs/v7

echo "--- V7 STEP 4: Retrieve ---"
python v7_pose/retrieve.py --input-dir runs/v7 --model-path runs/v7/model.pth

echo "--- V7 STEP 5: Evaluate ---"
python v7_pose/evaluate.py --input-dir runs/v7

echo "=== V7 DONE ==="

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo ""
echo "=== ALL VERSIONS COMPLETE ==="
echo "Results saved to:"
echo "  runs/v1/metrics.json           (V1: Raw ResNet50)"
echo "  runs/v2/metrics.json           (V2: ResNet50 + Triplet)"
echo "  runs/v3/metrics.json           (V3: ResNet50 + ProxyAnchor)"
echo "  runs/v4/metrics.json           (V4: ResNet50 + SAM + ArcFace)"
echo "  runs/v5/metrics.json           (V5: ResNet50 + SAM + Triplet)"
echo "  runs/dinov2_full/metrics.json  (V6A: DINOv2 full-image)"
echo "  runs/dinov2_triplet/metrics.json (V6B: DINOv2 + Triplet)"
echo "  runs/v7/metrics.json           (V7: YOLOv8 Pose + Triplet)"
