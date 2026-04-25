#!/bin/bash -l
#SBATCH --job-name=vase-all-v1-v5
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=8:00:00
#SBATCH --output=all_v1_v5_%j.out
#SBATCH --error=all_v1_v5_%j.err
#SBATCH --export=NONE

set -e

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval

cd /home/woody/iwi5/iwi5419h/vase_project

# ─────────────────────────────────────────────
# V1 — Raw ResNet50, no metric learning
# ─────────────────────────────────────────────
echo "=== V1: Raw ResNet50 Baseline ==="

echo "--- V1 STEP 1: Extract Features ---"
python v1_baseline/extract_features.py --output-dir runs/v1

echo "--- V1 STEP 2: Retrieve (full ranking, no model) ---"
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

echo ""
echo "=== ALL VERSIONS COMPLETE ==="
echo "Results saved to:"
echo "  runs/v1/metrics.json"
echo "  runs/v2/metrics.json"
echo "  runs/v3/metrics.json"
echo "  runs/v4/metrics.json"
echo "  runs/v5/metrics.json"
