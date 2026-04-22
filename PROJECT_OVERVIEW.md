# Vase Retrieval - V5 DINOv2 Branch Overview

**Type:** ML Research - Content-Based Image Retrieval  
**Author:** Midhun Somanunnithan  
**Supervisor:** Mathias Zinnen  
**Branch:** `v5-dinov2-retrieval`  
**Deadline:** June 12, 2026

---

## What This Branch Is

This is the clean DINOv2 branch.

It does **not** use SAM.

The SAM experiment is preserved on:

```text
v4-sam-triplet
```

V5 uses full original images and improves the feature extractor:

```text
image
        -> DINOv2 feature
        -> Triplet Loss MLP
        -> FAISS retrieval
        -> evaluation
```

---

## Best Result

| Method | Hardware | mAP | Acc@1 | Acc@10 |
|--------|----------|-----|-------|--------|
| V5b - DINOv2 + Triplet | TinyGPU A100 | **83.39%** | **80.77%** | **100.00%** |

Previous best:

| Method | Hardware | mAP | Acc@1 | Acc@10 |
|--------|----------|-----|-------|--------|
| V2 - ResNet50 + Triplet | TinyGPU A100 | 54.04% | 42.31% | 92.31% |

---

## Why V5 Improved Results

SAM tried to improve retrieval by cropping images, but cropping removed useful context.

DINOv2 keeps the full image and provides stronger visual features than ResNet50. Triplet Loss then learns a vase-specific embedding space from those features.

Main idea:

```text
Better full-image features > automatic cropping for this dataset
```

---

## Important V5 Files

| File/Folder | Purpose |
|-------------|---------|
| `pipeline_dinov2/common.py` | Shared dataset, split, normalization, metric helpers |
| `pipeline_dinov2/extract_features.py` | Extracts DINOv2 features from full images |
| `pipeline_dinov2/train.py` | Trains MLP with Triplet Loss on DINOv2 features |
| `pipeline_dinov2/retrieve.py` | Builds FAISS index and retrieves full ranking |
| `pipeline_dinov2/evaluate.py` | Computes mAP, Acc@1, Acc@10 |
| `pipeline_dinov2/cache_model.py` | Caches DINOv2 for offline compute nodes |
| `job_v5_dinov2.sh` | TinyGPU job script |
| `demo_v5_dinov2.py` | Visual V5 demo |
| `demo_all_versions.py` | Visual comparison across saved versions |

---

## Run V5 on TinyGPU

```bash
cd /home/woody/iwi5/iwi5419h/vase_project
sbatch.tinygpu job_v5_dinov2.sh
```

Outputs:

```text
runs/dinov2_full/
runs/dinov2_triplet/
```

---

## Demo

Run locally:

```powershell
cd C:\PROJECTS\vase_urn_project
python demo_all_versions.py
```

Or V5 only:

```powershell
python demo_v5_dinov2.py
```

---

## Branch vs Version

A branch is a Git workspace:

```text
v4-sam-triplet
v5-dinov2-retrieval
```

A version is an experiment/method:

```text
V4 = SAM + Triplet
V5 = DINOv2 + Triplet
```

This branch contains the V5 method. The V4 method is preserved on the V4 branch.
