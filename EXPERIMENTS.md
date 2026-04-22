# Experiment Results - V5 DINOv2 Branch

**Project:** CBIR for archaeological vase retrieval  
**Supervisor:** Mathias Zinnen (FAU)  
**Branch:** `v5-dinov2-retrieval`  
**Last updated:** April 22, 2026

---

## Branch Purpose

This branch is the clean V5 DINOv2 retrieval branch.

It does not use SAM, crop files, `detect_crop.py`, or `sam_vit_b.pth`.

V5 uses full original images:

```text
original image
        -> DINOv2 feature extractor
        -> optional Triplet Loss MLP
        -> FAISS retrieval
        -> full-ranking evaluation
```

The previous SAM experiment is preserved separately on:

```text
v4-sam-triplet
```

---

## Results

| Version | Description | Hardware | mAP | Acc@1 | Acc@10 |
|---------|-------------|----------|-----|-------|--------|
| V1 - Baseline | Raw ResNet50 features, no MLP | RTX 3050 (local artifacts) | 10.31% | 15.38% | 30.77% |
| V2 - ResNet50 + Triplet | ResNet50 features + MLP + Triplet Loss | RTX 3050 (local artifacts) | 44.67% | 34.62% | 90.38% |
| V2 - ResNet50 + Triplet | ResNet50 features + MLP + Triplet Loss | A100 GPU (historic best) | 54.04% | 42.31% | 92.31% |
| V5a - DINOv2 Full Image | DINOv2 features + FAISS full ranking | A100 GPU (TinyGPU) | 32.53% | 26.92% | 82.69% |
| V5b - DINOv2 + Triplet | DINOv2 features + MLP + Triplet Loss | A100 GPU (TinyGPU) | **83.39%** | **80.77%** | **100.00%** |

Main result:

> V5b DINOv2 + Triplet Loss is the best current method with 83.39% mAP.

Important fairness note:

V5 uses a corrected full-ranking evaluator. Before the final report, older ResNet methods should be re-evaluated with the same full-ranking evaluator for the cleanest final comparison.

---

## V5 Files

| File/Folder | Purpose |
|-------------|---------|
| `pipeline_dinov2/` | DINOv2 feature extraction, training, retrieval, evaluation |
| `job_v5_dinov2.sh` | TinyGPU SLURM job for V5a and V5b |
| `demo_v5_dinov2.py` | Visual demo for V5a vs V5b |
| `demo_all_versions.py` | Visual comparison of available saved result versions |
| `runs/dinov2_full/` | V5a generated artifacts |
| `runs/dinov2_triplet/` | V5b generated artifacts |

`runs/` is ignored by git because it contains generated experiment artifacts.

---

## V5a - DINOv2 Full Image

Pipeline:

```text
image -> DINOv2 ViT-S/14 feature -> L2 normalize -> FAISS -> evaluate
```

Result:

```text
mAP        : 0.3253
Accuracy@1 : 26.92%
Accuracy@10: 82.69%
```

Interpretation:

> DINOv2 features alone are useful but not enough to beat the previous ResNet50 + Triplet method.

---

## V5b - DINOv2 + Triplet

Pipeline:

```text
image -> DINOv2 ViT-S/14 feature -> MLP 384 -> 512 -> 128 -> Triplet Loss -> FAISS -> evaluate
```

Result:

```text
mAP        : 0.8339
Accuracy@1 : 80.77%
Accuracy@10: 100.00%
```

Interpretation:

> DINOv2 provides a stronger visual representation than ResNet50, and Triplet Loss successfully adapts those features to the vase retrieval task.

---

## TinyGPU Run

Valid V5 job:

```text
Job ID: 1581942
```

DINOv2 had to be cached on the login node because compute nodes could not access GitHub:

```text
/home/woody/iwi5/iwi5419h/torch_cache
```

Run:

```bash
cd /home/woody/iwi5/iwi5419h/vase_project
sbatch.tinygpu job_v5_dinov2.sh
```

---

## Professor Explanation

Short explanation:

> V5 removes the SAM crop step and returns to full original images. Instead of changing the input image, it improves the feature representation by replacing ResNet50 with DINOv2. DINOv2 alone achieved 32.53% mAP, but DINOv2 features combined with Triplet Loss achieved 83.39% mAP, making it the strongest current method.
