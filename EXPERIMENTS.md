# Experiment Results

**Project:** CBIR for archaeological vase retrieval  
**Supervisor:** Mathias Zinnen (FAU)  
**Branch:** `v5-dinov2-retrieval`  
**Last updated:** July 10, 2026

---

## Version Overview

| Folder | Version | Description |
|--------|---------|-------------|
| `v1_baseline/` | V1 | Raw ResNet50, no metric learning |
| `v2_resnet_triplet/` | V2 | ResNet50 + MLP + Triplet Loss |
| `v3_proxyanchor/` | V3 | ResNet50 + MLP + ProxyAnchor Loss |
| `v4_arcface/` | V4 | SAM crop + ResNet50 + ArcFace Loss |
| `v5_sam_triplet/` | V5 | SAM crop + ResNet50 + Triplet Loss |
| `v6_dinov2/` | V6 ★ best | DINOv2 ViT-S/14 + MLP + Triplet Loss |
| `v7_pose/` | V7 | Pose keypoints + Triplet Loss (generic COCO → vase-tuned V7b) |

---

## Results

| Version | Description | Hardware | mAP | Acc@1 | Acc@10 |
|---------|-------------|----------|-----|-------|--------|
| V1 - Baseline | Raw ResNet50 features, no MLP | A100 (TinyGPU) | 15.51% | 15.38% | 28.85% |
| V2 - ResNet50 + Triplet | ResNet50 + MLP + Triplet Loss | A100 (TinyGPU) | 54.79% | 44.23% | 96.15% |
| V3 - ResNet50 + ProxyAnchor | ResNet50 + MLP + ProxyAnchor Loss | A100 (TinyGPU) | 57.43% | 44.23% | 96.15% |
| V4 - SAM crop + ArcFace ⚠️ | SAM crop + ResNet50 + ArcFace Loss | A100 (TinyGPU) | 11.55% | 9.62% | 30.77% |
| V5 - SAM crop + Triplet ⚠️ | SAM crop + ResNet50 + Triplet Loss | A100 (TinyGPU) | 11.55% | 9.62% | 30.77% |
| V6a - DINOv2 Full Image | DINOv2 ViT-S/14 + FAISS full ranking | A100 (TinyGPU) | 32.53% | 26.92% | 82.69% |
| V6b - DINOv2 + Triplet | DINOv2 ViT-S/14 + MLP + Triplet Loss | A100 (TinyGPU) | **83.39%** | **80.77%** | **100.00%** |
| V7 - Generic Pose + Triplet | Off-the-shelf YOLOv8 (COCO human) keypoints + Triplet Loss | A100 (TinyGPU) | 39.76% | 38.46% | 57.69% |
| V7b - Vase Figure-Pose + Triplet | YOLOv11 fine-tuned on vase keypoints + Triplet Loss | RTX 3080 (TinyGPU) | 45.55% | 38.46% | 67.31% |

> ⚠️ **V4/V5 results are invalid and must be re-run** (July 10, 2026): the original
> job scripts called `retrieve.py` without `--model-path`, so the trained
> ArcFace/Triplet heads were never applied — the numbers above measure *raw*
> ResNet50 features on SAM crops, which is also why V4 and V5 are identical.
> The job scripts now pass `--model-path`; re-run both to get real numbers.

Main result:

> V6b DINOv2 + Triplet Loss is the best method with 83.39% mAP and 100% Accuracy@10.

Key finding:

> Raw (untrained) features on SAM crops (V4/V5 as run) score below even the
> full-image baseline, suggesting cropping removes global context that helps
> retrieval — but whether cropping hurts *trained* models is open until the
> V4/V5 re-run. All versions use the same full-ranking evaluator, and all
> versions must be run from the same project directory (same
> `train_folders.txt`/`test_folders.txt`) to be directly comparable.

---

## V6 Files

| File/Folder | Purpose |
|-------------|---------|
| `v6_dinov2/` | DINOv2 feature extraction, training, retrieval, evaluation |
| `v6_dinov2/job_v6_dinov2.sh` | TinyGPU SLURM job for V6a and V6b |
| `runs/dinov2_full/` | V6a generated artifacts (gitignored) |
| `runs/dinov2_triplet/` | V6b generated artifacts (gitignored) |

---

## V6a - DINOv2 Full Image

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

> DINOv2 features alone do not beat ResNet50 + Triplet. The backbone is stronger, but without fine-tuning the features are not task-specific.

---

## V6b - DINOv2 + Triplet

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

Compute nodes have no internet, so model weights must be pre-cached from the
login node first:

```text
Cache location: /home/hpc/iwi5/iwi5419h/torch_cache
```

Submit V6 (the job script `cd`s into `PROJECT_DIR` itself — check that the
`PROJECT_DIR` value at the top of the script matches the actual deployment):

```bash
sbatch.tinygpu v6_dinov2/job_v6_dinov2.sh
```
