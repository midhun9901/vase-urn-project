# Experiment Results

**Project:** CBIR for archaeological vase retrieval  
**Supervisor:** Mathias Zinnen (FAU)  
**Branch:** `v5-dinov2-retrieval`  
**Last updated:** September 4, 2026

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
| V1 - Baseline | Raw ResNet50 features, no MLP | RTX 3050 | 15.51% | 15.09% | 30.19% |
| V2 - ResNet50 + Triplet | ResNet50 + MLP + Triplet Loss | RTX 3050 | 50.70% | 43.40% | 84.91% |
| V3 - ResNet50 + ProxyAnchor | ResNet50 + MLP + ProxyAnchor Loss | RTX 3050 | 58.87% | 52.83% | 88.68% |
| V4 - SAM crop + ArcFace ⚠️ | SAM crop + ResNet50 + ArcFace Loss | A100 (TinyGPU) | 11.55% | 9.62% | 30.77% |
| V5 - SAM crop + Triplet ⚠️ | SAM crop + ResNet50 + Triplet Loss | A100 (TinyGPU) | 11.55% | 9.62% | 30.77% |
| V6a - DINOv2 Full Image | DINOv2 ViT-S/14 + FAISS full ranking | RTX 3050 | 32.66% | 26.42% | 83.02% |
| V6b - DINOv2 + Triplet | DINOv2 ViT-S/14 + MLP + Triplet Loss | RTX 3050 | **78.26%** | **77.36%** | **100.00%** |
| V7 - Generic Pose + Triplet | Off-the-shelf YOLOv8 (COCO human) keypoints + Triplet Loss | RTX 3050 | 36.93% | 33.96% | 67.92% |
| V7b - Vase Figure-Pose + Triplet ⚠️ | YOLOv11 fine-tuned on vase keypoints + Triplet Loss | RTX 3080 (TinyGPU) | 45.55% | 38.46% | 67.31% |

> ⚠️ **V4/V5 results are invalid and must be re-run** (July 10, 2026): the original
> job scripts called `retrieve.py` without `--model-path`, so the trained
> ArcFace/Triplet heads were never applied — the numbers above measure *raw*
> ResNet50 features on SAM crops, which is also why V4 and V5 are identical.
> The job scripts now pass `--model-path`; re-run both to get real numbers.

> ⚠️ **V7b is diagnostic and must be re-run** (September 3, 2026): the original
> 238-image pose pool overlapped the retrieval test set. The audit found 51
> byte-identical test images and one additional same-named counterpart; 44 of
> the 52 overlaps entered pose training and eight entered pose validation.
> `v7_pose/train_pose.py` now excludes hash or filename matches before creating
> the pose train/validation split. The recorded 45.55% mAP is not a clean
> held-out result and must not be used for comparative claims.

The clean V1/V2/V3/V6a/V6b/V7 rows above were rerun on September 4 after
adding `.bmp` support. They use all 253 supplied vase images: 200 training
images and 53 test images across the unchanged 80/21 folder split. V4, V5,
and V7b remain historical diagnostics from the earlier 250-image loader.

Main result:

> Among the BMP-complete clean held-out runs, V6b DINOv2 + Triplet Loss is the
> best method with 78.26% mAP and 100% Accuracy@10.

Key finding:

> Raw (untrained) features on SAM crops (V4/V5 as run) score below even the
> full-image baseline, suggesting cropping removes global context that helps
> retrieval — but whether cropping hurts *trained* models is open until the
> V4/V5 re-run. All versions use the same full-ranking evaluator, and all
> clean versions were rerun with the same full-ranking evaluator and must be
> run from the same project directory (same
> `train_folders.txt`/`test_folders.txt`) to be directly comparable.

---

## V6 Files

| File/Folder | Purpose |
|-------------|---------|
| `v6_dinov2/` | DINOv2 feature extraction, training, retrieval, evaluation |
| `v6_dinov2/job_v6_dinov2.sh` | TinyGPU SLURM job for V6a and V6b |
| `runs/bmp_fix/dinov2_full/` | BMP-complete V6a artifacts (gitignored) |
| `runs/bmp_fix/dinov2_triplet/` | BMP-complete V6b artifacts and model (gitignored) |

The other BMP-complete artifacts and trained projections are in
`runs/bmp_fix/v1`, `runs/bmp_fix/v2`, `runs/bmp_fix/v3`, and
`runs/bmp_fix/pose_generic`. V1 and V6a have no trained projection weights.

---

## V6a - DINOv2 Full Image

Pipeline:

```text
image -> DINOv2 ViT-S/14 feature -> L2 normalize -> FAISS -> evaluate
```

Result:

```text
mAP        : 0.3266
Accuracy@1 : 26.42%
Accuracy@10: 83.02%
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
mAP        : 0.7826
Accuracy@1 : 77.36%
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
