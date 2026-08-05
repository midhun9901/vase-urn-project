# Vase Retrieval (CBIR for Archaeological Vases)

Content-based image retrieval on historical vase illustrations (Hamilton engravings),
matching different depictions of the same vase across two dataset parts. Seven
experiment versions live side by side, each in its own folder with an identical
pipeline layout. See [EXPERIMENTS.md](EXPERIMENTS.md) for results.

## Layout

| Folder | Method |
|--------|--------|
| `v1_baseline/` | Raw ResNet50 features, no training |
| `v2_resnet_triplet/` | ResNet50 + MLP + Triplet Loss |
| `v3_proxyanchor/` | ResNet50 + MLP + ProxyAnchor Loss |
| `v4_arcface/` | SAM crop + ResNet50 + ArcFace Loss |
| `v5_sam_triplet/` | SAM crop + ResNet50 + Triplet Loss |
| `v6_dinov2/` | DINOv2 ViT-S/14 + MLP + Triplet Loss (best) |
| `v7_pose/` | Vase-tuned YOLOv11 Figure-Pose keypoints + Triplet Loss |

Each folder contains `extract_features.py` → (`train.py`) → `retrieve.py` →
`evaluate.py`, a `common.py` (identical in every folder so each folder stays
standalone — if you change one, sync all seven), and a SLURM job script.

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset under `data/`:

```text
data/pairs_pt1/Bildpaare Teil 1/<vase-id>/*.jpg
data/pairs_pt2/Bildpaare-Triplets Teil 2/<vase-id>/*.jpg
```

## Running locally

From the project root, e.g. V6:

```bash
python v6_dinov2/extract_features.py --output-dir runs/dinov2_triplet
python v6_dinov2/train.py --input-dir runs/dinov2_triplet --output-dir runs/dinov2_triplet
python v6_dinov2/retrieve.py --input-dir runs/dinov2_triplet --model-path runs/dinov2_triplet/model.pth
python v6_dinov2/evaluate.py --input-dir runs/dinov2_triplet
```

All outputs go to `runs/` (gitignored).

## The train/test split

The first extraction run creates `train_folders.txt` / `test_folders.txt` at the
project root (80/20 split at vase-folder level, seed 42, paths relative to the
project root). Every version reuses these files, so **all versions must run from
the same project directory** to be comparable. Delete both files to regenerate
the split.

## Running on the cluster (TinyGPU)

The job scripts set `PROJECT_DIR` near the top — verify it matches your
deployment before submitting, and keep it identical across all versions:

```bash
sbatch.tinygpu v6_dinov2/job_v6_dinov2.sh
```

Compute nodes have no internet access, so cache models first from a login node:

```bash
python v6_dinov2/cache_model.py --torch-home <your torch cache>   # DINOv2
```

V4/V5 additionally need the SAM checkpoint (`sam_vit_b.pth`); its location can
be overridden with the `SAM_CHECKPOINT` env var or `--sam-checkpoint`.
