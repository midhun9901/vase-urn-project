# Folder Structure Handover
**Project:** CBIR for Archaeological Vase Retrieval  
**Student:** Midhun Somanunnithan  
**Supervisor:** Mathias Zinnen (FAU)  
**Deadline:** June 12, 2026  
**Date of handover:** April 25, 2026  

---

## Current State Summary

The repo is currently on the `v5-dinov2-retrieval` branch with a partially clean structure.
A cluster job (`1583512`) may still be running on TinyGPU — check with `squeue -u iwi5419h`.

The user is on the `arcface` branch when this doc was written — switch back first:
```bash
git checkout v5-dinov2-retrieval
```

---

## Git Branches (All 5 Versions)

| Branch | Version | Description |
|---|---|---|
| `v1-branch` | V1 | Raw ResNet50 features, no metric learning |
| `master` | V2 | ResNet50 + MLP + Triplet Loss |
| `arcface` | V3 | ResNet50 + ArcFace Loss + SAM crop detection |
| `v4-sam-triplet` | V4 | ResNet50 + SAM crop + Triplet Loss (best SAM combo) |
| `v5-dinov2-retrieval` | V5 ★ ACTIVE | DINOv2 + MLP + Triplet Loss |

---

## Experiment Results

| Version | Description | mAP | Acc@1 | Acc@10 | Notes |
|---|---|---|---|---|---|
| V1 | Raw ResNet50, no training | 10.31% | 15.38% | 30.77% | Old evaluator (top-10 only) |
| V2 | ResNet50 + Triplet Loss | 54.04% | 42.31% | 92.31% | Old evaluator (top-10 only) |
| V3 | ResNet50 + ArcFace | unknown | unknown | unknown | Results not saved |
| V4 | SAM crop + Triplet Loss | unknown | unknown | unknown | Results not saved |
| V5a | DINOv2 full image only | 32.53% | 26.92% | 82.69% | Corrected full-ranking evaluator |
| V5b | DINOv2 + Triplet Loss | **83.39%** | **80.77%** | **100%** | Corrected full-ranking evaluator ★ best |

**Important:** V1 and V2 used the old evaluator (top-10 only). V5 uses the corrected full-ranking evaluator. They are not directly comparable. Re-evaluating V1 and V2 with the corrected evaluator is a pending task before final report.

---

## Current Folder Structure (v5-dinov2-retrieval branch)

```
vase_urn_project/
│
├── v1_baseline/                    ← V1 scripts (moved from pipeline_baseline/)
│   ├── evaluate.py
│   ├── evaluate_baseline.py
│   ├── retrieve.py
│   └── retrieve_baseline.py
│
├── v2_resnet_triplet/              ← V2 scripts (moved from pipeline_metric_learning/)
│   ├── extract_features.py
│   ├── train.py
│   ├── retrieve.py
│   └── evaluate.py
│
├── v5_dinov2/                      ← V5 scripts (ACTIVE) ★
│   ├── common.py                   ← shared utilities, project_root(), compute_metrics()
│   ├── extract_features.py         ← DINOv2 ViT-S/14 feature extraction
│   ├── train.py                    ← MLP 384→512→128 + Triplet Loss
│   ├── retrieve.py                 ← FAISS full-ranking retrieval
│   ├── evaluate.py                 ← mAP, Acc@1, Acc@10 + saves metrics.json
│   ├── cache_model.py              ← pre-downloads DINOv2 to cluster cache
│   └── README.md
│
├── tools/
│   ├── split_data.py               ← 80/20 train/test split (run ONCE only)
│   ├── hpc_manager.py              ← GUI to manage cluster jobs via SSH
│   └── local_manager.py            ← GUI to run pipeline locally
│
├── demo/
│   ├── demo.py                     ← main visual retrieval demo
│   ├── demo_v5_dinov2.py           ← V5a vs V5b comparison demo
│   └── demo_all_versions.py        ← compare all saved version results
│
├── data/                           ← gitignored
│   ├── pairs_pt1/Bildpaare Teil 1/ ← 71 folders, book illustration vases
│   └── pairs_pt2/Bildpaare-Triplets Teil 2/ ← 30 folders, mixed photos+illustrations
│
├── runs/                           ← gitignored, generated artifacts
│   ├── dinov2_full/                ← V5a results
│   │   ├── metrics.json
│   │   ├── retrieval_indices.npy
│   │   ├── retrieval_labels.npy
│   │   ├── test_embeddings.npy
│   │   └── train_embeddings.npy
│   └── dinov2_triplet/             ← V5b results
│       ├── metrics.json
│       ├── model.pth
│       ├── retrieval_indices.npy
│       └── retrieval_labels.npy
│
├── logs/                           ← gitignored, local run logs
├── job_v5_dinov2.sh                ← SLURM job script for V5 on TinyGPU A100
├── train_folders.txt               ← gitignored, DO NOT DELETE
├── test_folders.txt                ← gitignored, DO NOT DELETE
├── EXPERIMENTS.md                  ← full results table and version history
├── PROJECT_OVERVIEW.md
├── master-context-midhun.md        ← personal notes (can be deleted)
├── meeting_questions.md            ← personal notes (can be deleted)
└── proxyanchor_questions.txt       ← empty, can be deleted
```

---

## Proposed Final Clean Structure (NOT YET IMPLEMENTED)

The goal is one branch, all versions visible, professional for professor presentation:

```
vase_urn_project/
│
├── v1_baseline/
│   ├── extract_features.py
│   ├── retrieve.py
│   ├── evaluate.py
│   └── job_v1_baseline.sh
│
├── v2_resnet_triplet/
│   ├── extract_features.py
│   ├── train.py
│   ├── retrieve.py
│   ├── evaluate.py
│   └── job_v2_resnet_triplet.sh
│
├── v3_arcface/                     ← needs to be pulled from arcface branch
│   ├── extract_features.py
│   ├── detect_crop.py
│   ├── train.py
│   ├── retrieve.py
│   ├── evaluate.py
│   └── job_v3_arcface.sh
│
├── v4_sam_triplet/                 ← needs to be pulled from v4-sam-triplet branch
│   ├── detect_crop.py
│   ├── extract_features.py
│   ├── train.py
│   ├── retrieve.py
│   ├── evaluate.py
│   └── job_v4_sam_triplet.sh
│
├── v5_dinov2/                      ★ ACTIVE
│   ├── common.py
│   ├── extract_features.py
│   ├── train.py
│   ├── retrieve.py
│   ├── evaluate.py
│   ├── cache_model.py
│   └── job_v5_dinov2.sh
│
├── results/                        ← all version metrics in one place
│   ├── v1_metrics.json
│   ├── v2_metrics.json
│   ├── v3_metrics.json
│   ├── v4_metrics.json
│   └── v5_metrics.json
│
├── tools/
│   ├── split_data.py               ← run ONCE before anything else
│   ├── hpc_manager.py
│   └── local_manager.py
│
├── demo/
│   ├── demo.py
│   ├── demo_v5_dinov2.py
│   └── demo_all_versions.py
│
├── data/
├── runs/
├── train_folders.txt               ← NEVER DELETE
├── test_folders.txt                ← NEVER DELETE
└── EXPERIMENTS.md
```

---

## What Still Needs To Be Done

### Immediate (wait for cluster job 1583512 to finish first)
- [ ] Verify V5 results from cluster job are correct
- [ ] Copy fresh `metrics.json` from cluster to local `runs/`

### Folder structure completion
- [ ] Pull V3 scripts from `arcface` branch → put in `v3_arcface/`
- [ ] Pull V4 scripts from `v4-sam-triplet` branch → put in `v4_sam_triplet/`
- [ ] Write `job_v1_baseline.sh`, `job_v2_resnet_triplet.sh`, `job_v3_arcface.sh`, `job_v4_sam_triplet.sh`
- [ ] Create `results/` folder and populate with all version metrics
- [ ] Delete noise files: `master-context-midhun.md`, `meeting_questions.md`, `proxyanchor_questions.txt`, `PROJECT_OVERVIEW.md`

### Results (before thesis submission)
- [ ] Re-evaluate V1 and V2 using corrected full-ranking evaluator (currently their results used old top-10 evaluator — not comparable to V5)
- [ ] Save V3 and V4 metrics.json (results were never saved, only printed to terminal)

---

## V5 Pipeline (How It Works)

```
image → DINOv2 ViT-S/14 → 384-dim features
                               ↓
                    MLP: 384 → 512 → 128
                    (trained with Triplet Loss)
                               ↓
                    FAISS full-ranking search
                               ↓
                    mAP, Acc@1, Acc@10
```

Run order on cluster:
```bash
python v5_dinov2/extract_features.py --output-dir runs/dinov2_full --model-name dinov2_vits14
python v5_dinov2/retrieve.py --input-dir runs/dinov2_full
python v5_dinov2/evaluate.py --input-dir runs/dinov2_full

python v5_dinov2/train.py --input-dir runs/dinov2_triplet --output-dir runs/dinov2_triplet
python v5_dinov2/retrieve.py --input-dir runs/dinov2_triplet --model-path runs/dinov2_triplet/model.pth
python v5_dinov2/evaluate.py --input-dir runs/dinov2_triplet
```

Or just submit: `sbatch.tinygpu job_v5_dinov2.sh`

---

## HPC Cluster Info

| Field | Value |
|---|---|
| Cluster | FAU NHR TinyGPU |
| Login node | `tinyx.nhr.fau.de` |
| Gateway | `csnhr.nhr.fau.de` (ProxyJump) |
| Username | `iwi5419h` |
| SSH key | `~/.ssh/id_ed25519_nhr_fau` |
| Project dir | `/home/woody/iwi5/iwi5419h/vase_project` |
| Torch cache | `/home/woody/iwi5/iwi5419h/torch_cache` |
| GPU | A100 (`--partition=a100 --gres=gpu:a100:1`) |
| Conda env | `vaseretrieval` |

SSH command:
```bash
ssh -J iwi5419h@csnhr.nhr.fau.de iwi5419h@tinyx.nhr.fau.de
```

Submit job:
```bash
sbatch.tinygpu job_v5_dinov2.sh
squeue -u iwi5419h
tail -f v5_dinov2_<JOBID>.out
scancel <JOBID>
```

---

## Critical Rules

1. **Never delete `train_folders.txt` or `test_folders.txt`** — these define the 80/20 split. Deleting them makes all version results incomparable.
2. **Run `split_data.py` only once** — ever. All versions must use the same split.
3. **Never submit the same job twice** — check `squeue -u iwi5419h` first. Cancel duplicates with `scancel <JOBID>`.
4. **To roll back everything** to before today's restructure: `git reset --hard pre-restructure`

---

## Job Scripts (All Versions)

### V1 — Raw ResNet50 baseline
```bash
python split_data.py
python extract_features.py
python retrieve_baseline.py
python evaluate_baseline.py
```

### V2 — ResNet50 + Triplet Loss
```bash
python split_data.py
python extract_features.py
python train.py
python retrieve.py
python evaluate.py
```

### V3 — ResNet50 + ArcFace + SAM crop
```bash
python split_data.py
python detect_crop.py
python extract_features.py
python train.py
python retrieve.py
python evaluate.py
```

### V4 — ResNet50 + SAM crop + Triplet Loss
```bash
python split_data.py
python detect_crop.py
python extract_features.py
python train.py
python retrieve.py
python evaluate.py
```

### V5 — DINOv2 + Triplet Loss (use job_v5_dinov2.sh)
```bash
python v5_dinov2/extract_features.py --output-dir runs/dinov2_full
python v5_dinov2/retrieve.py --input-dir runs/dinov2_full
python v5_dinov2/evaluate.py --input-dir runs/dinov2_full
python v5_dinov2/train.py --input-dir runs/dinov2_triplet
python v5_dinov2/retrieve.py --input-dir runs/dinov2_triplet --model-path runs/dinov2_triplet/model.pth
python v5_dinov2/evaluate.py --input-dir runs/dinov2_triplet
```
