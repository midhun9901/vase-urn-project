# Vase Retrieval — Master Project

**Type:** ML Research (Content-Based Image Retrieval)
**Author:** Midhun Somanunnithan (midhun.somanunnithan@fau.de)
**Supervisor:** Mathias Zinnen (mathias.zinnen@fau.de)
**Deadline:** June 12, 2026
**Status:** Phases 1-2 Complete, Phases 3-5 Pending

---

## What It Is

A CBIR (Content-Based Image Retrieval) system for archaeological vase images. Given a query image of an ancient vase, find visually similar vases across a dataset of 101 image folders. Like Google Image Search for archaeology.

---

## Full Pipeline (Per Supervisor)

```
Dataset (101 folders of vase images)
        |
        v
Detection — detect figures on vases (YOLO / Faster RCNN)
        |
        v
3 Retrieval Approaches:
  1. Naive ResNet features (baseline)         --> DONE
  2. Metric Learning — Triplet Loss           --> DONE
  3. Pose Estimation — skeleton keypoints     --> NEXT
        |
        v
Evaluation — compare all 3 approaches (mAP, Acc@1, Acc@10)
        |
        v
Final Report — June 12, 2026
```

---

## Tech Stack

| Component | Tech |
|-----------|------|
| Language | Python 3.10+ |
| ML Framework | PyTorch + TorchVision (ResNet-50) |
| Metric Learning | pytorch-metric-learning (Triplet Loss, MultiSimilarityMiner) |
| Similarity Search | FAISS (faiss-gpu) |
| Data | NumPy (<2.0 — faiss crashes with numpy 2.x) |
| Compute | FAU HPC TinyGPU cluster (NVIDIA A100) |
| Job Scheduler | SLURM (`sbatch.tinygpu`) |
| Local GUI | Tkinter (pipeline manager UIs) |

---

## Dataset

| Folder | Contents | Count |
|--------|----------|-------|
| `pairs_pt1` | Old book illustrations (consistent style) | 71 folders |
| `pairs_pt2` | Museum photos + illustrations (harder, mixed) | 30 folders |
| `urns_small` | Urn dataset (ignored — focus on vases only) | — |
| **Total** | | **101 folders** |

- **Split:** 80/20 train/test, seed=42 (deterministic)
- **Location on HPC:** `/home/woody/iwi5/iwi5419h/vase_project/`

---

## Pipeline Scripts (Run in Order)

| # | Script | What It Does | Output |
|---|--------|-------------|--------|
| 1 | `split_data.py` | 80/20 train/test split, seed=42 | `train_folders.txt`, `test_folders.txt` |
| 2 | `extract_features.py` | ResNet-50 feature extraction (2048-dim) | `train_embeddings.npy`, `test_embeddings.npy`, `*_labels.npy` |
| 3 | `train.py` | MLP 2048→512→128 with Triplet Loss + MultiSimilarityMiner, 50 epochs | `model.pth` |
| 4 | `retrieve.py` | FAISS index on 128-dim embeddings, top-10 retrieval per query | `retrieval_indices.npy`, `retrieval_labels.npy` |
| 5 | `evaluate.py` | Compute mAP, Accuracy@1, Accuracy@10 | Console output |

**Baseline pipeline** (no metric learning): `pipeline_baseline/retrieve.py` → `pipeline_baseline/evaluate.py` (uses raw 2048-dim features directly)

---

## Results History

| Date | Setup | mAP | Acc@1 | Acc@10 | Notes |
|------|-------|-----|-------|--------|-------|
| Week 5 | CPU | 30.27% | 23.08% | 76.92% | Baseline, slow |
| Mar 24, 2026 | **A100 GPU (TinyGPU)** | **54.04%** | **42.31%** | **92.31%** | Best run |
| Apr 8, 2026 | CPU (Woody — wrong cluster) | 29.25% | 23.73% | 83.05% | Wrong cluster, no GPU |

**Always run on TinyGPU with GPU.** CPU runs on Woody give unreliable/worse results.

---

## Project Structure

```
vase_urn_project/
├── data/
│   ├── pairs_pt1/                    # 71 folders (book illustrations)
│   └── pairs_pt2/                    # 30 folders (museum photos)
│
├── pipeline_baseline/                # Approach 1: Raw ResNet features
│   ├── retrieve.py                   # FAISS on 2048-dim features
│   └── evaluate.py                   # mAP, Acc@1, Acc@10
│
├── pipeline_metric_learning/         # Approach 2: Triplet Loss embeddings
│   ├── train.py                      # MLP training (2048→512→128)
│   ├── retrieve.py                   # FAISS on 128-dim embeddings
│   └── evaluate.py                   # mAP, Acc@1, Acc@10
│
├── split_data.py                     # 80/20 split, seed=42
├── extract_features.py               # ResNet-50 feature extraction
├── train.py                          # Top-level training script
├── retrieve.py                       # Top-level retrieval script
├── evaluate.py                       # Top-level evaluation script
├── evaluate_baseline.py              # Baseline evaluation
├── retrieve_baseline.py              # Baseline retrieval
│
├── hpc_manager.py                    # Tkinter GUI for HPC job management (SSH/Paramiko)
├── local_manager.py                  # Tkinter GUI for local pipeline execution
├── job.sh                            # SLURM batch script (A100, 2h)
│
├── master-context-midhun.md          # Full project context document
│
├── [Generated artifacts]
│   ├── train_embeddings.npy          # 2048-dim ResNet features (train)
│   ├── test_embeddings.npy           # 2048-dim ResNet features (test)
│   ├── train_labels.npy / test_labels.npy
│   ├── model.pth                     # Trained MLP weights
│   ├── retrieval_indices.npy         # Top-10 neighbor indices
│   ├── retrieval_labels.npy
│   ├── retrieval_indices_baseline.npy
│   └── retrieval_labels_baseline.npy
│
└── train_folders.txt / test_folders.txt
```

---

## HPC Setup (FAU TinyGPU)

| Item | Value |
|------|-------|
| Cluster | TinyGPU (`tinyx.nhr.fau.de`) — has A100 GPUs |
| Username | `iwi5419h` |
| Storage | `$WORK = /home/woody/iwi5/iwi5419h` (954GB quota) |
| Conda env | `vaseretrieval` (Python 3.10, PyTorch cu121, faiss-gpu) |
| Submit cmd | `sbatch.tinygpu job.sh` (NOT `sbatch` — that's Woody/CPU) |
| SSH | `ssh tinygpu` via ProxyJump through `csnhr` |

### Quick Commands
```bash
ssh tinygpu                                    # Connect
module load python && conda activate vaseretrieval  # Activate env
cd /home/woody/iwi5/iwi5419h/vase_project      # Go to project
sbatch.tinygpu job.sh                           # Submit GPU job
squeue -u $USER                                 # Check queue
cat vase_JOBID.out                              # Read output
scancel JOBID                                   # Cancel job
```

### Upload from Local
```bash
scp -rO "c:/PROJECTS/vase_urn_project" iwi5419h@woody.nhr.fau.de:~/vase_project/
```

---

## Timeline & Roadmap

### Completed

| Phase | Task | Status |
|-------|------|--------|
| **1** | **Baseline — Raw ResNet-50 features + FAISS** | DONE |
| | Extract 2048-dim features from ResNet-50 (pretrained ImageNet) | |
| | Build FAISS index, retrieve top-10 neighbors | |
| | Evaluate: mAP=54.04%, Acc@1=42.31%, Acc@10=92.31% | |
| **2** | **Metric Learning — Triplet Loss** | DONE |
| | Train MLP projection head: 2048 → 512 → 128 | |
| | Loss: TripletMarginLoss + MultiSimilarityMiner | |
| | 50 epochs on A100 GPU | |
| | FAISS index on 128-dim learned embeddings | |

### Upcoming

| Phase | Task | Timeline | Details |
|-------|------|----------|---------|
| **2b** | **Improve Metric Learning** | Mar–Apr | Fine-tune ResNet-50 end-to-end (not just MLP head). Try ProxyAnchor loss (better for small datasets with few images per class) |
| **3** | **Pose Estimation** | Apr 1–28 | Extract skeleton keypoints from figures on vases. Use keypoint-based features as a 3rd retrieval approach. Compare against baseline and triplet loss |
| **4** | **Figure Detection** | Apr 29–May 19 | Train YOLO or Faster RCNN to detect figures/scenes painted on vases. Crop detected regions → feed into retrieval pipeline. This is the "Detection" step in the full pipeline |
| **5** | **Comparison + Final Report** | May 20–Jun 12 | Compare all 3 approaches (baseline, metric learning, pose estimation) with and without figure detection. Write final master's project report. **Deadline: June 12, 2026** |

### Phase 3: Pose Estimation (Detail)

**Goal:** Use body pose/skeleton keypoints of figures painted on vases as features for retrieval.

**Approach:**
- Detect human figures on vase images (may overlap with Phase 4)
- Extract keypoint coordinates (joints: head, shoulders, elbows, wrists, hips, knees, ankles)
- Encode pose as feature vector (normalized keypoint positions or pose descriptor)
- Build FAISS index on pose features
- Retrieve: given query vase, find vases with similar figure poses
- Evaluate with same metrics: mAP, Acc@1, Acc@10

**Challenges:**
- Vase figures are stylized ancient art, not real humans — standard pose estimators (OpenPose, MediaPipe) may struggle
- May need fine-tuning or domain adaptation
- Some vases have multiple figures — need per-figure extraction
- Occlusion and partial figures common in ancient art

### Phase 4: Figure Detection (Detail)

**Goal:** Automatically detect and crop figures/scenes painted on vases before feeding into retrieval.

**Approach:**
- Train object detector (YOLO v5/v8 or Faster RCNN) on annotated vase images
- Classes: human figure, animal, decorative pattern (TBD with supervisor)
- Crop detected regions → extract features → retrieve
- Compare: full-image retrieval vs. detected-region retrieval

**Why this matters:**
- Current pipeline uses the entire vase image including background, handles, lips
- Detecting and cropping just the painted figures should improve retrieval precision
- The "Detection" step is explicitly part of the professor's required pipeline

### Phase 5: Final Comparison & Report

**Compare all approaches:**

| Approach | Features | Dim |
|----------|----------|-----|
| Baseline | Raw ResNet-50 | 2048 |
| Metric Learning | Triplet Loss MLP | 128 |
| Pose Estimation | Skeleton keypoints | TBD |
| + Figure Detection | Each above, but on cropped figures | Same |

**Metrics:** mAP (primary), Accuracy@1, Accuracy@10

**Report structure (TBD with supervisor):**
- Introduction & motivation
- Related work (CBIR, metric learning, pose estimation for art)
- Dataset description
- Methodology (3 approaches + detection)
- Experiments & results
- Discussion & comparison
- Conclusion

---

## IEEE Citations

| # | Reference |
|---|-----------|
| [1] | He et al., "Deep Residual Learning for Image Recognition," CVPR 2016 |
| [2] | Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019 |
| [3] | Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering," CVPR 2015 |
| [4] | Musgrave et al., "PyTorch Metric Learning," arXiv 2020 |
| [5] | Johnson et al., "Billion-Scale Similarity Search with GPUs," IEEE TBD 2021 |
| [6] | Zhou et al., "Recent Advance in Content-based Image Retrieval," arXiv 2017 |
| [7] | Wang et al., "Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning," CVPR 2019 |

---

## Key Links

| Resource | Link |
|----------|------|
| HPC Portal | https://portal.hpc.fau.de/ |
| TinyGPU Docs | https://doc.nhr.fau.de/clusters/tinygpu/ |
| Reference Codebase | https://github.com/mathiaszinnen/marki-retrieval |
| PyTorch Metric Learning | https://kevinmusgrave.github.io/pytorch-metric-learning/ |
| FAISS Docs | https://faiss.ai/ |

---

## Tools

| Tool | Usage |
|------|-------|
| Antigravity IDE | Gemini 3 Flash (everyday), Claude Opus 4.6 (hard tasks) |
| NotebookLM | "Midhun Master Project - Vase Retrieval" notebook |
| Git Bash | SSH to HPC (`ssh tinygpu`) |

---

## Note: `vase_project/` (Older Version)

`C:\PROJECTS\vase_project\` is an earlier baseline version (March 2026) with:
- Hardcoded HPC paths (not portable)
- Flat structure (no pipeline_baseline/pipeline_metric_learning split)
- No GUI managers (hpc_manager.py, local_manager.py)
- Same results but less organized

**Use `vase_urn_project/` for all active work.** Archive `vase_project/` as reference only.
