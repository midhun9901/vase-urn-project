# Master Project Context — Midhun Somanunnithan
**Last updated: April 8, 2026**
Use this file to start a fresh Claude chat with full context.

---

## Who You Are
- **Name:** Midhun Somanunnithan
- **FAU Email:** midhun.somanunnithan@fau.de
- **Supervisor:** Mathias Zinnen (mathias.zinnen@fau.de)
- **Project Deadline:** June 12, 2026
- **HPC Username:** iwi5419h
- **HPC Account Valid Until:** 2026-09-12

---

## Project Goal
Build a CBIR (Content-Based Image Retrieval) system that finds visually similar vases using metric learning — like Google Image Search for archaeological artifacts.

**Full pipeline per Professor Mathias:**
```
Dataset (101 folders of vase images)
        ↓
Detection — detect figures on vases (YOLO/Faster RCNN)
        ↓
3 Retrieval Approaches:
  1. Naive ResNet features (baseline)     ✅ DONE
  2. Metric Learning — Triplet Loss       ✅ DONE
  3. Pose Estimation — skeleton keypoints ← NEXT (April)
        ↓
Evaluation — compare all 3 approaches
        ↓
Final Report — June 12, 2026
```

---

## Data
- **Location on HPC:** `/home/woody/iwi5/iwi5419h/vase_project/`
- `pairs_pt1 (1)/Bildpaare Teil 1/` — 71 folders, old book illustrations (consistent style)
- `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/` — 30 folders, museum photos + illustrations (harder)
- `urns_small (1)/` — urns dataset (ignore for now, focus on vases)
- Total: 101 folders

**Key notes from Mathias:**
- Focus on **vases only** (ignore urns)
- Evaluate with **mAP** (most important metric), then Acc@1, Acc@10
- Reference codebase: https://github.com/mathiaszinnen/marki-retrieval

---

## Pipeline Scripts (run in order)
| Script | What it does |
|--------|-------------|
| `split_data.py` | 80/20 train/test split, seed=42, saves train/test_folders.txt |
| `extract_features.py` | ResNet50, extracts 2048-dim feature vectors |
| `train.py` | MLP 2048→512→128, Triplet Loss + MultiSimilarityMiner, 50 epochs |
| `retrieve.py` | FAISS index, top-10 retrieval per query |
| `evaluate.py` | Computes mAP, Accuracy@1, Accuracy@10 |

**Coding style:** Simple, clean, plain variable names, print statements for progress, no over-engineering.

---

## Results History

| Date | Setup | mAP | Acc@1 | Acc@10 |
|------|-------|-----|-------|--------|
| Week 5 baseline | CPU | 30.27% | 23.08% | 76.92% |
| March 24, 2026 | A100 GPU (TinyGPU) | **54.04%** | **42.31%** | **92.31%** |
| April 8, 2026 | CPU (Woody — wrong cluster) | 29.25% | 23.73% | 83.05% |

**Note:** Always run on TinyGPU with GPU for real results. CPU runs are unreliable/slower.

---

## How to SSH (Simple Version)

1. Open **Git Bash** (not CMD or PowerShell)
2. Type:
   ```bash
   ssh tinygpu
   ```
3. You're in. You'll see: `iwi5419h@tinygpu:~$`

That's it. The SSH key handles the password automatically.

**If it says "Host not found" or errors** — your SSH config file is missing. It lives at `C:\Users\Asus\.ssh\config`. See the SSH Config section below to recreate it.

### SSH Key Status (updated April 15, 2026)
- Old keys (`rog_midhun`) on portal are outdated — safe to delete
- New key generated today on this laptop: `C:\Users\Asus\.ssh\id_ed25519`
- Uploaded to portal as alias `laptop`
- Key distribution takes up to 2 hours — if SSH still fails, wait and retry
- If SSH config is missing, recreate it as shown below

---

## HPC Setup — Correct Way

### Two clusters (don't mix them up)
| Cluster | Host | Has GPU? | Use for |
|---------|------|----------|---------|
| **TinyGPU** | tinyx.nhr.fau.de | ✅ A100 GPU | Training — always use this |
| Woody | woody.nhr.fau.de | ❌ CPU only | Don't use for training |

### SSH Config (C:\Users\Asus\.ssh\config)
```
Host csnhr
    HostName csnhr.nhr.fau.de
    User iwi5419h
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no

Host tinygpu
    HostName tinyx.nhr.fau.de
    User iwi5419h
    ProxyJump csnhr
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
```

### Connect (always use this, not the long version)
```bash
ssh tinygpu
```

### Storage
- **$WORK = `/home/woody/iwi5/iwi5419h`** ← use this for everything (954GB quota)
- Home directory has only ~100GB quota — don't use for data

---

## HPC Environment Setup (already done, just activate)
```bash
module load python
conda activate vaseretrieval
```

If conda environment is missing, recreate:
```bash
conda config --add envs_dirs $WORK/conda/envs
conda config --add pkgs_dirs $WORK/conda/pkgs
conda create -n vaseretrieval python=3.10 -y
conda activate vaseretrieval
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-metric-learning faiss-gpu
pip install "numpy<2"  # important — faiss-gpu crashes with numpy 2.x
pip install faiss-gpu --force-reinstall
```

---

## Slurm Job Script (job.sh on HPC)
```bash
#!/bin/bash -l
#SBATCH --job-name=vase-retrieval
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --output=vase_%j.out
#SBATCH --error=vase_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
conda activate vaseretrieval

cd /home/woody/iwi5/iwi5419h/vase_project

python split_data.py
python extract_features.py
python train.py
python retrieve.py
python evaluate.py
```

**Submit with:** `sbatch.tinygpu job.sh` (NOT `sbatch` — that's for Woody)

---

## Important Commands
```bash
# Connect
ssh tinygpu

# Go to project
cd /home/woody/iwi5/iwi5419h/vase_project

# Activate environment
module load python
conda activate vaseretrieval

# Upload files from local PC (run from local terminal, not SSH)
scp -rO "c:/PROJECTS/vase_urn_project" iwi5419h@woody.nhr.fau.de:~/vase_project/

# Submit job
sbatch.tinygpu job.sh

# Check job queue
squeue -u $USER

# Watch queue every 30 seconds
watch -n 30 squeue -u $USER

# Read job output
cat vase_JOBID.out

# Cancel a job
scancel JOBID

# Check cluster status
sinfo

# Check storage quota
shownicerquota.pl
```

---

## Project File Structure on HPC
```
/home/woody/iwi5/iwi5419h/vase_project/
├── split_data.py
├── extract_features.py
├── train.py
├── retrieve.py
├── evaluate.py
├── job.sh
├── train_folders.txt
├── test_folders.txt
├── model.pth
├── train_embeddings.npy
├── test_embeddings.npy
├── train_labels.npy
├── test_labels.npy
├── retrieval_indices.npy
├── retrieval_labels.npy
├── pairs_pt1 (1)/
│   └── Bildpaare Teil 1/
├── pairs_pt2 (1)/
│   └── Bildpaare-Triplets Teil 2/
└── urns_small (1)/
```

---

## Remaining Timeline
| Phase | Task | Timeline | Status |
|-------|------|----------|--------|
| 2 | Improve Metric Learning (fine-tune ResNet50 end-to-end) | Mar–Apr | ⏭️ Next |
| 3 | Pose Estimation (skeleton keypoints) | Apr 1–28 | Pending |
| 4 | Figure Detection (YOLO/Faster RCNN) | Apr 29–May 19 | Pending |
| 5 | Compare all approaches + Final Report | May 20–Jun 12 | Pending |

---

## Next Session — Start Here
1. Connect: `ssh tinygpu` (in Git Bash)
2. Activate: `module load python && conda activate vaseretrieval`
3. Go to project: `cd /home/woody/iwi5/iwi5419h/vase_project`
4. Next task: Write fine-tuning script for ResNet50 end-to-end training
5. Then try ProxyAnchor loss (better for small datasets with few images per class)

---

## Tools
- **IDE:** Antigravity IDE
  - Gemini 3 Flash for everyday coding (unlimited)
  - Claude Opus 4.6 for hard tasks only (quota burns fast)
- **NotebookLM:** "Midhun Master Project - Vase Retrieval" notebook (all sources uploaded)

## IEEE Citations (Verified)

**[1]** K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

**[2]** A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.

**[3]** F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, 2015, pp. 815–823.

**[4]** K. Musgrave, S. Belongie, and S.-N. Lim, "PyTorch Metric Learning," *arXiv preprint arXiv:2008.09164*, 2020.

**[5]** J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021.

**[6]** W. Zhou, H. Li, and Q. Tian, "Recent Advance in Content-based Image Retrieval: A Literature Survey," *arXiv preprint arXiv:1706.06064*, 2017.

**[7]** X. Wang, X. Han, W. Huang, D. Dong, and M. R. Scott, "Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, 2019, pp. 5022–5030.

---

## Meeting Questions (April 16, 2026 — Mathias)

1. **Registration** — What are the next steps for formally registering the master's project?
2. **Project direction** — Should I continue toward pose estimation, or improve metric learning first?
3. **Report** — When to start? What structure? Is there a FAU CS LaTeX template? IEEE citations?

---

## Key Links
| Resource | Link |
|----------|------|
| HPC Portal | https://portal.hpc.fau.de/ |
| HPC Docs | https://doc.nhr.fau.de/ |
| TinyGPU Docs | https://doc.nhr.fau.de/clusters/tinygpu/ |
| PyTorch Metric Learning | https://kevinmusgrave.github.io/pytorch-metric-learning/ |
| FAISS Docs | https://faiss.ai/ |
| Reference Codebase | https://github.com/mathiaszinnen/marki-retrieval |
