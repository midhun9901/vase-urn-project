# Vase Retrieval - V4 SAM Branch Overview

**Type:** ML Research - Content-Based Image Retrieval  
**Author:** Midhun Somanunnithan  
**Supervisor:** Mathias Zinnen  
**Branch:** `v4-sam-triplet`  
**Deadline:** June 12, 2026

---

## Project Goal

Build a retrieval system for archaeological vase images. Given a query vase image, retrieve visually similar vase images from the dataset.

---

## V4 Purpose

This branch preserves the SAM crop experiment.

SAM was tested as a zero-shot preprocessing step:

```text
original image
        -> SAM automatic mask generation
        -> selected crop
        -> ResNet50 feature extraction
        -> Triplet Loss MLP
        -> FAISS retrieval
        -> evaluation
```

SAM is not used in the later V5 DINOv2 branch.

---

## Dataset

| Folder | Contents | Count |
|--------|----------|-------|
| `pairs_pt1` | Old book illustrations | 71 folders |
| `pairs_pt2` | Museum photos and illustrations | 30 folders |
| `urns_small` | Ignored for this project | - |
| **Total** | Vase folders | **101 folders** |

Split:

```text
80 train folders
21 test folders
seed = 42
```

---

## V4 Result

Valid SAM run:

| Setup | Job ID | Crops | mAP | Acc@1 | Acc@10 |
|-------|--------|-------|-----|-------|--------|
| SAM crops + ResNet50 + Triplet Loss | `1581893` | 250 | 18.89% | 13.46% | 69.23% |

Best pre-V5 result:

| Setup | mAP | Acc@1 | Acc@10 |
|-------|-----|-------|--------|
| Full-image ResNet50 + Triplet Loss | 54.04% | 42.31% | 92.31% |

Conclusion:

> SAM cropping reduced retrieval performance. It should be kept as a documented failed/negative experiment, not used as the main method.

---

## Important Files

| File | Purpose |
|------|---------|
| `detect_crop.py` | Runs SAM and saves crops |
| `job.sh` | TinyGPU SAM + Triplet job |
| `SAM_CROP_INSPECTION.md` | Visual inspection notes for SAM crops |
| `inspect_sam_crops.py` | Helper script for side-by-side crop review sheets |
| `EXPERIMENTS.md` | V4 result documentation |

---

## Run on TinyGPU

```bash
cd /home/woody/iwi5/iwi5419h/vase_project
sbatch.tinygpu job.sh
```

The SAM checkpoint is expected as:

```text
sam_vit_b.pth
```
