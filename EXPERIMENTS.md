# Experiment Results — Vase/Urn Metric Learning

**Project:** CBIR for archaeological vase retrieval  
**Supervisor:** Mathias Zinnen (FAU)  
**Deadline:** June 12, 2026  

---

## Pipeline Overview

```
ResNet50 (2048-dim features)
        ↓
[Optional] MLP embedding (2048 → 512 → 128)
        ↓
FAISS index → Top-10 retrieval
        ↓
Evaluate: mAP, Acc@1, Acc@10
```

---

## Results

| Version | Description | Hardware | mAP | Acc@1 | Acc@10 |
|---------|-------------|----------|-----|-------|--------|
| V1 — Baseline | Raw ResNet50 features, no MLP, no metric learning | CPU | 30.27% | 23.08% | 76.92% |
| V1 — Baseline | Raw ResNet50 features, no MLP, no metric learning | RTX 3050 (local) | 10.31% | 15.38% | 30.77% |
| V2 — Triplet Loss | MLP + TripletMarginLoss + MultiSimilarityMiner | A100 GPU (TinyGPU) | **54.04%** | **42.31%** | **92.31%** |
| V2 — Triplet Loss | MLP + TripletMarginLoss + MultiSimilarityMiner | RTX 3050 (local) | 44.67% | 34.62% | 90.38% |
| V3 — ArcFace | MLP + ArcFaceLoss (margin=28.6, scale=64) | RTX 3050 (local) | 31.14% | 26.92% | 75.00% |
| V4 — SAM + Triplet | SAM crops + MLP + TripletMarginLoss + MultiSimilarityMiner | RTX 3050 (local) | pending | — | — |

> **Note:** V2 TinyGPU result is the gold standard. Local RTX 3050 results are for development comparison only.

---

## Version Details

### V1 — Baseline (No Metric Learning)
- **Scripts:** `extract_features.py` → `retrieve_baseline.py` → `evaluate_baseline.py`
- **Feature dim:** 2048 (raw ResNet50 pool layer)
- **FAISS index:** `IndexFlatL2(2048)`
- **No training step**

### V2 — Triplet Loss
- **Scripts:** Full pipeline — `split_data.py` → `extract_features.py` → `train.py` → `retrieve.py` → `evaluate.py`
- **Model:** MLP 2048 → 512 → ReLU → 128
- **Loss:** `TripletMarginLoss` + `MultiSimilarityMiner`
- **Optimizer:** Adam (lr=0.001), 50 epochs, batch=32
- **Feature dim:** 128 (post-MLP, L2-normalized)
- **FAISS index:** `IndexFlatL2(128)`

### V3 — ArcFace Loss
- **Scripts:** Same as V2
- **Model:** MLP 2048 → 512 → ReLU → 128 (identical architecture)
- **Loss:** `ArcFaceLoss(num_classes=K, embedding_size=128, margin=28.6, scale=64)`
- **Optimizer:** Adam (lr=0.001) over model params + ArcFace params, 50 epochs, batch=32
- **Key difference from V2:** Angular margin classification — learns tight angular boundaries per class
- **Feature dim:** 128

### V4 — SAM + Triplet Loss
- **Scripts:** `split_data.py` → `detect_crop.py` → `extract_features.py` → `train.py` → `retrieve.py` → `evaluate.py`
- **New step:** `detect_crop.py` runs SAM (vit_b) on every image, saves cropped figure regions to `crops/`
- **Model:** MLP 2048 → 512 → ReLU → 128
- **Loss:** `TripletMarginLoss` + `MultiSimilarityMiner` (same as V2)
- **Key difference from V2:** Features extracted from SAM-cropped figure regions, not full images
- **Feature dim:** 128

---

## Next Steps

- [ ] Install SAM + download checkpoint, run V4 locally
- [ ] Run V4 on TinyGPU A100 for fair GPU comparison
- [ ] Pose estimation approach (skeleton keypoints)
- [ ] Final comparison across all approaches
