# Experiment Results - V4 SAM Branch

**Project:** CBIR for archaeological vase retrieval  
**Supervisor:** Mathias Zinnen (FAU)  
**Branch:** `v4-sam-triplet`  
**Last updated:** April 22, 2026

---

## Results

| Version | Description | Hardware | mAP | Acc@1 | Acc@10 |
|---------|-------------|----------|-----|-------|--------|
| V1 - Baseline | Raw ResNet50 features, no MLP, no metric learning | CPU | 30.27% | 23.08% | 76.92% |
| V1 - Baseline | Raw ResNet50 features, no MLP, no metric learning | RTX 3050 (local) | 10.31% | 15.38% | 30.77% |
| V2 - Triplet Loss | ResNet50 features + MLP + TripletMarginLoss + MultiSimilarityMiner | A100 GPU (TinyGPU) | **54.04%** | **42.31%** | **92.31%** |
| V2 - Triplet Loss | ResNet50 features + MLP + TripletMarginLoss + MultiSimilarityMiner | RTX 3050 (local) | 44.67% | 34.62% | 90.38% |
| V3 - ArcFace | ResNet50 features + MLP + ArcFaceLoss | RTX 3050 (local) | 31.14% | 26.92% | 75.00% |
| V4 - SAM + Triplet | SAM crops + ResNet50 features + MLP + TripletMarginLoss | A100 GPU (TinyGPU) | 18.89% | 13.46% | 69.23% |

---

## V4 SAM Summary

V4 tested SAM as a zero-shot crop generator before the existing ResNet50 + Triplet Loss retrieval pipeline.

Pipeline:

```text
split_data.py
        -> detect_crop.py
        -> extract_features.py
        -> train.py
        -> retrieve.py
        -> evaluate.py
```

Valid SAM job:

```text
Job ID: 1581893
Crops created: 250
mAP: 18.89%
Accuracy@1: 13.46%
Accuracy@10: 69.23%
```

Important correction:

The earlier job `1581869` is not a valid SAM result because `detect_crop.py` failed when OpenCV (`cv2`) was missing. The job continued without crops because the old `job.sh` did not stop on errors.

Fixes made:

- Installed compatible OpenCV in the `vaseretrieval` conda environment.
- Restored NumPy below 2.0 for FAISS compatibility.
- Added `set -e` to `job.sh`.
- Reran SAM as job `1581893`.

Conclusion:

> SAM cropping worked technically, but it reduced retrieval performance. Many crops were visually plausible, but the crop step likely removed useful full-vase context or produced inconsistent crop regions across matching pairs.

SAM is therefore documented as a negative/analysis experiment, not used as the final retrieval method.
