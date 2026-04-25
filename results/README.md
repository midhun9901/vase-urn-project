# Results

All version metrics collected in one place.

| File | Version | mAP | Acc@1 | Acc@10 | Evaluator |
|---|---|---|---|---|---|
| v1_metrics.json | V1 Raw ResNet50 | — | — | — | needs re-eval |
| v2_metrics.json | V2 ResNet50 + Triplet | — | — | — | needs re-eval |
| v3_metrics.json | V3 ResNet50 + ProxyAnchor | — | — | — | not yet run |
| v4_metrics.json | V4 ResNet50 + ArcFace + SAM | — | — | — | not yet run |
| v5_metrics.json | V5 ResNet50 + SAM + Triplet | — | — | — | not yet run |
| v6a_dinov2_full_metrics.json | V6a DINOv2 full-image | 32.53% | 26.92% | 82.69% | full-ranking |
| v6b_dinov2_triplet_metrics.json | V6b DINOv2 + Triplet | **83.39%** | **80.77%** | **100%** | full-ranking |

**Note:** V1 and V2 results were originally computed with the old top-10 evaluator and are not directly comparable to V6.
Re-run their `evaluate.py` scripts using the corrected full-ranking evaluator before thesis submission.
