# V6 - DINOv2 Retrieval

This is a standalone experiment pipeline for full-image DINOv2 retrieval.

It writes all outputs into `runs/` so it does not overwrite artifacts from other versions.

## Experiments

### V6a - DINOv2 Full-Image Retrieval

```text
image -> DINOv2 feature -> L2 normalize -> FAISS full ranking -> evaluate
```

Run:

```bash
python v6_dinov2/extract_features.py --output-dir runs/dinov2_full
python v6_dinov2/retrieve.py --input-dir runs/dinov2_full
python v6_dinov2/evaluate.py --input-dir runs/dinov2_full
```

### V6b - DINOv2 + Triplet Loss

```text
image -> DINOv2 feature -> MLP 384 -> 512 -> 128 -> Triplet Loss -> FAISS full ranking -> evaluate
```

Run:

```bash
python v6_dinov2/extract_features.py --output-dir runs/dinov2_triplet
python v6_dinov2/train.py --input-dir runs/dinov2_triplet --output-dir runs/dinov2_triplet
python v6_dinov2/retrieve.py --input-dir runs/dinov2_triplet --model-path runs/dinov2_triplet/model.pth
python v6_dinov2/evaluate.py --input-dir runs/dinov2_triplet
```

## Notes

- The default model is `dinov2_vits14` from `facebookresearch/dinov2` via `torch.hub`.
- The first run may need internet access to download/cache the DINOv2 model.
- Evaluation retrieves the full test ranking, then skips the query itself. This fixes the older `k=10` issue where Acc@10 effectively used only 9 non-self results.
