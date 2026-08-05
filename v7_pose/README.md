# V7 - Vase Figure-Pose Retrieval

Pose-based retrieval using the **vase-specific** keypoint annotations from the
supervisor, rather than the generic COCO human-pose model used in the first V7
attempt.

```text
image -> (optional image-field crop) -> YOLO Figure-Pose -> 51-dim keypoints
      -> MLP 51 -> 128 -> 64 -> Triplet Loss -> FAISS full ranking -> evaluate
```

## Why this replaces the original V7

The first V7 loaded the off-the-shelf `yolov8n-pose.pt`, which is trained to
find **human** skeletons. On vase imagery it only reacts to clearly painted
figures and returns all-zero vectors otherwise, which capped the result at
~39.8% mAP. This version fine-tunes a pose model on the annotated
`Figure-Pose` dataset so the keypoints describe the actual vase figures.

## Assets (not committed — kept under `v7_pose/assets/`, gitignored)

| Asset | Purpose |
|-------|---------|
| `assets/pose_dataset/` | 17-keypoint `Figure-Pose` annotations (238 train + test splits) |
| `assets/image_field.pt` | YOLOv11 image-field (motif) detector, for optional cropping |

On the cluster, upload the same `assets/` folder (compute nodes have no internet
to download base weights either — pre-cache `yolo11n-pose.pt` as well).

## Run

```bash
# 1. Fine-tune the pose model on the vase keypoint dataset
python v7_pose/train_pose.py --output-dir runs/pose_model

# 2. Extract figure keypoints (add --crop-field to run pose on the image field only)
python v7_pose/extract_features.py --output-dir runs/pose_triplet \
    --pose-weights runs/pose_model/pose_best.pt

# 3. Train projection, retrieve, evaluate
python v7_pose/train.py    --input-dir runs/pose_triplet --output-dir runs/pose_triplet
python v7_pose/retrieve.py --input-dir runs/pose_triplet --model-path runs/pose_triplet/model.pth
python v7_pose/evaluate.py --input-dir runs/pose_triplet
```

Or submit the whole thing: `sbatch.tinygpu v7_pose/job_v7_pose.sh`.

## Notes

- `extract_features.py` prints how many images yielded a detected figure — watch
  this; if it is still low after fine-tuning, pose is genuinely sparse on this
  data and the negative-result framing stands.
- `train_pose.py` ignores the shipped `train.txt` (its `data/images/...` prefix
  does not match the real layout) and builds a deterministic train/val split
  from the actual image files.
