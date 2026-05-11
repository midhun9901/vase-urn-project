import argparse

import numpy as np

from common import project_root

# COCO 17 keypoint indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="runs/v7")
    parser.add_argument("--output-dir", default="runs/v7")
    return parser.parse_args()


def normalize_keypoints(kps):
    """
    kps: (N, 17, 3) raw x,y,conf from YOLO
    Returns (N, 34) normalized x,y coordinates.

    Normalization: center skeleton at hip midpoint, scale by torso length
    (midpoint of shoulders to midpoint of hips). Undetected images (all zeros)
    remain as zero vectors.
    """
    N = kps.shape[0]
    feats = np.zeros((N, 34), dtype=np.float32)

    for i in range(N):
        k = kps[i]  # (17, 3)
        if k.sum() == 0.0:
            continue  # undetected — leave as zeros

        hip_mid = (k[LEFT_HIP, :2] + k[RIGHT_HIP, :2]) / 2.0
        shoulder_mid = (k[LEFT_SHOULDER, :2] + k[RIGHT_SHOULDER, :2]) / 2.0
        torso = np.linalg.norm(shoulder_mid - hip_mid)
        if torso < 1.0:
            torso = 1.0  # degenerate skeleton — avoid division by near-zero

        xy = k[:, :2].copy()
        xy -= hip_mid
        xy /= torso
        feats[i] = xy.flatten()

    return feats


def main():
    args = parse_args()
    base = project_root()
    in_dir = base / args.input_dir
    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        kps = np.load(in_dir / f"{split}_keypoints.npy")
        det = np.load(in_dir / f"{split}_detected.npy")
        lbl = np.load(in_dir / f"{split}_labels.npy")

        feats = normalize_keypoints(kps)
        np.save(out_dir / f"{split}_embeddings.npy", feats)
        np.save(out_dir / f"{split}_labels.npy", lbl)

        detected = int(det.sum())
        total = len(det)
        print(f"{split}: {detected}/{total} with keypoints ({100 * det.mean():.1f}%) | shape: {feats.shape}")

    print(f"Done. Features saved to: {out_dir}")


if __name__ == "__main__":
    main()
