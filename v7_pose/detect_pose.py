import argparse

import numpy as np
from ultralytics import YOLO

from common import image_files, load_or_create_split, project_root, save_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/v7")
    parser.add_argument("--model-path", default="/home/woody/iwi5/iwi5419h/vase_project/yolov8x-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def detect_split(model, folders, conf_threshold):
    all_keypoints = []
    all_detected = []
    all_labels = []
    all_paths = []

    for label, folder in enumerate(folders):
        for img_path in image_files(folder):
            try:
                results = model(str(img_path), conf=conf_threshold, verbose=False)
                kps = results[0].keypoints
                kp_data = kps.data.cpu().numpy() if kps is not None else np.zeros((0, 17, 3))

                if kp_data.shape[0] > 0:
                    # pick detection with highest mean keypoint confidence
                    mean_confs = kps.conf.cpu().numpy().mean(axis=1)
                    best = int(np.argmax(mean_confs))
                    all_keypoints.append(kp_data[best])  # (17, 3): x, y, conf
                    all_detected.append(True)
                else:
                    all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                    all_detected.append(False)
            except Exception as e:
                print(f"Skipped {img_path}: {e}")
                all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                all_detected.append(False)

            all_labels.append(label)
            all_paths.append(img_path)

    return (
        np.asarray(all_keypoints, dtype=np.float32),
        np.asarray(all_detected, dtype=bool),
        np.asarray(all_labels, dtype=np.int64),
        all_paths,
    )


def main():
    args = parse_args()
    base = project_root()
    out = base / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLOv8-pose from: {args.model_path}")
    model = YOLO(args.model_path)

    train_folders, test_folders = load_or_create_split(base)
    print(f"Train: {len(train_folders)} folders | Test: {len(test_folders)} folders")

    print("Detecting poses in train split...")
    train_kps, train_det, train_lbl, train_paths = detect_split(model, train_folders, args.conf)
    np.save(out / "train_keypoints.npy", train_kps)
    np.save(out / "train_detected.npy", train_det)
    np.save(out / "train_labels.npy", train_lbl)
    save_paths(out / "train_image_paths.txt", train_paths)
    print(f"Train: {int(train_det.sum())}/{len(train_det)} figures detected ({100 * train_det.mean():.1f}%)")

    print("Detecting poses in test split...")
    test_kps, test_det, test_lbl, test_paths = detect_split(model, test_folders, args.conf)
    np.save(out / "test_keypoints.npy", test_kps)
    np.save(out / "test_detected.npy", test_det)
    np.save(out / "test_labels.npy", test_lbl)
    save_paths(out / "test_image_paths.txt", test_paths)
    print(f"Test: {int(test_det.sum())}/{len(test_det)} figures detected ({100 * test_det.mean():.1f}%)")

    print(f"Done. Keypoints saved to: {out}")


if __name__ == "__main__":
    main()
