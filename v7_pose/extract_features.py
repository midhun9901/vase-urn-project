import argparse
import os

import numpy as np

from common import image_files, load_or_create_split, project_root, save_paths

KEYPOINT_DIM = 51  # 17 keypoints * 3 (x, y, conf), normalized by image size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/pose_triplet")
    parser.add_argument("--model-name", default="yolov8n-pose.pt")
    parser.add_argument("--torch-home", default=os.environ.get("TORCH_HOME", "/home/woody/iwi5/iwi5419h/torch_cache"))
    return parser.parse_args()


def load_model(model_name):
    from ultralytics import YOLO
    print(f"Loading YOLOv8 pose model: {model_name}")
    return YOLO(model_name)


def extract_keypoints(model, path):
    """Run YOLOv8 pose inference and return (feature, detected).

    The feature is a flat 51-dim vector of (x, y, conf) for each of 17 joints,
    with x normalised by image width and y by image height so the vector is
    scale-invariant. Returns (zeros, False) when no detection is found — note
    that all such images collapse to the same embedding downstream.
    """
    results = model(str(path), verbose=False)
    result = results[0]

    h, w = result.orig_shape[:2]

    if result.keypoints is None or len(result.keypoints.data) == 0:
        return np.zeros(KEYPOINT_DIM, dtype=np.float32), False

    # pick the detection with the highest box confidence
    boxes = result.boxes
    if boxes is not None and len(boxes.conf) > 0:
        best = int(boxes.conf.argmax())
    else:
        best = 0

    kp = result.keypoints.data[best].cpu().numpy()  # (17, 3)
    kp[:, 0] /= w   # normalise x
    kp[:, 1] /= h   # normalise y
    return kp.flatten().astype(np.float32), True


def extract_split(model, folders):
    embeddings = []
    labels = []
    paths = []
    no_detection = 0

    for label, folder in enumerate(folders):
        files = image_files(folder)
        for path in files:
            try:
                feat, detected = extract_keypoints(model, path)
                if not detected:
                    no_detection += 1
                    print(f"No pose detected (zero vector): {path}")
                embeddings.append(feat)
                labels.append(label)
                paths.append(path)
            except Exception as e:
                print(f"Skipped {path}: {e}")

    total = len(labels)
    if total:
        print(f"No-detection images: {no_detection}/{total} ({no_detection / total * 100:.1f}%)"
              " — these share one identical zero embedding and distort retrieval among themselves")
    return np.asarray(embeddings, dtype=np.float32), np.asarray(labels, dtype=np.int64), paths


def main():
    args = parse_args()
    os.environ["TORCH_HOME"] = args.torch_home
    base = project_root()
    out = base / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train_folders, test_folders = load_or_create_split(base)
    print(f"Train folders: {len(train_folders)} | Test folders: {len(test_folders)}")

    model = load_model(args.model_name)

    print("Extracting train keypoints...")
    train_emb, train_lbl, train_paths = extract_split(model, train_folders)
    np.save(out / "train_embeddings.npy", train_emb)
    np.save(out / "train_labels.npy", train_lbl)
    save_paths(out / "train_image_paths.txt", train_paths)
    print(f"Train embeddings: {train_emb.shape}")

    print("Extracting test keypoints...")
    test_emb, test_lbl, test_paths = extract_split(model, test_folders)
    np.save(out / "test_embeddings.npy", test_emb)
    np.save(out / "test_labels.npy", test_lbl)
    save_paths(out / "test_image_paths.txt", test_paths)
    print(f"Test embeddings: {test_emb.shape}")

    (out / "model_name.txt").write_text(args.model_name, encoding="utf-8")
    print(f"Done. Features saved to: {out}")


if __name__ == "__main__":
    main()
