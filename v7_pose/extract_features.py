import argparse
import os

import numpy as np
from PIL import Image

from common import image_files, load_or_create_split, project_root, save_paths

KEYPOINT_DIM = 51  # 17 keypoints * 3 (x, y, conf), normalized by image size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/pose_triplet")
    parser.add_argument("--pose-weights", default="runs/pose_model/pose_best.pt",
                        help="Vase Figure-Pose model trained by train_pose.py")
    parser.add_argument("--crop-field", action="store_true",
                        help="Detect the image field first and run pose on that crop")
    parser.add_argument("--field-weights", default="v7_pose/assets/image_field.pt",
                        help="Image-field (motif) detector used when --crop-field is set")
    parser.add_argument("--torch-home", default="/home/hpc/iwi5/iwi5419h/torch_cache")
    return parser.parse_args()


def load_model(weights):
    from ultralytics import YOLO
    print(f"Loading YOLO model: {weights}")
    return YOLO(weights)


def best_box(result):
    """Index of the highest-confidence detection, or None when there is none."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or len(boxes.conf) == 0:
        return None
    return int(boxes.conf.argmax())


def crop_to_field(field_model, img):
    """Return the image cropped to the highest-confidence image field, or the
    original image when nothing is detected."""
    result = field_model(img, verbose=False)[0]
    idx = best_box(result)
    if idx is None:
        return img
    x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy()
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def extract_keypoints(pose_model, img):
    """Run pose inference on a PIL image and return a flat 51-dim vector of
    (x, y, conf) for the 17 joints of the highest-confidence figure, with x/y
    normalised by image size. Returns zeros when no figure is detected."""
    w, h = img.size
    result = pose_model(img, verbose=False)[0]

    if result.keypoints is None or len(result.keypoints.data) == 0:
        return np.zeros(KEYPOINT_DIM, dtype=np.float32)

    idx = best_box(result)
    if idx is None:
        idx = 0

    kp = result.keypoints.data[idx].cpu().numpy()  # (17, 3): x, y, conf
    kp[:, 0] /= max(w, 1)
    kp[:, 1] /= max(h, 1)
    return kp.flatten().astype(np.float32)


def extract_split(pose_model, field_model, folders):
    embeddings = []
    labels = []
    paths = []

    for label, folder in enumerate(folders):
        for path in image_files(folder):
            try:
                img = Image.open(path).convert("RGB")
                if field_model is not None:
                    img = crop_to_field(field_model, img)
                feat = extract_keypoints(pose_model, img)
                embeddings.append(feat)
                labels.append(label)
                paths.append(path)
            except Exception as e:
                print(f"Skipped {path}: {e}")

    return np.asarray(embeddings, dtype=np.float32), np.asarray(labels, dtype=np.int64), paths


def main():
    args = parse_args()
    os.environ["TORCH_HOME"] = args.torch_home
    base = project_root()
    out = base / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train_folders, test_folders = load_or_create_split(base)
    print(f"Train folders: {len(train_folders)} | Test folders: {len(test_folders)}")

    pose_model = load_model(base / args.pose_weights)
    field_model = None
    if args.crop_field:
        field_model = load_model(base / args.field_weights)
        print("Image-field cropping: ON")

    print("Extracting train keypoints...")
    train_emb, train_lbl, train_paths = extract_split(pose_model, field_model, train_folders)
    np.save(out / "train_embeddings.npy", train_emb)
    np.save(out / "train_labels.npy", train_lbl)
    save_paths(out / "train_image_paths.txt", train_paths)
    print(f"Train embeddings: {train_emb.shape}")

    print("Extracting test keypoints...")
    test_emb, test_lbl, test_paths = extract_split(pose_model, field_model, test_folders)
    np.save(out / "test_embeddings.npy", test_emb)
    np.save(out / "test_labels.npy", test_lbl)
    save_paths(out / "test_image_paths.txt", test_paths)
    print(f"Test embeddings: {test_emb.shape}")

    nonzero = int(np.count_nonzero(np.abs(train_emb).sum(axis=1)))
    print(f"Train images with a detected figure: {nonzero}/{len(train_emb)}")

    (out / "model_name.txt").write_text(str(args.pose_weights), encoding="utf-8")
    print(f"Done. Features saved to: {out}")


if __name__ == "__main__":
    main()
