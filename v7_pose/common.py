import os
import random
from pathlib import Path

import numpy as np


SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def project_root():
    return Path(os.environ.get("VASE_PROJECT_DIR", Path(__file__).resolve().parents[1]))


def first_existing(paths):
    for path in paths:
        if path.is_dir():
            return path
    raise FileNotFoundError("None of these dataset paths exist:\n" + "\n".join(str(p) for p in paths))


def dataset_roots(base):
    pt1 = first_existing([
        base / "data" / "pairs_pt1" / "Bildpaare Teil 1",
        base / "pairs_pt1 (1)" / "Bildpaare Teil 1",
        base / "pairs_pt1" / "Bildpaare Teil 1",
    ])
    pt2 = first_existing([
        base / "data" / "pairs_pt2" / "Bildpaare-Triplets Teil 2",
        base / "pairs_pt2 (1)" / "Bildpaare-Triplets Teil 2",
        base / "pairs_pt2" / "Bildpaare-Triplets Teil 2",
    ])
    return [pt1, pt2]


def make_split(base):
    folders = []
    for root in dataset_roots(base):
        root_folders = []
        for p in root.iterdir():
            if p.is_dir():
                root_folders.append(p)
        root_folders = sorted(root_folders, key=lambda p: p.name)
        folders.extend(root_folders)

    random.seed(SEED)
    random.shuffle(folders)
    split = int(0.8 * len(folders))
    return folders[:split], folders[split:]


def load_or_create_split(base):
    train_file = base / "train_folders.txt"
    test_file = base / "test_folders.txt"
    if train_file.exists() and test_file.exists():
        train = []
        for line in train_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                train.append(Path(line))
        test = []
        for line in test_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                test.append(Path(line))
        return train, test

    train, test = make_split(base)
    train_file.write_text("\n".join(str(p) for p in train), encoding="utf-8")
    test_file.write_text("\n".join(str(p) for p in test), encoding="utf-8")
    return train, test


def image_files(folder):
    images = []
    for p in folder.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return sorted(images, key=lambda p: p.name.lower())


def l2_normalize(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def save_paths(path, paths):
    Path(path).write_text("\n".join(str(p) for p in paths), encoding="utf-8")


def compute_metrics(indices, labels):
    n = len(labels)
    ap_list = []
    acc1 = 0
    acc10 = 0

    for i in range(n):
        query_label = labels[i]
        ranked = []
        for idx in indices[i]:
            if idx != i:
                ranked.append(idx)
        ranked_labels = labels[ranked]

        if len(ranked_labels) and ranked_labels[0] == query_label:
            acc1 += 1

        if query_label in ranked_labels[:10]:
            acc10 += 1

        correct = 0
        precision_sum = 0.0
        total_relevant = int(np.sum(labels == query_label) - 1)
        for rank, lbl in enumerate(ranked_labels):
            if lbl == query_label:
                correct += 1
                precision_sum += correct / (rank + 1)

        if total_relevant > 0:
            ap_list.append(precision_sum / total_relevant)
        else:
            ap_list.append(0.0)

    return {
        "mAP": float(np.mean(ap_list)),
        "Accuracy@1": acc1 / n * 100,
        "Accuracy@10": acc10 / n * 100,
    }
