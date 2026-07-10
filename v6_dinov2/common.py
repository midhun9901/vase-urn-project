import os
import random
from pathlib import Path

import numpy as np


# Fixed seed so train/test split is reproducible across runs and machines
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def project_root():
    # Use env var override on HPC; fall back to two levels up from this file (project root)
    return Path(os.environ.get("VASE_PROJECT_DIR", Path(__file__).resolve().parents[1]))


def first_existing(paths):
    # Cluster mounts things differently depending on the node, so try each candidate path
    for path in paths:
        if path.is_dir():
            return path
    raise FileNotFoundError("None of these dataset paths exist:\n" + "\n".join(str(p) for p in paths))


def dataset_roots(base):
    # Resolve the actual disk location of both dataset parts (pt1 = book illustrations, pt2 = photo/illustration pairs)
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
    # Collect every vase folder from both dataset parts, sort for determinism, then shuffle
    folders = []
    for root in dataset_roots(base):
        root_folders = []
        for p in root.iterdir():
            if p.is_dir():
                root_folders.append(p)
        root_folders = sorted(root_folders, key=lambda p: p.name)
        folders.extend(root_folders)

    # Split at folder level (not image level) so all images of a vase go to the same partition
    random.seed(SEED)
    random.shuffle(folders)
    split = int(0.8 * len(folders))
    return folders[:split], folders[split:]  # 80% train, 20% test


def read_split_file(path, base):
    folders = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        # Older split files stored absolute paths; new ones are relative to the project root
        if not p.is_absolute():
            p = base / p
        if not p.is_dir():
            raise FileNotFoundError(
                f"Split folder does not exist: {p}\n"
                f"(listed in {path}; delete train_folders.txt/test_folders.txt to regenerate the split)"
            )
        folders.append(p)
    return folders


def load_or_create_split(base):
    train_file = base / "train_folders.txt"
    test_file = base / "test_folders.txt"

    # Reuse existing split files so all versions compare against the same train/test partition
    if train_file.exists() and test_file.exists():
        return read_split_file(train_file, base), read_split_file(test_file, base)

    # First run: create the split and persist it, relative to the project root so it is portable
    train, test = make_split(base)
    train_file.write_text("\n".join(str(p.relative_to(base)) for p in train), encoding="utf-8")
    test_file.write_text("\n".join(str(p.relative_to(base)) for p in test), encoding="utf-8")
    return train, test


def image_files(folder):
    # Return only image files, sorted so ordering is consistent across OS/filesystem
    images = []
    for p in folder.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return sorted(images, key=lambda p: p.name.lower())


def l2_normalize(x, eps=1e-12):
    # Divide each row vector by its L2 norm; eps prevents division by zero for zero vectors
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def save_paths(path, paths):
    # Write one absolute image path per line, used later to map retrieval results back to files
    Path(path).write_text("\n".join(str(p) for p in paths), encoding="utf-8")


def compute_metrics(indices, labels):
    # indices: (N_queries, N_gallery) ranked by similarity; labels: integer class per image
    n = len(labels)
    ap_list = []
    acc1 = 0
    acc10 = 0

    for i in range(n):
        query_label = labels[i]
        # Remove the query itself from its own ranked list (it's always distance=0)
        ranked = []
        for idx in indices[i]:
            if idx != i:
                ranked.append(idx)
        ranked_labels = labels[ranked]

        # Accuracy@1: top result matches the query class
        if len(ranked_labels) and ranked_labels[0] == query_label:
            acc1 += 1

        # Accuracy@10: at least one of top-10 results matches the query class
        if query_label in ranked_labels[:10]:
            acc10 += 1

        # Average Precision: rewards finding relevant images early in the ranked list
        correct = 0
        precision_sum = 0.0
        total_relevant = int(np.sum(labels == query_label) - 1)  # exclude the query itself
        for rank, lbl in enumerate(ranked_labels):
            if lbl == query_label:
                correct += 1
                precision_sum += correct / (rank + 1)  # precision at this rank position

        if total_relevant > 0:
            ap_list.append(precision_sum / total_relevant)
        else:
            ap_list.append(0.0)

    return {
        "mAP": float(np.mean(ap_list)),
        "Accuracy@1": acc1 / n * 100,
        "Accuracy@10": acc10 / n * 100,
    }
