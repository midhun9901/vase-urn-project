import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from common import load_or_create_split, project_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", default="runs/v5/crops")
    parser.add_argument(
        "--sam-checkpoint",
        default="/home/hpc/iwi5/iwi5419h/vase_project/vase_urn_project/sam_vit_b.pth",
    )
    return parser.parse_args()


def get_crop(img_array, mask_generator):
    h, w = img_array.shape[:2]
    total_area = h * w
    masks = mask_generator.generate(img_array)
    valid = []
    for m in masks:
        if 0.05 * total_area < m["area"] < 0.85 * total_area:
            valid.append(m)
    if not valid:
        return Image.fromarray(img_array)
    best = max(valid, key=lambda m: m["area"])
    bbox_ints = []
    for v in best["bbox"]:
        bbox_ints.append(int(v))
    x, y, bw, bh = bbox_ints
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    return Image.fromarray(img_array[y1:y2, x1:x2])


def crop_folder(folder, base, crops_dir, mask_generator):
    rel = Path(folder).relative_to(base)
    out_folder = crops_dir / rel
    out_folder.mkdir(parents=True, exist_ok=True)
    files = []
    for f in Path(folder).iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            files.append(f)
    for f in files:
        out_path = out_folder / f.name
        if out_path.exists():
            continue
        try:
            img = np.array(Image.open(f).convert("RGB"))
            crop = get_crop(img, mask_generator)
            crop.save(out_path)
        except Exception as e:
            print(f"  Skipped {f}: {e}")
            Image.open(f).convert("RGB").save(out_path)


def main():
    args = parse_args()
    base = project_root()
    crops_dir = base / args.crops_dir
    crops_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")
    print("Loading SAM...")
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=500,
    )
    print("SAM loaded.")

    train_folders, test_folders = load_or_create_split(base)
    all_folders = train_folders + test_folders
    print(f"Processing {len(all_folders)} folders...")

    for i, folder in enumerate(all_folders):
        print(f"[{i + 1}/{len(all_folders)}] {Path(folder).name}")
        crop_folder(folder, base, crops_dir, mask_generator)

    print(f"Done. Crops saved to: {crops_dir}")


if __name__ == "__main__":
    main()
