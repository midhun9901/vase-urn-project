import os
import numpy as np
from PIL import Image
import torch

BASE = os.environ.get("VASE_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
CROPS_DIR = os.path.join(BASE, "crops")
CHECKPOINT = os.path.join(BASE, "sam_vit_b.pth")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=8,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=500,
)
print("SAM loaded.")


def resize_for_sam(img_array, max_side=1024):
    h, w = img_array.shape[:2]
    if max(h, w) <= max_side:
        return img_array
    scale = max_side / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return np.array(Image.fromarray(img_array).resize((new_w, new_h)))


def get_crop(img_array):
    h, w = img_array.shape[:2]

    img_for_sam = resize_for_sam(img_array)
    sh, sw = img_for_sam.shape[:2]
    sam_area = sh * sw

    masks = mask_generator.generate(img_for_sam)
    torch.cuda.empty_cache()

    # filter on resized image area
    valid = [m for m in masks if 0.05 * sam_area < m["area"] < 0.85 * sam_area]

    if not valid:
        return Image.fromarray(img_array)

    # pick largest valid mask, scale bbox back to original image size
    best = max(valid, key=lambda m: m["area"])
    x, y, bw, bh = [int(v) for v in best["bbox"]]
    scale_x = w / sw
    scale_y = h / sh
    x = int(x * scale_x)
    y = int(y * scale_y)
    bw = int(bw * scale_x)
    bh = int(bh * scale_y)

    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def crop_folder(folder):
    rel = os.path.relpath(folder, BASE)
    out_folder = os.path.join(CROPS_DIR, rel)
    os.makedirs(out_folder, exist_ok=True)

    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for file in files:
        out_path = os.path.join(out_folder, file)
        if os.path.exists(out_path):
            continue
        path = os.path.join(folder, file)
        try:
            img = np.array(Image.open(path).convert("RGB"))
            crop = get_crop(img)
            crop.save(out_path)
        except Exception as e:
            print(f"  Skipped {path}: {e}")
            Image.open(path).convert("RGB").save(out_path)


with open(os.path.join(BASE, "train_folders.txt")) as f:
    train_folders = f.read().splitlines()

with open(os.path.join(BASE, "test_folders.txt")) as f:
    test_folders = f.read().splitlines()

all_folders = train_folders + test_folders
print(f"Processing {len(all_folders)} folders...")

for i, folder in enumerate(all_folders):
    print(f"[{i+1}/{len(all_folders)}] {os.path.basename(folder)}")
    crop_folder(folder)

print(f"Done! Crops saved to: {CROPS_DIR}")
