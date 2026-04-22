import math
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE / "data"
CROPS_ROOT = BASE / "crops_remote"
OUT_DIR = BASE / "crop_review"
ALL_OUT_DIR = BASE / "crop_review_all"
SUMMARY_CSV = BASE / "crop_review_summary.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
THUMB_W = 260
THUMB_H = 210
LABEL_H = 46
GAP = 16
ROWS_PER_SHEET = 8


def find_original(crop_path):
    rel = crop_path.relative_to(CROPS_ROOT)
    rel_text = str(rel)
    candidates = [
        BASE / rel,
        DATA_ROOT / rel_text.replace("pairs_pt1 (1)", "pairs_pt1").replace("pairs_pt2 (1)", "pairs_pt2"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def thumb(path):
    img = Image.open(path).convert("RGB")
    original_size = img.size
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), "white")
    x = (THUMB_W - img.width) // 2
    y = (THUMB_H - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas, original_size


def draw_label(draw, xy, text):
    draw.text(xy, text, fill=(20, 20, 20), font=ImageFont.load_default())


def crop_ratio(orig_size, crop_size):
    ow, oh = orig_size
    cw, ch = crop_size
    return (cw * ch) / max(1, ow * oh)


def make_sheet(items, sheet_index):
    width = THUMB_W * 2 + GAP * 3
    row_h = THUMB_H + LABEL_H
    height = row_h * len(items) + GAP
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for row, (orig_path, crop_path, ratio) in enumerate(items):
        y = GAP + row * row_h
        orig_img, orig_size = thumb(orig_path)
        crop_img, crop_size = thumb(crop_path)

        sheet.paste(orig_img, (GAP, y + LABEL_H))
        sheet.paste(crop_img, (GAP * 2 + THUMB_W, y + LABEL_H))

        rel = crop_path.relative_to(CROPS_ROOT)
        label = str(rel)
        if len(label) > 78:
            label = "..." + label[-75:]

        draw_label(draw, (GAP, y), f"Original {orig_size[0]}x{orig_size[1]}")
        draw_label(draw, (GAP * 2 + THUMB_W, y), f"SAM crop {crop_size[0]}x{crop_size[1]} ({ratio:.0%} area)")
        draw_label(draw, (GAP, y + 18), label)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"sam_crop_review_{sheet_index:02d}.jpg"
    sheet.save(out_path, quality=92)
    return out_path


def make_all_sheet(items, sheet_index):
    width = THUMB_W * 2 + GAP * 3
    row_h = THUMB_H + LABEL_H
    height = row_h * len(items) + GAP
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for row, (orig_path, crop_path, ratio) in enumerate(items):
        y = GAP + row * row_h
        orig_img, orig_size = thumb(orig_path)
        crop_img, crop_size = thumb(crop_path)

        sheet.paste(orig_img, (GAP, y + LABEL_H))
        sheet.paste(crop_img, (GAP * 2 + THUMB_W, y + LABEL_H))

        rel = crop_path.relative_to(CROPS_ROOT)
        label = str(rel)
        if len(label) > 78:
            label = "..." + label[-75:]

        draw_label(draw, (GAP, y), f"Original {orig_size[0]}x{orig_size[1]}")
        draw_label(draw, (GAP * 2 + THUMB_W, y), f"SAM crop {crop_size[0]}x{crop_size[1]} ({ratio:.0%} area)")
        draw_label(draw, (GAP, y + 18), label)

    ALL_OUT_DIR.mkdir(exist_ok=True)
    out_path = ALL_OUT_DIR / f"sam_crop_all_{sheet_index:02d}.jpg"
    sheet.save(out_path, quality=92)
    return out_path


def write_summary(pairs):
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "crop_path",
            "original_path",
            "original_width",
            "original_height",
            "crop_width",
            "crop_height",
            "crop_area_ratio",
            "size_category",
        ])
        for orig_path, crop_path, ratio in pairs:
            orig_size = Image.open(orig_path).size
            crop_size = Image.open(crop_path).size
            if ratio < 0.20:
                category = "very_small"
            elif ratio < 0.50:
                category = "medium_small"
            elif ratio > 0.90:
                category = "almost_full_image"
            else:
                category = "large"
            writer.writerow([
                str(crop_path.relative_to(CROPS_ROOT)),
                str(orig_path.relative_to(BASE)),
                orig_size[0],
                orig_size[1],
                crop_size[0],
                crop_size[1],
                f"{ratio:.4f}",
                category,
            ])


def main():
    if not CROPS_ROOT.exists():
        raise SystemExit(f"Missing crops folder: {CROPS_ROOT}")

    pairs = []
    for crop_path in sorted(CROPS_ROOT.rglob("*")):
        if crop_path.suffix.lower() not in IMAGE_EXTS:
            continue
        orig_path = find_original(crop_path)
        if not orig_path:
            continue
        try:
            orig_size = Image.open(orig_path).size
            crop_size = Image.open(crop_path).size
        except Exception:
            continue
        pairs.append((orig_path, crop_path, crop_ratio(orig_size, crop_size)))

    print(f"Comparable crop pairs: {len(pairs)}")
    if not pairs:
        return

    write_summary(pairs)

    smallest = sorted(pairs, key=lambda x: x[2])[:16]
    largest = sorted(pairs, key=lambda x: x[2], reverse=True)[:8]
    middle = sorted(pairs, key=lambda x: abs(x[2] - 0.5))[:8]
    review_items = smallest + middle + largest

    sheets = []
    for i in range(math.ceil(len(review_items) / ROWS_PER_SHEET)):
        chunk = review_items[i * ROWS_PER_SHEET : (i + 1) * ROWS_PER_SHEET]
        sheets.append(make_sheet(chunk, i + 1))

    print("Wrote review sheets:")
    for sheet in sheets:
        print(sheet)

    all_sheets = []
    for i in range(math.ceil(len(pairs) / ROWS_PER_SHEET)):
        chunk = pairs[i * ROWS_PER_SHEET : (i + 1) * ROWS_PER_SHEET]
        all_sheets.append(make_all_sheet(chunk, i + 1))

    counts = {
        "very_small": sum(1 for _, _, r in pairs if r < 0.20),
        "medium_small": sum(1 for _, _, r in pairs if 0.20 <= r < 0.50),
        "large": sum(1 for _, _, r in pairs if 0.50 <= r <= 0.90),
        "almost_full_image": sum(1 for _, _, r in pairs if r > 0.90),
    }
    print("Wrote all-crop review sheets:")
    print(ALL_OUT_DIR)
    print(f"Sheet count: {len(all_sheets)}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print("Crop area categories:")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
