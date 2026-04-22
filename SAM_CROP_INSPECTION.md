# SAM Crop Visual Inspection

**Date:** April 22, 2026  
**Source folder:** `crops_remote/`  
**Original data folder:** `data/`  
**Valid SAM job:** `1581893`  
**Total SAM crop files checked:** 250  

---

## What Was Checked

Each SAM crop was matched against its original image using the same relative folder path and filename.

Example:

```text
Original:
data/pairs_pt1/Bildpaare Teil 1/045/hamilton1791bd1_0215.jpg

SAM crop:
crops_remote/pairs_pt1 (1)/Bildpaare Teil 1/045/hamilton1791bd1_0215.jpg
```

The folder structure and filenames are intentionally the same. The difference is inside the image: the crop should contain only the region selected by SAM.

---

## Generated Review Files

Full side-by-side sheets were generated here:

```text
crop_review_all/
```

There are 32 review sheets:

```text
sam_crop_all_01.jpg
...
sam_crop_all_32.jpg
```

Each row shows:

```text
original image | SAM crop
```

A CSV summary was also generated:

```text
crop_review_summary.csv
```

It contains original dimensions, crop dimensions, crop area ratio, and a size category for every crop.

---

## Crop Size Summary

| Category | Meaning | Count |
|----------|---------|-------|
| very_small | Crop area is less than 20% of original image | 25 |
| medium_small | Crop area is 20-50% of original image | 60 |
| large | Crop area is 50-90% of original image | 77 |
| almost_full_image | Crop area is more than 90% of original image | 88 |

This distribution shows that SAM was inconsistent:

- Some crops are too small and remove important context.
- Some crops are almost the full image, so they do not meaningfully crop anything.
- Many crops are in between, but visual inspection is needed to know whether they selected the right region.

---

## Most Suspicious Very Small Crops

These should be inspected first because they likely removed important visual information:

| Crop Area | Crop Size | Original Size | File |
|-----------|-----------|---------------|------|
| 8.7% | 548x388 | 1121x2171 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.01/Bild3.jpg` |
| 9.4% | 2273x175 | 2273x1862 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.01/inghirami1852_0242.jpg` |
| 9.6% | 2911x218 | 2911x2277 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.26/gerhard1850bd2_0053.jpg` |
| 9.6% | 2911x218 | 2911x2277 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.28/gerhard1850bd2_0054.jpg` |
| 10.4% | 154x206 | 591x518 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.04/BAPD 310305, ATHENIAN, Boston (MA), Museum of Fine Arts, 00.330.jpg` |
| 10.8% | 4294x317 | 4294x2950 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.07/gerhard1840bd1_0253.jpg` |
| 10.8% | 1677x269 | 1677x2500 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.05/London British Museum 1836.2-24.10.jpg` |
| 11.0% | 269x102 | 417x600 | `pairs_pt2 (1)/Bildpaare-Triplets Teil 2/Nr.27/BAPD 213554, ATHENIAN, London, British Museum, London, British Museum, 1978,0411.5.jpg` |

Observed pattern from the review sheets:

- Some crops select only a watermark or page strip.
- Some crops select only the vase foot/base.
- Some crops select only the top rim or a narrow horizontal band.
- Some crops select a small detail rather than the full vase or painted scene.

These are likely harmful for retrieval because they remove the visual context needed to match vase pairs.

---

## Almost Full-Image Crops

88 crops are more than 90% of the original image area. These are not necessarily harmful, but they also do not give the intended benefit of cropping.

Examples:

| Crop Area | File |
|-----------|------|
| 100.0% | `pairs_pt1 (1)/Bildpaare Teil 1/002/hamilton1791bd1_0172.jpg` |
| 100.0% | `pairs_pt1 (1)/Bildpaare Teil 1/004/hamilton1791bd1_0174.jpg` |
| 100.0% | `pairs_pt1 (1)/Bildpaare Teil 1/007/hamilton1791bd1_0177.jpg` |
| 100.0% | `pairs_pt1 (1)/Bildpaare Teil 1/037/hamilton1791bd1_0207.jpg` |
| 100.0% | `pairs_pt1 (1)/Bildpaare Teil 1/043/hamilton1791bd1_0213.jpg` |

This means the SAM preprocessing was inconsistent: some images were cropped aggressively, while others were effectively unchanged.

---

## Conclusion

The visual crop inspection supports the metric result.

Valid SAM result:

```text
mAP        : 18.89%
Accuracy@1 : 13.46%
Accuracy@10: 69.23%
```

This is much worse than full-image Triplet Loss:

```text
mAP        : 54.04%
Accuracy@1 : 42.31%
Accuracy@10: 92.31%
```

Main reason:

> The current SAM heuristic does not consistently select the retrieval-relevant region. It sometimes crops irrelevant strips/details, sometimes keeps almost the full image, and sometimes removes useful vase/figure context.

Therefore, simple SAM cropping should not be used as the main retrieval preprocessing method.

---

## Recommended Report Wording

> A visual inspection of the 250 SAM crops showed that the automatic crop-selection heuristic was inconsistent. Some crops selected very small or irrelevant regions such as page strips, vase rims, bases, or decorative details, while many other crops remained almost identical to the original full image. This explains the large drop in retrieval performance from 54.04% mAP for full-image Triplet Loss to 18.89% mAP for SAM-cropped Triplet Loss. The result suggests that zero-shot SAM segmentation alone is insufficient for reliable retrieval preprocessing on this stylized archaeological vase dataset.
