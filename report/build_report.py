# -*- coding: utf-8 -*-
"""Build the master-project report PDF in a LaTeX (Computer Modern) style."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, KeepTogether, PageBreak,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
FONTDIR = os.path.join(HERE, "fonts", "cm-unicode-0.7.0")
os.makedirs(FIG, exist_ok=True)

# ---- register Computer Modern (CMU Serif) for both reportlab and matplotlib --
_FACES = {"CMU": "cmunrm.ttf", "CMU-Bold": "cmunbx.ttf",
          "CMU-Italic": "cmunti.ttf", "CMU-BoldItalic": "cmunbi.ttf"}
for _name, _fn in _FACES.items():
    pdfmetrics.registerFont(TTFont(_name, os.path.join(FONTDIR, _fn)))
    fm.fontManager.addfont(os.path.join(FONTDIR, _fn))
pdfmetrics.registerFontFamily("CMU", normal="CMU", bold="CMU-Bold",
                              italic="CMU-Italic", boldItalic="CMU-BoldItalic")

INK = colors.black
STEEL = colors.HexColor("#36618e")
GREY = colors.HexColor("#555555")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "figure.dpi": 220,
})

# ---- data (clean held-out results only) -------------------------------------
VERS = ["V1", "V2", "V3", "V6a", "V6b", "V7"]
MAP = [15.51, 50.70, 58.87, 32.66, 78.26, 36.93]
A1 = [15.09, 43.40, 52.83, 26.42, 77.36, 33.96]
A10 = [30.19, 84.91, 88.68, 83.02, 100.0, 67.92]


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 1.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.4); ax.axis("off")
    stages = [
        ("Feature\nExtraction", "backbone:\nResNet50 / DINOv2\n/ YOLO pose"),
        ("Projection\nTraining", "MLP + Triplet /\nProxyAnchor loss"),
        ("Retrieval", "FAISS index,\nfull ranking"),
        ("Evaluation", "mAP, Acc@1,\nAcc@10"),
    ]
    w, h, gap = 2.15, 1.25, 0.33
    x = 0.25
    for i, (tt, sub) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x, 0.7), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.0, edgecolor="#1f2a44", facecolor="#eef2f7"))
        ax.text(x + w / 2, 1.55, tt, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#1f2a44")
        ax.text(x + w / 2, 1.0, sub, ha="center", va="center",
                fontsize=6.6, color="#555555")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.32), (x + w + gap, 1.32),
                         arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
                         color="#36618e"))
        x += w + gap
    plt.tight_layout(pad=0.2)
    p = os.path.join(FIG, "fig1_pipeline.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_results():
    import numpy as np
    x = np.arange(len(VERS)); bw = 0.27
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.bar(x - bw, MAP, bw, label="mAP", color="#36618e")
    ax.bar(x, A1, bw, label="Acc@1", color="#7fa6cb")
    ax.bar(x + bw, A10, bw, label="Acc@10", color="#cdd9e8")
    ax.set_ylabel("score (%)"); ax.set_ylim(0, 108)
    ax.set_xticks(x); ax.set_xticklabels(VERS)
    for i, v in enumerate(MAP):
        ax.text(i - bw, v + 1.4, f"{v:.0f}", ha="center", fontsize=6.5, color="#1f2a44")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
    plt.tight_layout(pad=0.3)
    p = os.path.join(FIG, "fig2_results.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_qualitative():
    """Query + top-5 retrieved for V6b, drawn from the saved run artifacts.
    Falls back to the existing PNG if the run outputs are not present."""
    out = os.path.join(FIG, "fig3_qualitative.png")
    proot = os.path.dirname(HERE)
    run = os.path.join(proot, "runs", "bmp_fix", "dinov2_triplet")
    try:
        import numpy as np
        from PIL import Image
        from matplotlib.patches import Patch
        def local(p):
            for m, sub in [("Bildpaare-Triplets Teil 2", "data/pairs_pt2"),
                           ("Bildpaare Teil 1", "data/pairs_pt1")]:
                if m in p:
                    return os.path.join(proot, sub, p[p.index(m):]).replace("\\", "/")
            return p
        paths = [local(x) for x in open(os.path.join(run, "test_image_paths.txt"),
                                        encoding="utf-8").read().splitlines()]
        idx = np.load(os.path.join(run, "retrieval_indices.npy"))
        lab = np.load(os.path.join(run, "test_labels.npy"))
        nimg = len(paths)
        ranked = lambda q: [j for j in idx[q] if j != q]
        rel = {q: int((lab == lab[q]).sum() - 1) for q in range(nimg)}
        sel, seen = [], set()
        for q in sorted([q for q in range(nimg) if rel[q] >= 1], key=lambda q: -rel[q]):
            if lab[q] in seen:
                continue
            sel.append(q); seen.add(lab[q])
            if len(sel) == 5:
                break
        K = 5
        def load(p):
            im = Image.open(p).convert("RGB"); im.thumbnail((240, 240)); return im
        fq, ax = plt.subplots(len(sel), K + 1, figsize=(2.0 * (K + 1), 2.05 * len(sel)))
        for r, q in enumerate(sel):
            rk = ranked(q)[:K]; found = sum(1 for j in rk if lab[j] == lab[q])
            cells = [(q, "query", "#1f2a44")] + [(j, f"#{c+1}",
                     ("#2e7d32" if lab[j] == lab[q] else "#c62828")) for c, j in enumerate(rk)]
            for cix, (j, title, color) in enumerate(cells):
                a = ax[r, cix]
                try: a.imshow(load(paths[j]))
                except Exception: a.text(.5, .5, "?", ha="center")
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values(): s.set_color(color); s.set_linewidth(3.2)
                if r == 0: a.set_title(title, fontsize=10, color=color)
            ax[r, 0].set_ylabel(f"{found}/{rel[q]} found", fontsize=8.5, color="#1f2a44")
        fq.legend(handles=[Patch(color="#2e7d32", label="same folder class (relevant)"),
                           Patch(color="#c62828", label="different folder class")],
                  loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout(rect=[0, 0.03, 1, 1.0])
        fq.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fq)
    except Exception as e:
        print("fig_qualitative: reusing existing PNG (", e, ")")
    return out


f1, f2, f3 = fig_pipeline(), fig_results(), fig_qualitative()

# ---- paragraph styles (Computer Modern) --------------------------------------
body = ParagraphStyle("body", fontName="CMU", fontSize=10.5, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=7, textColor=INK)
h1 = ParagraphStyle("h1", fontName="CMU-Bold", fontSize=13.5, leading=16,
                    spaceBefore=15, spaceAfter=6, textColor=INK, keepWithNext=1)
ctitle = ParagraphStyle("ctitle", fontName="CMU-Bold", fontSize=25, leading=30,
                        alignment=TA_CENTER, textColor=INK)
csub = ParagraphStyle("csub", fontName="CMU", fontSize=14.5, leading=19,
                      alignment=TA_CENTER, textColor=INK)
cmeta = ParagraphStyle("cmeta", fontName="CMU", fontSize=12.5, leading=17,
                       alignment=TA_CENTER, textColor=INK)
csmall = ParagraphStyle("csmall", fontName="CMU", fontSize=11, leading=16,
                        alignment=TA_CENTER, textColor=INK)
absth = ParagraphStyle("absth", fontName="CMU-Bold", fontSize=11.5, leading=14,
                       alignment=TA_CENTER, textColor=INK, spaceAfter=5)
abst = ParagraphStyle("abst", fontName="CMU", fontSize=10, leading=14,
                      alignment=TA_JUSTIFY, textColor=INK,
                      leftIndent=1.1 * cm, rightIndent=1.1 * cm)
caption = ParagraphStyle("caption", fontName="CMU", fontSize=9, leading=12,
                         alignment=TA_CENTER, textColor=INK, spaceBefore=3, spaceAfter=12)
reference = ParagraphStyle("reference", parent=body, fontSize=9.2, leading=12,
                           leftIndent=0.65 * cm, firstLineIndent=-0.65 * cm,
                           spaceAfter=5, alignment=TA_JUSTIFY)
cell = ParagraphStyle("cell", fontName="CMU", fontSize=9, leading=11)
cellc = ParagraphStyle("cellc", fontName="CMU", fontSize=9, leading=11, alignment=TA_CENTER)
hcell = ParagraphStyle("hcell", fontName="CMU-Bold", fontSize=9, leading=11, alignment=TA_CENTER)
cellb = ParagraphStyle("cellb", fontName="CMU-Bold", fontSize=9, leading=11)
cellbc = ParagraphStyle("cellbc", fontName="CMU-Bold", fontSize=9, leading=11, alignment=TA_CENTER)


def P(t, s=body): return Paragraph(t, s)


def heading(num, txt): return Paragraph(f"{num}&nbsp;&nbsp;{txt}", h1)


def cap(t):
    if ". " in t:
        lab, rest = t.split(". ", 1)
        t = f"<b>{lab}.</b> {rest}"
    return Paragraph(t, caption)


def img(path, width_cm, capt):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = width_cm * cm; hh = w * ih / iw
    return KeepTogether([Image(path, width=w, height=hh), cap(capt)])


story = []

# ---- cover page --------------------------------------------------------------
story += [
    Spacer(1, 3.2 * cm),
    P("Metric Learning for Archaeological Vase Retrieval", ctitle),
    Spacer(1, 0.25 * cm),
    P("Master Project Report", csub),
    Spacer(1, 0.4 * cm),
    HRFlowable(width="55%", thickness=0.8, color=INK, spaceBefore=2, spaceAfter=16),
    P("Midhun Somanunnithan", cmeta),
    Spacer(1, 0.45 * cm),
    P("Supervised by Mathias Zinnen", csmall),
    Spacer(1, 5.5 * cm),
    P("Friedrich-Alexander-Universit&auml;t Erlangen-N&uuml;rnberg", csmall),
    P("September 2026", csmall),
    PageBreak(),
]

# ---- abstract ----------------------------------------------------------------
story += [
    Spacer(1, 0.2 * cm),
    P("Abstract", absth),
    P("This report describes a content-based image retrieval system for archaeological vase images, built around metric learning. Given a query image, the system ranks a database by visual similarity in a learned embedding space. Relevance is defined by the dataset's folder-level classes. I compare ResNet50 and DINOv2 features with several metric-learning losses, and I investigate background removal and pose-based descriptions. Among the clean held-out experiments, DINOv2 features adapted with a triplet-loss projection perform best, reaching 78.26% mean average precision and retrieving a relevant match within the top ten for every test query. A later audit found that two intended trained SAM-crop runs did not load their projection heads and that the vase-specific pose detector had been trained on images overlapping the retrieval test set. Those runs are retained as diagnostics but excluded from comparative conclusions. The validated experiments share one folder-level split and one full-ranking evaluator.", abst),
    Spacer(1, 0.3 * cm),
]

# ---- 1 introduction ----------------------------------------------------------
story += [heading("1", "Introduction"),
P("This project studies content-based image retrieval for archaeological vase imagery. It forms part of a broader research setting concerned with provenance, the reception of antiquity, research networks, and the comparison of image fields, figures, and figure constellations across photographs and printed reproductions. The concrete task evaluated here is narrower: given a photograph or drawing of a vase, rank the other images assigned to the same dataset folder ahead of images from other folders."),
P("The task is a retrieval problem rather than a closed-set classification problem. There are only a few images per folder-level class, and at test time the goal is to rank a database by similarity to a query. This makes metric learning suitable: the model is trained to place images with the same folder label close together and images with different labels farther apart. The evaluation therefore measures recovery of same-folder images; it does not independently establish that every folder represents one physical object or one uniquely defined motif."),
P("The work was carried out as a sequence of versioned experiments, from a plain baseline up to a strong final method. Exploratory runs are retained where they shaped the design, but only experiments that preserve the held-out protocol are used for comparative conclusions.")]

# ---- 2 objectives ------------------------------------------------------------
story += [heading("2", "Objectives and Scope"),
P("The objectives were to build a reproducible retrieval pipeline that turns vase images into embeddings and evaluates ranking quality with standard metrics; to compare feature backbones and loss functions in order to find a representation that works well on a small dataset with few images per folder class; to test two further ideas, isolating the vase from its background, which I tried after the early results, and using the pose of the painted figures as a similarity signal, which the project material pointed to; and to run the experiments on the university HPC cluster with one SLURM job script per version, so that each experiment can be repeated."),
P("The evaluated scope is whole-image retrieval on the two vase collections, pairs_pt1 and pairs_pt2. The urn collection, urns_small, was provided as a possible second task but was deliberately deferred because its object-based folders do not yet give reliable motif labels. No urn experiment or result is claimed in this report. The report also does not attempt instance segmentation, text-based search, or cross-modal matching. The evaluation is intrinsic, using a held-out portion of the vase data rather than an external benchmark.")]

# ---- 3 preparation and background -------------------------------------------
story += [heading("3", "Preparation and Background"),
P("During the first few weeks, I combined background study with a hands-on implementation of the basic retrieval system. Starting from the first baseline, I researched the topics as they became necessary: fundamentals of deep learning and neural-network training, followed by practical PyTorch work [14] with tensors, training loops, datasets and data loaders, training and evaluation modes, and device-agnostic code for laptop and GPU execution. At the same time, I studied the metric-learning material recommended for the project and set up access to the university HPC cluster, including SSH, the module and conda environment, and SLURM job submission."),
P("This learn-by-building process shaped the rest of the work. The project was run as a sequence of versioned experiments rather than as a single model. For the validated comparisons, each version changes one principal decision, retains the same data split and evaluator, and records its result. Two implementation faults found during the final audit are documented separately rather than treated as controlled comparisons."),
P("<b>Metric learning losses.</b> Triplet margin loss pulls an anchor towards a positive of the same class and pushes it away from a negative of a different class, subject to a margin [3]. It is paired with the multi-similarity miner [4], which selects informative examples. ProxyAnchor replaces explicit triplets with learnable class proxies [5], while ArcFace is an angular-margin classification loss [6]."),
P("<b>Embedding and retrieval.</b> After training, each image is mapped to a vector, the vectors are L2-normalised, and retrieval is performed with an exact flat L2 index in FAISS [10]. Because the test set is small, I search the full ranking rather than a fixed top-k, which removes an artefact discussed in Section 8."),
P("<b>Evaluation metrics.</b> I report mean average precision (mAP), Accuracy@1, and Accuracy@10. mAP rewards placing all relevant items high in the ranking and is the primary metric. Accuracy@1 is the fraction of queries whose nearest neighbour is correct, and Accuracy@10 is the fraction with at least one correct item in the top ten. The query itself is removed from its own ranking before scoring."),
P("<b>Backbones.</b> ResNet50 is a residual convolutional network pretrained on ImageNet [1,16]. DINOv2 ViT-S/14 is a self-supervised vision transformer designed to provide transferable visual features [2]. The pose branch uses Ultralytics YOLOv8 and YOLO11 pose models [8,9], represented with the 17-keypoint COCO layout [12,13].")]

# ---- 4 dataset ---------------------------------------------------------------
story += [heading("4", "Dataset"),
P("The evaluated vase data consists of two parts provided for the project. The first part, pairs_pt1, contains 71 folders of book-illustration vases, and the second, pairs_pt2, contains 30 folders of mixed photographs and illustrations, for 101 folders in total. Following the supplied organisation, each folder is treated as one retrieval class and all images inside it are considered relevant to one another. The loader accepts JPG, JPEG, PNG, and BMP files, giving 253 images across the 101 folder classes. This is a small dataset by deep-learning standards and was a constant constraint on method choice."),
P("The split is made at the level of folders, not images, so that every image carrying a given folder label falls entirely into either the training or the test set. This prevents same-class images from appearing on both sides. With seed 42, the approximate 80/20 split contains 80 training classes (200 images) and 21 test classes (53 images). Every test class has at least two images, so every query has at least one relevant non-self result. The split is computed once, written to disk, and reused by every retrieval stage."),
P("For the pose experiments a second annotated collection was provided. It contains 238 images with Figure-Pose labels in the COCO 17-keypoint layout [12,13], created by Basit, plus image-field annotations and separate pose-test folders. A final audit found that this pool contains 51 byte-identical retrieval-test images and one additional same-named counterpart. In the original deterministic 85/15 split, 44 of these 52 overlapping images entered pose-model training and eight entered pose validation. Consequently, the recorded V7b retrieval score is not a clean held-out result. The corrected training script now excludes retrieval-test matches before creating the pose train/validation split; a new V7b score requires retraining.")]

# (Background theory is folded into Section 3, Preparation and Background.)

# ---- 5 method + fig1 ---------------------------------------------------------
story += [heading("5", "System and Method"),
P("Every version follows the same four-stage pipeline, shown in Figure 1, and the stages share one common module so that results stay comparable."),
img(f1, 15.5, "Figure 1. The shared four-stage retrieval pipeline. Only the backbone and the loss change between versions; the split, the index, and the evaluator are identical."),
P("<b>Feature extraction.</b> ResNet50 and DINOv2 inputs are resized to 256 pixels on the shorter side, centre-cropped to 224 by 224, converted to tensors, and normalised with the ImageNet channel statistics [16]. The ResNet50 classifier is removed to yield a 2,048-dimensional vector. DINOv2 ViT-S/14 yields a 384-dimensional vector. The frozen backbone outputs, labels, and file paths are stored as arrays."),
P("<b>Projection training.</b> A multilayer perceptron maps input to hidden to output dimensions with a ReLU between the two linear layers. The output is L2-normalised before the loss. All projection heads train for 80 epochs with Adam [15], learning rate 0.001, batch size 32, and an M-per-class sampler with two samples per folder class. Triplet runs use the default PyTorch Metric Learning triplet margin and the multi-similarity miner [11]. ProxyAnchor uses margin 0.1 and alpha 32. The DINOv2 layout is 384 to 512 to 128; the ResNet50 layout is 2,048 to 512 to 128; and the pose layout is 51 to 128 to 64."),
P("<b>Retrieval.</b> The test embeddings are optionally passed through the trained projection, L2-normalised, indexed with FAISS, and searched against themselves to produce a full ranking per query. <b>Evaluation.</b> The shared metric routine computes mAP, Accuracy@1, and Accuracy@10 from the ranking and the labels."),
P("Seed 42 fixes the data split and the Python, NumPy, and PyTorch random states; deterministic cuDNN settings are enabled for projection training. These controls improve repeatability, although small numerical differences can remain across hardware and software environments. The results are single-run measurements on one split, so close scores should not be interpreted as statistically significant differences.")]

# ---- 6 experiments -----------------------------------------------------------
story += [heading("6", "Experimental Methodology"),
P("The experiments are numbered as versions. Table 1 and Figure 2 contain only runs that preserve the held-out protocol. Table 2 separately records three diagnostic runs whose implementation or data independence was later found to be invalid."),
P("<b>V1, Raw ResNet50 baseline.</b> Features are taken directly from an ImageNet-pretrained ResNet50 with no metric learning, indexed, and evaluated. This sets the floor. <b>V2, ResNet50 with triplet loss.</b> The same features are passed through the trained MLP with triplet loss and the multi-similarity miner. <b>V3, ResNet50 with ProxyAnchor loss.</b> Identical to V2 except that the loss is replaced by ProxyAnchor, chosen because the dataset is small."),
P("<b>V4 and V5, SAM-crop diagnostics.</b> The Segment Anything Model [7] generates masks with ViT-B, 16 points per side, predicted-IoU threshold 0.88, stability threshold 0.95, and minimum region area 500 pixels. Masks covering 5% to 85% of the image are retained; the largest is cropped with ten pixels of padding. V4 was intended to apply an ArcFace head (margin 28.6 degrees, scale 64), and V5 a triplet head. The recorded retrieval commands omitted the saved model path, however, so both scores measure the same raw ResNet50 features on SAM crops. They do not compare ArcFace with triplet loss and are diagnostic only."),
P("<b>V6a, DINOv2 without fine-tuning.</b> The backbone is switched to DINOv2 ViT-S/14 and the features are ranked directly, with no projection training. <b>V6b, DINOv2 with triplet loss.</b> The DINOv2 features are passed through the MLP trained with triplet loss, combining a strong representation with task-specific adaptation."),
P("<b>V7, Generic pose keypoints.</b> A different idea: describe each image by the pose of its painted figure rather than by its appearance. When a figure is detected, the pose model returns seventeen COCO keypoints [12,13]; I take the single highest-confidence figure and normalise its (x, y, confidence) triples, dividing x by the image width and y by the height, into one 51-dimensional vector. Images without a detection receive an all-zero vector. This keypoint vector is the <i>only</i> input to retrieval: no appearance, colour, or texture information is combined with it. It is then passed through the same pipeline as the other versions, an MLP (51 to 128 to 64) trained with triplet loss and ranked with FAISS. V7 uses an off-the-shelf YOLOv8 pose model [8], which is trained on human bodies."),
P("<b>V7b, Vase-specific pose diagnostic.</b> To replace the generic detector, YOLO11 [9] was fine-tuned for 100 epochs at 640-pixel input resolution and batch size 16 on Basit's Figure-Pose labels. The same 51-dimensional keypoint representation and triplet projection were then used. The original 238-image pose pool was randomly divided 85/15 without first excluding retrieval-test images. As detailed in Section 4, this contaminated the detector training and validation subsets. V7b therefore tests pipeline feasibility but does not isolate the effect of domain-specific pose training. The corrected procedure excludes all retrieval-test images before the pose train/validation split and must be rerun before a clean comparison can be made.")]

# ---- 7 results + tables + figures -------------------------------------------
tbl_head = [Paragraph(x, hcell) for x in ["Version", "Method", "mAP", "Acc@1", "Acc@10"]]
rows = [
    ("V1", "Raw ResNet50, no MLP", "15.51%", "15.09%", "30.19%"),
    ("V2", "ResNet50 + MLP + Triplet", "50.70%", "43.40%", "84.91%"),
    ("V3", "ResNet50 + MLP + ProxyAnchor", "58.87%", "52.83%", "88.68%"),
    ("V6a", "DINOv2 features, no training", "32.66%", "26.42%", "83.02%"),
    ("V6b", "DINOv2 + MLP + Triplet", "78.26%", "77.36%", "100.00%"),
    ("V7", "Generic YOLOv8 pose + Triplet", "36.93%", "33.96%", "67.92%"),
]
data = [tbl_head]
for r in rows:
    isbest = r[0] == "V6b"
    sc, scc = (cellb, cellbc) if isbest else (cell, cellc)
    data.append([Paragraph(r[0], scc), Paragraph(r[1], sc),
                 Paragraph(r[2], scc), Paragraph(r[3], scc), Paragraph(r[4], scc)])
tbl = Table(data, colWidths=[1.6 * cm, 8.4 * cm, 2.1 * cm, 2.0 * cm, 2.2 * cm])
tbl.setStyle(TableStyle([
    ("LINEABOVE", (0, 0), (-1, 0), 1.1, INK),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
    ("LINEBELOW", (0, -1), (-1, -1), 1.1, INK),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
]))

diag_head = [Paragraph(x, hcell) for x in
             ["Version", "Recorded run", "Hardware", "mAP", "Acc@1", "Acc@10", "Exclusion reason"]]
diag_rows = [
    ("V4", "SAM crop + raw ResNet50", "A100 (TinyGPU)", "11.55%", "9.62%", "30.77%",
     "ArcFace projection not loaded"),
    ("V5", "SAM crop + raw ResNet50", "A100 (TinyGPU)", "11.55%", "9.62%", "30.77%",
     "Triplet projection not loaded"),
    ("V7b", "Vase YOLOv11 pose + Triplet", "RTX 3080 (TinyGPU)", "45.55%", "38.46%", "67.31%",
     "Pose training pool overlaps retrieval test set"),
]
diag_data = [diag_head]
for r in diag_rows:
    diag_data.append([Paragraph(r[0], cellc), Paragraph(r[1], cell),
                      Paragraph(r[2], cellc), Paragraph(r[3], cellc),
                      Paragraph(r[4], cellc), Paragraph(r[5], cellc), Paragraph(r[6], cell)])
diag_tbl = Table(diag_data, colWidths=[1.6 * cm, 3.5 * cm, 2.3 * cm, 1.4 * cm, 1.4 * cm,
                                       1.5 * cm, 4.6 * cm])
diag_tbl.setStyle(TableStyle([
    ("LINEABOVE", (0, 0), (-1, 0), 1.1, INK),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
    ("LINEBELOW", (0, -1), (-1, -1), 1.1, INK),
    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f4f4")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
    ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
]))

story += [heading("7", "Results"),
P("Table 1 and Figure 2 report the six runs that use the same folder-level test split and do not train any component on retrieval-test images."),
KeepTogether([tbl, Spacer(1, 0.15 * cm),
              cap("Table 1. Clean held-out retrieval results on the 53 test images (21 folder classes) of the complete 253-image dataset. All six runs were executed in the same hardware and software environment (Section 9). V6b is the strongest evaluated method.")]),
img(f2, 15.5, "Figure 2. Clean held-out retrieval scores. DINOv2 with a triplet-trained projection (V6b) performs best on this split."),
P("V6b reaches 78.26% mAP, 77.36% Accuracy@1, and 100% Accuracy@10. These are single-run results on 53 queries from 21 held-out folder classes. The qualitative examples in Figure 3 show successful matches across drawings and photographs as well as remaining false positives."),
img(f3, 15.0, "Figure 3. Qualitative retrieval for V6b. Each row is a query (left) followed by its top five retrieved images; green borders mark the same folder class and red borders a different folder class. Once every relevant image has been retrieved, later irrelevant items do not reduce average precision."),
P("Table 2 preserves the numerical outputs of three earlier flawed runs for auditability. They were recorded on the NHR@FAU TinyGPU cluster with the earlier 250-image loader, were not rerun after BMP support was added, and are neither included in Figure 2 nor used to rank methods."),
KeepTogether([diag_tbl, Spacer(1, 0.15 * cm),
              cap("Table 2. Diagnostic results excluded from comparative claims. V4 and V5 did not apply their trained projections; V7b used overlapping pose-training and retrieval-test images.")])]

# ---- 8 discussion ------------------------------------------------------------
story += [heading("8", "Discussion"),
P("Metric learning is strongly beneficial in the two matched comparisons available here. ResNet50 rises from 15.51% mAP in V1 to 50.70% in V2, while DINOv2 rises from 32.66% in V6a to 78.26% in V6b. These results support the use of a task-adapted projection in this pipeline; they do not establish that metric learning is necessary for every retrieval formulation."),
P("With the ResNet50 backbone fixed, ProxyAnchor in V3 scores 8.17 percentage points above triplet loss in V2. Accuracy@1 is 9.43 points higher and Accuracy@10 is 3.77 points higher. Because each method was run once on one split, these differences are descriptive rather than evidence that one loss is generally superior. The larger V2-to-V6b gap still suggests that the feature representation was the more influential choice in this experiment."),
P("The SAM-crop diagnostic gives only limited evidence. With the earlier 250-image loader, raw ResNet50 features on the SAM crops scored 11.55% mAP, compared with 15.51% for raw full images in the matching V1 run with the same loader. This suggests that the chosen masking and cropping rule may remove useful context. It does not establish how ArcFace or triplet projections perform on cropped features, because those projections were not loaded during retrieval. Corrected V4 and V5 runs are required before drawing a trained-model conclusion about background removal."),
P("The clean generic-pose run V7 reaches 36.93% mAP, well below V6b. It detected a figure in 169 of 200 training images; the remaining 31 all-zero feature vectors help explain the gap. The earlier vase-specific detector produced detections on 197 of 198 retrieval-training images and recorded 45.55% retrieval mAP, but its detector had already seen exact retrieval-test images. Its 0.92 box mAP@50, 0.75 pose mAP@50, and retrieval score are therefore diagnostic only. A leakage-free rerun is needed to measure whether domain-specific pose training improves retrieval."),
P("Earlier in the project, retrieval requested only ten neighbours and then removed the self-match, leaving at most nine usable results for Accuracy@10. The final evaluator searches the complete test gallery and removes the query before scoring. All values retained in Table 1 use this corrected evaluator.")]

# ---- 9 implementation --------------------------------------------------------
story += [heading("9", "Implementation and Reproducibility"),
P("The experiments were run on the TinyGPU cluster of the Erlangen National High Performance Computing Center (NHR@FAU), with jobs submitted through SLURM over SSH; early development and debugging used a local GPU. In the recorded cluster runs, versions V1 to V7 ran on NVIDIA A100 nodes, and V7b ran on an NVIDIA GeForce RTX 3080 node because the A100 queue was heavily contended. The diagnostic results in Table 2 come from these cluster runs. After BMP support was added, the six clean versions in Table 1 were rerun with the complete 253-image dataset on a local machine with an NVIDIA GeForce RTX 3050 GPU, using PyTorch 2.5.1 [14] with CUDA 12.1, Ultralytics 8.4.33 [8,9], FAISS 1.13.2 [10], and PyTorch Metric Learning 2.9.0 [11]. Versioned scripts cover feature extraction, projection training where applicable, retrieval, and evaluation."),
P("The compute nodes have no internet access, so DINOv2, YOLO, and SAM weights must be cached before submission. The corrected V4 and V5 job scripts now pass their saved projection paths to retrieval. The corrected pose-training script compares hashes and case-insensitive filenames against the retrieval-test images, removes matches from the annotated pose pool, and aborts if any overlap remains before creating its deterministic train/validation lists."),
P("The fixed seed, saved folder-level split, shared evaluator, dependency list, and per-version job scripts support repetition. Final archival should include the code, README, model weights, dataset or an authorised dataset reference, and the exact train/test split files. Because the reported scores are from single runs, repeating the main comparisons over several folder-level splits would provide a better estimate of variability.")]

# ---- 10 conclusion -----------------------------------------------------------
story += [heading("10", "Conclusion and Future Work"),
P("The project produced a working retrieval pipeline for archaeological vase images. Among the BMP-complete clean held-out experiments, DINOv2 with a triplet-trained projection is strongest at 78.26% mAP, 77.36% Accuracy@1, and 100% Accuracy@10. The generic pose-only representation reaches 36.93% mAP. These scores measure retrieval of images sharing the same supplied folder label. The current evidence is insufficient to conclude whether trained SAM crops or vase-specific pose detection improve retrieval, because the former did not load their projections and the latter used overlapping detector-training and retrieval-test images."),
P("The immediate next experiments are corrected V4/V5 retrieval and leakage-free V7b training. Beyond those checks, useful directions include repeated folder-level splits, more images per class, a larger or lightly fine-tuned DINOv2 backbone, fusion of appearance and pose embeddings, evaluation against an external catalogue, and later investigation of the urn data once reliable motif-level relevance labels are available.")]

# ---- 11 references -----------------------------------------------------------
story += [PageBreak(), heading("11", "References"),
P('[1] K. He, X. Zhang, S. Ren, and J. Sun. "Deep Residual Learning for Image Recognition." <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, pp. 770-778, 2016. DOI: <link href="https://doi.org/10.1109/CVPR.2016.90">10.1109/CVPR.2016.90</link>.', reference),
P('[2] M. Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." <i>Transactions on Machine Learning Research</i>, 2024. <link href="https://openreview.net/forum?id=a68SUt6zFt">openreview.net/forum?id=a68SUt6zFt</link>.', reference),
P('[3] F. Schroff, D. Kalenichenko, and J. Philbin. "FaceNet: A Unified Embedding for Face Recognition and Clustering." <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, pp. 815-823, 2015. DOI: <link href="https://doi.org/10.1109/CVPR.2015.7298682">10.1109/CVPR.2015.7298682</link>.', reference),
P('[4] X. Wang, X. Han, W. Huang, D. Dong, and M. R. Scott. "Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning." <i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, pp. 5022-5030, 2019. DOI: <link href="https://doi.org/10.1109/CVPR.2019.00516">10.1109/CVPR.2019.00516</link>.', reference),
P('[5] S. Kim, D. Kim, M. Cho, and S. Kwak. "Proxy Anchor Loss for Deep Metric Learning." <i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, pp. 3235-3244, 2020. DOI: <link href="https://doi.org/10.1109/CVPR42600.2020.00330">10.1109/CVPR42600.2020.00330</link>.', reference),
P('[6] J. Deng, J. Guo, N. Xue, and S. Zafeiriou. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." <i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, pp. 4690-4699, 2019. DOI: <link href="https://doi.org/10.1109/CVPR.2019.00482">10.1109/CVPR.2019.00482</link>.', reference),
P('[7] A. Kirillov et al. "Segment Anything." <i>Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)</i>, pp. 4015-4026, 2023. DOI: <link href="https://doi.org/10.1109/ICCV51070.2023.00371">10.1109/ICCV51070.2023.00371</link>.', reference),
P('[8] G. Jocher, A. Chaurasia, and J. Qiu. <i>Ultralytics YOLOv8</i>, version 8.0.0 [Computer software]. Ultralytics, 2023. <link href="https://github.com/ultralytics/ultralytics">github.com/ultralytics/ultralytics</link> (accessed 4 September 2026).', reference),
P('[9] G. Jocher and J. Qiu. <i>Ultralytics YOLO11</i>, version 11.0.0 [Computer software]. Ultralytics, 2024. <link href="https://docs.ultralytics.com/models/yolo11/">docs.ultralytics.com/models/yolo11</link> (accessed 4 September 2026). Experiments used the ultralytics Python package, version 8.4.33.', reference),
P('[10] J. Johnson, M. Douze, and H. Jegou. "Billion-Scale Similarity Search with GPUs." <i>IEEE Transactions on Big Data</i>, vol. 7, no. 3, pp. 535-547, 2021. DOI: <link href="https://doi.org/10.1109/TBDATA.2019.2921572">10.1109/TBDATA.2019.2921572</link>.', reference),
P('[11] K. Musgrave, S. Belongie, and S.-N. Lim. "PyTorch Metric Learning." arXiv:2008.09164, 2020. DOI: <link href="https://doi.org/10.48550/arXiv.2008.09164">10.48550/arXiv.2008.09164</link>. Software version 2.9.0 used in the clean reruns.', reference),
P('[12] T.-Y. Lin et al. "Microsoft COCO: Common Objects in Context." <i>Computer Vision - ECCV 2014</i>, LNCS 8693, pp. 740-755, 2014. DOI: <link href="https://doi.org/10.1007/978-3-319-10602-1_48">10.1007/978-3-319-10602-1_48</link>.', reference),
P('[13] Ultralytics. "COCO-Pose Dataset Specification." <i>Ultralytics YOLO Documentation</i>, 17-keypoint, 3-value layout. <link href="https://docs.ultralytics.com/datasets/pose/coco/">docs.ultralytics.com/datasets/pose/coco</link> (accessed 4 September 2026).', reference),
P('[14] A. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." <i>Advances in Neural Information Processing Systems 32</i>, pp. 8024-8035, 2019. <link href="https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library">papers.neurips.cc/paper/9015</link>.', reference),
P('[15] D. P. Kingma and J. Ba. "Adam: A Method for Stochastic Optimization." <i>3rd International Conference on Learning Representations (ICLR)</i>, 2015. arXiv:1412.6980. DOI: <link href="https://doi.org/10.48550/arXiv.1412.6980">10.48550/arXiv.1412.6980</link>.', reference),
P('[16] O. Russakovsky et al. "ImageNet Large Scale Visual Recognition Challenge." <i>International Journal of Computer Vision</i>, vol. 115, pp. 211-252, 2015. DOI: <link href="https://doi.org/10.1007/s11263-015-0816-y">10.1007/s11263-015-0816-y</link>.', reference)]


def footer(canvas, doc):
    if doc.page == 1:
        return  # title page is unnumbered
    canvas.saveState()
    canvas.setFont("CMU", 10)
    canvas.setFillColor(INK)
    canvas.drawCentredString(A4[0] / 2.0, 1.2 * cm, str(doc.page - 1))
    canvas.restoreState()


out = os.path.join(HERE, "Vase_Urn_Retrieval_Report.pdf")
doc = SimpleDocTemplate(out, pagesize=A4, topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                        title="Metric Learning for Archaeological Vase Retrieval",
                        author="Midhun Somanunnithan")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF:", out)
