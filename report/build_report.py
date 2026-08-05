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

# ---- data (verified results) -------------------------------------------------
VERS = ["V1", "V2", "V3", "V4", "V5", "V6a", "V6b", "V7", "V7b"]
MAP = [15.51, 54.79, 57.43, 11.55, 11.55, 32.53, 83.39, 39.76, 45.55]
A1 = [15.38, 44.23, 44.23, 9.62, 9.62, 26.92, 80.77, 38.46, 38.46]
A10 = [28.85, 96.15, 96.15, 30.77, 30.77, 82.69, 100.0, 57.69, 67.31]


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


def fig_pose():
    import numpy as np
    labels = ["mAP", "Acc@1", "Acc@10"]
    gen = [39.76, 38.46, 57.69]; tuned = [45.55, 38.46, 67.31]
    x = np.arange(len(labels)); bw = 0.36
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.bar(x - bw / 2, gen, bw, label="V7  generic YOLOv8", color="#c4c4c4")
    ax.bar(x + bw / 2, tuned, bw, label="V7b  vase YOLOv11", color="#36618e")
    for i, v in enumerate(gen):
        ax.text(i - bw / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8, color="#555555")
    for i, v in enumerate(tuned):
        ax.text(i + bw / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8, color="#1f2a44")
    ax.set_ylabel("score (%)"); ax.set_ylim(0, 85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6); ax.set_axisbelow(True)
    plt.tight_layout(pad=0.3)
    p = os.path.join(FIG, "fig3_pose.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


f1, f2, f3 = fig_pipeline(), fig_results(), fig_pose()

# ---- paragraph styles (Computer Modern) --------------------------------------
body = ParagraphStyle("body", fontName="CMU", fontSize=10.5, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=7, textColor=INK)
h1 = ParagraphStyle("h1", fontName="CMU-Bold", fontSize=13.5, leading=16,
                    spaceBefore=15, spaceAfter=6, textColor=INK)
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
    P("Metric Learning for Vase and Urn Retrieval", ctitle),
    Spacer(1, 0.25 * cm),
    P("Master Project Report", csub),
    Spacer(1, 0.4 * cm),
    HRFlowable(width="55%", thickness=0.8, color=INK, spaceBefore=2, spaceAfter=16),
    P("Midhun Somanunnithan", cmeta),
    Spacer(1, 0.45 * cm),
    P("Supervised by Mathias Zinnen", csmall),
    Spacer(1, 5.5 * cm),
    P("Friedrich-Alexander-Universit&auml;t Erlangen-N&uuml;rnberg", csmall),
    P("June 2026", csmall),
    PageBreak(),
]

# ---- abstract ----------------------------------------------------------------
story += [
    Spacer(1, 0.2 * cm),
    P("Abstract", absth),
    P("This report describes a content-based image retrieval system for archaeological vases and urns, built around metric learning. Given a query image, the system ranks a database by visual similarity in a learned embedding space. I compare two feature backbones, ResNet50 and DINOv2, together with several metric-learning losses, and I test two further ideas: removing the background with the Segment Anything Model, and describing each image by the pose of its painted figures. The strongest method, DINOv2 features adapted with a triplet-loss projection, reaches 83.39% mean Average Precision and retrieves a correct match within the top ten for every test query. Background removal and pose both underperform, and the report examines why. Following an initial preparation phase, the work was carried out as a series of controlled, versioned experiments that share one pipeline and one evaluator and run reproducibly on the university HPC cluster.", abst),
    Spacer(1, 0.3 * cm),
]

# ---- 1 introduction ----------------------------------------------------------
story += [heading("1", "Introduction"),
P("This project deals with content-based image retrieval for archaeological vases and urns. The practical question is simple: given a photograph or drawing of a vase, find other images that show the same object. This is useful for art-historical research, where the same vessel often appears in several catalogues, museum records, and scholarly illustrations, but under different lighting, framing, and reproduction quality, so a keyword search is rarely enough to connect them."),
P("The task is a retrieval problem rather than a classification problem. The set of vases is open, there are only a few images per object, and at test time the goal is to rank a database by visual similarity to a query. This makes it a good fit for metric learning, where a model is trained to produce an embedding space in which images of the same object lie close together and images of different objects lie far apart. Once such an embedding exists, retrieval is a nearest-neighbour search."),
P("The work was carried out as a sequence of versioned experiments, from a plain baseline up to a strong final method, with two deliberate detours into ideas that did not pay off. The detours are kept in the report because they shaped the final design and because the negative results are informative in their own right.")]

# ---- 2 objectives ------------------------------------------------------------
story += [heading("2", "Objectives and Scope"),
P("The objectives were to build a reproducible retrieval pipeline that turns vase images into embeddings and evaluates ranking quality with standard metrics; to compare feature backbones and loss functions in order to find a representation that works well on a small dataset with few images per object; to test two further ideas, isolating the vase from its background, which I tried after the early results, and using the pose of the painted figures as a similarity signal, which the project material pointed to; and to run everything on the university HPC cluster in a way that can be repeated from a single job script."),
P("The scope is whole-image retrieval of vases and urns. The report does not attempt instance segmentation, text-based search, or cross-modal matching. The evaluation is intrinsic, using the held-out portion of the same dataset rather than an external benchmark.")]

# ---- 3 preparation and background -------------------------------------------
story += [heading("3", "Preparation and Background"),
P("Before any experiments, the first part of the project was spent building the background needed to carry the work out independently. I worked through the fundamentals of deep learning and neural-network training, and then the practical side in PyTorch: tensors, the training loop, datasets and data loaders, the difference between training and evaluation mode, and writing device-agnostic code that runs the same on a laptop and on a GPU node. In parallel I studied the metric-learning material recommended for the project and set myself up on the university HPC cluster, including SSH access, the module and conda environment, and submitting jobs through SLURM."),
P("This preparation shaped how the rest of the work was done. The project was run as a sequence of versioned experiments rather than as a single model. Each version changes one decision relative to the previous line of work, keeps the data split and the evaluator fixed, and is recorded together with its result, so that every comparison is controlled and any change in score can be attributed to the change that caused it. The remainder of this section summarises the concepts the versions build on."),
P("<b>Metric learning losses.</b> Triplet margin loss pulls an anchor towards a positive of the same class and pushes it away from a negative of a different class, subject to a margin. It is paired with a miner that selects informative triplets, here the multi-similarity miner, so that training does not waste effort on already-easy examples. I also tried ProxyAnchor loss, which replaces explicit triplets with learnable class proxies and is generally reported to behave well when each class has only a handful of samples, and ArcFace, an angular-margin classification loss often used for face recognition."),
P("<b>Embedding and retrieval.</b> After training, each image is mapped to a vector, the vectors are L2-normalised, and retrieval is performed with FAISS using an exact flat L2 index. Because the test set is small, I search the full ranking rather than a fixed top-k, which removes an artefact discussed in Section 8."),
P("<b>Evaluation metrics.</b> I report mean Average Precision (mAP), Accuracy@1, and Accuracy@10. mAP rewards placing all relevant items high in the ranking and is the primary metric. Accuracy@1 is the fraction of queries whose nearest neighbour is correct, and Accuracy@10 is the fraction with at least one correct item in the top ten. The query itself is removed from its own ranking before scoring."),
P("<b>Backbones.</b> Two feature extractors are central. ResNet50 is a standard convolutional network pretrained on ImageNet. DINOv2 (ViT-S/14) is a self-supervised vision transformer whose features are known to transfer well without fine-tuning. For the pose work I used the YOLO pose models from Ultralytics, which predict the 17 keypoints of a detected figure.")]

# ---- 4 dataset ---------------------------------------------------------------
story += [heading("4", "Dataset"),
P("The data consists of two parts provided for the project. The first part contains 71 folders of book-illustration vases, and the second contains 30 folders of mixed photographs and illustrations, for 101 folders in total. Each folder corresponds to one object identity and holds the small number of images that depict that object. After loading, the collection amounts to 250 images spread across the 101 identities, which is a small dataset by deep-learning standards and was a constant constraint on method choice."),
P("The split is made at the level of folders, not images, so that all images of a given object fall entirely into either the training or the test set. This avoids the obvious leak of having near-duplicate views of the same vase on both sides of the split. I use an 80/20 ratio with a fixed random seed, which produces 198 training images and 52 test images. The split is computed once, written to disk, and reused by every later stage, so all versions are evaluated on exactly the same partition."),
P("For the pose experiments a second, separately annotated dataset was used. It contains 238 training images with keypoint annotations in the COCO 17-keypoint layout, under a class named Figure-Pose, together with image-field (motif) annotations and a small set of test images. This dataset was only used to train the pose model; the retrieval evaluation always uses the same 101-folder vase collection.")]

# (Background theory is folded into Section 3, Preparation and Background.)

# ---- 5 method + fig1 ---------------------------------------------------------
story += [heading("5", "System and Method"),
P("Every version follows the same four-stage pipeline, shown in Figure 1, and the stages share one common module so that results stay comparable."),
img(f1, 15.5, "Figure 1. The shared four-stage retrieval pipeline. Only the backbone and the loss change between versions; the split, the index, and the evaluator are identical."),
P("<b>Feature extraction.</b> Each image is passed through the chosen backbone, and the resulting embeddings, labels, and file paths are saved as arrays. <b>Projection training.</b> A small multilayer perceptron is trained on the embeddings with a metric-learning loss. The network is input → hidden → output with a ReLU in between, and the output is L2-normalised before the loss. Training uses the Adam optimiser at a learning rate of 0.001, a batch size of 32, and an M-per-class sampler with two samples per class per batch so that each batch contains usable positive pairs. For the DINOv2 version the layout is 384 → 512 → 128; for the pose version it is 51 → 128 → 64."),
P("<b>Retrieval.</b> The test embeddings are optionally passed through the trained projection, L2-normalised, indexed with FAISS, and searched against themselves to produce a full ranking per query. <b>Evaluation.</b> The shared metric routine computes mAP, Accuracy@1, and Accuracy@10 from the ranking and the labels."),
P("Determinism is enforced throughout. A single seed fixes the data split, the Python, NumPy and PyTorch random states, and the sampler order, so a rerun reproduces the same numbers. Keeping the split and the evaluator in one shared module is what makes the cross-version comparison fair: when a number moves, it is because the method changed, not because the test set or the scoring changed.")]

# ---- 6 experiments -----------------------------------------------------------
story += [heading("6", "Experimental Methodology"),
P("The experiments are numbered as versions, following the controlled approach set out in Section 3. Each one changes a single design decision relative to the line of work it follows."),
P("<b>V1, Raw ResNet50 baseline.</b> Features are taken directly from an ImageNet-pretrained ResNet50 with no metric learning, indexed, and evaluated. This sets the floor. <b>V2, ResNet50 with triplet loss.</b> The same features are passed through the trained MLP with triplet loss and the multi-similarity miner. <b>V3, ResNet50 with ProxyAnchor loss.</b> Identical to V2 except that the loss is replaced by ProxyAnchor, chosen because the dataset is small."),
P("<b>V4, SAM crop with ArcFace.</b> Here I tested removing the background: the Segment Anything Model isolates and crops the vase before feature extraction, and the projection is trained with ArcFace loss. <b>V5, SAM crop with triplet loss.</b> The same cropping pipeline as V4 but with triplet loss, run specifically to check whether the loss rather than the cropping was responsible for V4's behaviour."),
P("<b>V6a, DINOv2 without fine-tuning.</b> The backbone is switched to DINOv2 ViT-S/14 and the features are ranked directly, with no projection training. <b>V6b, DINOv2 with triplet loss.</b> The DINOv2 features are passed through the MLP trained with triplet loss, combining a strong representation with task-specific adaptation."),
P("<b>V7, Generic pose keypoints.</b> A different idea: describe each image by the pose of its figure rather than its appearance. An off-the-shelf YOLOv8 pose model produces 17 keypoints per image, which are flattened into a fixed-length vector and fed into the same projection and retrieval stages. <b>V7b, Vase-specific pose keypoints.</b> After diagnosing why V7 was weak, I fine-tuned a YOLOv11 pose model on the annotated vase keypoint dataset and re-ran the pose pipeline with it.")]

# ---- 7 results + table + fig2 ------------------------------------------------
tbl_head = [Paragraph(x, hcell) for x in ["Version", "Method", "Hardware", "mAP", "Acc@1", "Acc@10"]]
rows = [
    ("V1", "Raw ResNet50, no MLP", "A100", "15.51%", "15.38%", "28.85%"),
    ("V2", "ResNet50 + MLP + Triplet", "A100", "54.79%", "44.23%", "96.15%"),
    ("V3", "ResNet50 + MLP + ProxyAnchor", "A100", "57.43%", "44.23%", "96.15%"),
    ("V4", "SAM crop + ArcFace", "A100", "11.55%", "9.62%", "30.77%"),
    ("V5", "SAM crop + Triplet", "A100", "11.55%", "9.62%", "30.77%"),
    ("V6a", "DINOv2 features, no training", "A100", "32.53%", "26.92%", "82.69%"),
    ("V6b", "DINOv2 + MLP + Triplet", "A100", "83.39%", "80.77%", "100.00%"),
    ("V7", "Generic YOLOv8 pose + Triplet", "A100", "39.76%", "38.46%", "57.69%"),
    ("V7b", "Vase YOLOv11 pose + Triplet", "RTX 3080", "45.55%", "38.46%", "67.31%"),
]
data = [tbl_head]
for r in rows:
    isbest = r[0] == "V6b"
    sc, scc = (cellb, cellbc) if isbest else (cell, cellc)
    data.append([Paragraph(r[0], scc), Paragraph(r[1], sc), Paragraph(r[2], scc),
                 Paragraph(r[3], scc), Paragraph(r[4], scc), Paragraph(r[5], scc)])
tbl = Table(data, colWidths=[1.6 * cm, 6.4 * cm, 2.0 * cm, 2.1 * cm, 2.0 * cm, 2.2 * cm])
tbl.setStyle(TableStyle([
    ("LINEABOVE", (0, 0), (-1, 0), 1.1, INK),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
    ("LINEBELOW", (0, -1), (-1, -1), 1.1, INK),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
]))

story += [heading("7", "Results"),
P("All versions are evaluated on the same held-out folders with the same full-ranking evaluator. Table 1 collects the numbers and Figure 2 shows them side by side."),
KeepTogether([tbl, Spacer(1, 0.15 * cm),
              cap("Table 1. Retrieval results across all versions. V6b is the best method.")]),
img(f2, 15.5, "Figure 2. Retrieval scores by version. Appearance-based DINOv2 with triplet loss (V6b) dominates; background cropping (V4, V5) falls below the raw baseline."),
P("The best method is V6b, DINOv2 features adapted with an MLP and triplet loss, at 83.39% mAP, 80.77% Accuracy@1, and 100% Accuracy@10. V7b ran on an RTX 3080 rather than an A100 because of cluster queue contention; the GPU type does not affect the metrics.")]

# ---- 8 discussion + fig3 -----------------------------------------------------
story += [heading("8", "Discussion"),
P("The first pattern in Table 1 is that metric learning is necessary. The jump from V1 (15.51%) to V2 (54.79%) shows that raw ImageNet features are not suited to this task and that even a small trained projection helps a great deal. The jump from V6a (32.53%) to V6b (83.39%) shows the same effect for DINOv2: the untrained features are already better than raw ResNet50 but only become strong once they are adapted with triplet loss. At the same time, ResNet50 with triplet loss reaches the mid-fifties while DINOv2 with the same loss reaches the low-eighties, so the representation matters more than the loss."),
P("The loss function, by contrast, is a second-order effect. V3 (ProxyAnchor, 57.43%) is only slightly above V2 (Triplet, 54.79%), and both share the same Accuracy@1 and Accuracy@10. On this dataset the choice between these two losses is minor compared with the choice of backbone."),
P("Background removal hurts. SAM cropping collapses performance to 11.55% mAP in both V4 and V5, below even the raw baseline. Because V4 and V5 use different losses and give identical numbers, the loss is not the cause; the cropping is. The global shape and surrounding context of the vessel carry much of the discriminative signal, and cutting the object out of its frame throws that away. After this I stopped preprocessing the input and focused on the representation, which led to DINOv2."),
P("The pose experiments tell a similar story from a different direction. The first attempt (V7, 39.76%) used a generic human-pose model. Inspecting its outputs showed the problem: the model is trained on real human bodies, and whenever it finds no figure in a painted scene it returns an all-zero vector. The fix was to train a pose model on the vase figures themselves. The fine-tuned YOLOv11 model trained well, with box mAP@50 around 0.92 and pose mAP@50 around 0.75 on validation, and it produced a real detection on 197 of 198 training images, about 99%. With proper features the retrieval rose to 45.55% mAP (Figure 3). Even so, a correct model that detects a figure on almost every image still reaches about half of what DINOv2 achieves, because keypoints describe the posture of the painted figure rather than the vessel as a whole."),
img(f3, 11.0, "Figure 3. Generic versus vase-specific pose. Fine-tuning improves mAP and Accuracy@10 while Accuracy@1 stays the same, and pose retrieval still trails the appearance-based method by a wide margin."),
P("A last point concerns the evaluation itself. Earlier in the project the retrieval used a fixed top-k of ten, which meant that Accuracy@10 effectively had only nine non-self results to work with. Switching to a full ranking and removing the query from its own list fixed this, and all reported numbers use the corrected evaluator.")]

# ---- 9 implementation --------------------------------------------------------
story += [heading("9", "Implementation and Reproducibility"),
P("The work runs on the FAU NHR TinyGPU cluster through SLURM. Each version has a job script that loads the conda environment, sets the cache location, and runs the four pipeline stages in order, so a complete experiment is a single submission. The environment uses PyTorch 2.5.1 with CUDA 12.1, Ultralytics 8.4.41, FAISS 1.7.2, and PyTorch Metric Learning 2.9.0."),
P("Two practical issues came up. First, the compute nodes have no internet access, so the DINOv2 and YOLO base weights have to be downloaded on the login node and cached before a job runs. Second, the vase keypoint dataset shipped with an image list whose path prefix did not match the actual folder layout, so I generate my own deterministic train/validation split from the real image files rather than relying on that list."),
P("Reproducibility rests on the fixed seed, the cached data split, and the shared evaluator. Large artefacts such as datasets, model weights, and generated embeddings are kept out of version control, while all code, job scripts, and the results table are tracked, so the repository is small and the experiments can be regenerated from scratch.")]

# ---- 10 conclusion -----------------------------------------------------------
story += [heading("10", "Conclusion and Future Work"),
P("The project produced a working, reproducible retrieval system for vases and urns and a clear comparison of design choices. The strongest method, the DINOv2 backbone with a triplet-trained MLP on top, reaches 83.39% mAP, with a correct match inside the top ten for every test query. Two intuitive ideas were tested and shown not to help: removing the background with SAM discards the global context the models rely on, and figure-pose keypoints, even from a model trained on the vases, reach 45.55% mAP against 83.39% for the appearance-based method."),
P("Several directions follow from this. The dataset is small, so collecting more images per object would likely raise the ceiling and make the comparison between losses more meaningful. A larger DINOv2 variant, or light fine-tuning of the backbone rather than only training a projection on top, is the natural next step for the appearance branch. Pose did not work as a standalone signal, but it was never tested as a complement to appearance: combining the DINOv2 embedding with the pose embedding, possibly after cropping to the detected image field, is still untried. Finally, validating the system against an external catalogue would give a stronger measure of real-world usefulness.")]


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
                        title="Metric Learning for Vase and Urn Retrieval",
                        author="Midhun Somanunnithan")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF:", out)
