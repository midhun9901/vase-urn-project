import json
import os
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


BASE = Path(__file__).resolve().parent.parent
RUN_FULL = BASE / "runs" / "dinov2_full"
RUN_TRIPLET = BASE / "runs" / "dinov2_triplet"

THUMB_W = 170
THUMB_H = 135
TOP_K = 9

BG = "#f6f3ee"
SURFACE = "#ffffff"
TEXT = "#202124"
MUTED = "#5f6368"
BORDER = "#d8d2c8"
BLUE = "#1f5fbf"
GREEN = "#20744a"
RED = "#a13a2f"


def read_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_metrics(run_dir):
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_path(path_text):
    path = Path(path_text)
    if path.exists():
        return path

    text = path_text.replace("\\", "/")
    markers = [
        ("pairs_pt1 (1)/Bildpaare Teil 1/", BASE / "data" / "pairs_pt1" / "Bildpaare Teil 1"),
        ("pairs_pt2 (1)/Bildpaare-Triplets Teil 2/", BASE / "data" / "pairs_pt2" / "Bildpaare-Triplets Teil 2"),
        ("data/pairs_pt1/Bildpaare Teil 1/", BASE / "data" / "pairs_pt1" / "Bildpaare Teil 1"),
        ("data/pairs_pt2/Bildpaare-Triplets Teil 2/", BASE / "data" / "pairs_pt2" / "Bildpaare-Triplets Teil 2"),
    ]
    for marker, local_root in markers:
        if marker in text:
            rel = text.split(marker, 1)[1]
            candidate = local_root / Path(rel)
            if candidate.exists():
                return candidate
    return path


def load_data():
    required = [
        RUN_FULL / "retrieval_indices.npy",
        RUN_FULL / "retrieval_labels.npy",
        RUN_FULL / "test_image_paths.txt",
        RUN_TRIPLET / "retrieval_indices.npy",
        RUN_TRIPLET / "retrieval_labels.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing V5 run files:\n" + "\n".join(missing))

    paths = [resolve_image_path(p) for p in read_lines(RUN_FULL / "test_image_paths.txt")]
    full_indices = np.load(RUN_FULL / "retrieval_indices.npy")
    triplet_indices = np.load(RUN_TRIPLET / "retrieval_indices.npy")
    labels = np.load(RUN_FULL / "retrieval_labels.npy")
    full_metrics = load_metrics(RUN_FULL)
    triplet_metrics = load_metrics(RUN_TRIPLET)

    return {
        "paths": paths,
        "labels": labels,
        "full_indices": full_indices,
        "triplet_indices": triplet_indices,
        "full_metrics": full_metrics,
        "triplet_metrics": triplet_metrics,
    }


def make_thumb(path, w=THUMB_W, h=THUMB_H):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((w, h), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), "white")
        x = (w - img.width) // 2
        y = (h - img.height) // 2
        canvas.paste(img, (x, y))
    except Exception:
        canvas = Image.new("RGB", (w, h), (230, 230, 230))
    return ImageTk.PhotoImage(canvas)


def metric_text(name, metrics):
    if not metrics:
        return f"{name}: metrics missing"
    return (
        f"{name}: mAP {metrics['mAP'] * 100:.2f}% | "
        f"Acc@1 {metrics['Accuracy@1']:.2f}% | "
        f"Acc@10 {metrics['Accuracy@10']:.2f}%"
    )


class V5Demo:
    def __init__(self, root):
        self.root = root
        self.root.title("V5 DINOv2 Retrieval Demo")
        self.root.geometry("1420x850")
        self.root.configure(bg=BG)
        self.data = load_data()
        self.query_idx = 0
        self.photos = {}
        self.build_ui()
        self.render()

    def build_ui(self):
        top = tk.Frame(self.root, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        top.pack(fill="x")

        title = tk.Label(top, text="V5 DINOv2 Retrieval Demo", bg=SURFACE, fg=TEXT, font=("Segoe UI", 16, "bold"))
        title.pack(side="left", padx=16, pady=12)

        metric = tk.Label(
            top,
            text=metric_text("DINOv2 + Triplet", self.data["triplet_metrics"]),
            bg=SURFACE,
            fg=GREEN,
            font=("Segoe UI", 11, "bold"),
        )
        metric.pack(side="right", padx=16)

        metric2 = tk.Label(
            top,
            text=metric_text("DINOv2 full", self.data["full_metrics"]),
            bg=SURFACE,
            fg=BLUE,
            font=("Segoe UI", 10),
        )
        metric2.pack(side="right", padx=16)

        nav = tk.Frame(self.root, bg=BG)
        nav.pack(fill="x", padx=16, pady=10)

        tk.Label(nav, text="Query:", bg=BG, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        self.query_var = tk.StringVar()
        self.query_box = ttk.Combobox(nav, textvariable=self.query_var, state="readonly", width=70)
        self.query_box["values"] = [self.short_name(p, i) for i, p in enumerate(self.data["paths"])]
        self.query_box.current(0)
        self.query_box.pack(side="left", padx=8)
        self.query_box.bind("<<ComboboxSelected>>", self.on_select)

        tk.Button(nav, text="Prev", command=self.prev_query, bg=SURFACE, fg=TEXT, relief="solid", bd=1).pack(side="left", padx=4)
        tk.Button(nav, text="Next", command=self.next_query, bg=SURFACE, fg=TEXT, relief="solid", bd=1).pack(side="left", padx=4)
        tk.Button(nav, text="Random", command=self.random_query, bg=SURFACE, fg=TEXT, relief="solid", bd=1).pack(side="left", padx=4)

        self.info = tk.Label(nav, text="", bg=BG, fg=MUTED, font=("Segoe UI", 10))
        self.info.pack(side="left", padx=16)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        left = tk.Frame(body, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, width=230)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(left, text="Query Image", bg=SURFACE, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(pady=(12, 8))
        self.query_img = tk.Label(left, bg=SURFACE)
        self.query_img.pack(pady=8)
        self.query_label = tk.Label(left, text="", bg=SURFACE, fg=MUTED, font=("Segoe UI", 9), wraplength=200, justify="center")
        self.query_label.pack(padx=12, pady=8)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.full_frame = self.make_results_frame(right, "V5a - DINOv2 Full Image", BLUE)
        self.triplet_frame = self.make_results_frame(right, "V5b - DINOv2 + Triplet", GREEN)

    def make_results_frame(self, parent, title, color):
        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True, pady=(0, 12))
        tk.Label(frame, text=title, bg=SURFACE, fg=color, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        grid = tk.Frame(frame, bg=SURFACE)
        grid.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        frame.grid_holder = grid
        return frame

    def short_name(self, path, index):
        parent = path.parent.name
        return f"{index + 1:02d}. {parent} / {path.name}"

    def on_select(self, _event=None):
        self.query_idx = self.query_box.current()
        self.render()

    def prev_query(self):
        self.query_idx = (self.query_idx - 1) % len(self.data["paths"])
        self.query_box.current(self.query_idx)
        self.render()

    def next_query(self):
        self.query_idx = (self.query_idx + 1) % len(self.data["paths"])
        self.query_box.current(self.query_idx)
        self.render()

    def random_query(self):
        self.query_idx = random.randrange(len(self.data["paths"]))
        self.query_box.current(self.query_idx)
        self.render()

    def non_self_results(self, indices):
        return [idx for idx in indices[self.query_idx] if idx != self.query_idx][:TOP_K]

    def clear_grid(self, frame):
        for child in frame.grid_holder.winfo_children():
            child.destroy()

    def render_result_grid(self, frame, indices, key_prefix):
        self.clear_grid(frame)
        query_label = self.data["labels"][self.query_idx]
        for col, idx in enumerate(indices):
            path = self.data["paths"][idx]
            correct = self.data["labels"][idx] == query_label
            card = tk.Frame(frame.grid_holder, bg=SURFACE, highlightbackground=(GREEN if correct else RED), highlightthickness=2)
            card.grid(row=0, column=col, padx=5, pady=4, sticky="n")

            photo = make_thumb(path)
            self.photos[f"{key_prefix}_{idx}_{col}"] = photo
            tk.Label(card, image=photo, bg=SURFACE).pack()
            rank_text = f"#{col + 1} " + ("MATCH" if correct else "DIFF")
            tk.Label(card, text=rank_text, bg=SURFACE, fg=(GREEN if correct else RED), font=("Segoe UI", 9, "bold")).pack(pady=(4, 0))
            tk.Label(card, text=path.parent.name, bg=SURFACE, fg=MUTED, font=("Segoe UI", 8), wraplength=THUMB_W).pack()

    def render(self):
        path = self.data["paths"][self.query_idx]
        query_photo = make_thumb(path, 200, 260)
        self.photos["query"] = query_photo
        self.query_img.configure(image=query_photo)
        self.query_label.configure(text=f"{path.parent.name}\n{path.name}")
        self.info.configure(text=f"Image {self.query_idx + 1} / {len(self.data['paths'])} | label {self.data['labels'][self.query_idx]}")

        self.render_result_grid(self.full_frame, self.non_self_results(self.data["full_indices"]), "full")
        self.render_result_grid(self.triplet_frame, self.non_self_results(self.data["triplet_indices"]), "triplet")


def main():
    root = tk.Tk()
    V5Demo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
