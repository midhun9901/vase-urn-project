import json
import os
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


BASE = Path(__file__).resolve().parent.parent
THUMB_W = 154
THUMB_H = 120
TOP_K = 9

BG = "#f5f2ec"
SURFACE = "#ffffff"
TEXT = "#202124"
MUTED = "#5f6368"
BORDER = "#d7d0c4"
BLUE = "#1f5fbf"
GREEN = "#20744a"
RED = "#a13a2f"
PURPLE = "#7048a8"
ORANGE = "#ad5f13"


def read_lines(path):
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def load_metrics(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def image_files(folder):
    images = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.append(p)
    return sorted(images, key=lambda p: p.name.lower())


def build_paths_from_test_folders():
    folders_file = BASE / "test_folders.txt"
    if not folders_file.exists():
        return []
    paths = []
    for folder in read_lines(folders_file):
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_text = folder.replace("\\", "/")
            if "pairs_pt1" in folder_text:
                suffix = folder_text.split("pairs_pt1", 1)[1].lstrip("/\\")
                folder_path = BASE / "data" / "pairs_pt1" / suffix
            elif "pairs_pt2" in folder_text:
                suffix = folder_text.split("pairs_pt2", 1)[1].lstrip("/\\")
                folder_path = BASE / "data" / "pairs_pt2" / suffix
        if folder_path.exists():
            paths.extend(image_files(folder_path))
    return paths


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
        total_relevant = int(np.sum(labels == query_label) - 1)
        correct = 0
        precision_sum = 0.0
        for rank, lbl in enumerate(ranked_labels):
            if lbl == query_label:
                correct += 1
                precision_sum += correct / (rank + 1)
                if correct == total_relevant:
                    break
        if total_relevant:
            ap_list.append(precision_sum / total_relevant)
        else:
            ap_list.append(0.0)
    return {"mAP": float(np.mean(ap_list)), "Accuracy@1": acc1 / n * 100, "Accuracy@10": acc10 / n * 100}


def load_methods():
    methods = []

    old_paths = build_paths_from_test_folders()
    if (BASE / "retrieval_indices_baseline.npy").exists() and (BASE / "retrieval_labels_baseline.npy").exists():
        labels = np.load(BASE / "retrieval_labels_baseline.npy")
        indices = np.load(BASE / "retrieval_indices_baseline.npy")
        if len(old_paths) == len(labels):
            methods.append({
                "name": "V1 - ResNet50 Baseline",
                "color": ORANGE,
                "paths": old_paths,
                "labels": labels,
                "indices": indices,
                "metrics": compute_metrics(indices, labels),
                "note": "raw ResNet50 features",
            })

    if (BASE / "retrieval_indices.npy").exists() and (BASE / "retrieval_labels.npy").exists():
        labels = np.load(BASE / "retrieval_labels.npy")
        indices = np.load(BASE / "retrieval_indices.npy")
        if len(old_paths) == len(labels):
            methods.append({
                "name": "V2 - ResNet50 + Triplet",
                "color": PURPLE,
                "paths": old_paths,
                "labels": labels,
                "indices": indices,
                "metrics": compute_metrics(indices, labels),
                "note": "old local artifacts",
            })

    v5_specs = [
        ("V5a - DINOv2 Full Image", BLUE, BASE / "runs" / "dinov2_full", "DINOv2 features"),
        ("V5b - DINOv2 + Triplet", GREEN, BASE / "runs" / "dinov2_triplet", "new best result"),
    ]
    for name, color, run_dir, note in v5_specs:
        required = [
            run_dir / "retrieval_indices.npy",
            run_dir / "retrieval_labels.npy",
            run_dir / "test_image_paths.txt",
        ]
        if all(p.exists() for p in required):
            raw_paths = read_lines(run_dir / "test_image_paths.txt")
            paths = []
            for p in raw_paths:
                paths.append(resolve_image_path(p))
            labels = np.load(run_dir / "retrieval_labels.npy")
            indices = np.load(run_dir / "retrieval_indices.npy")
            methods.append({
                "name": name,
                "color": color,
                "paths": paths,
                "labels": labels,
                "indices": indices,
                "metrics": load_metrics(run_dir / "metrics.json") or compute_metrics(indices, labels),
                "note": note,
            })

    if not methods:
        raise FileNotFoundError("No retrieval result files found.")
    return methods


def metric_text(metrics):
    return f"mAP {metrics['mAP'] * 100:.2f}% | Acc@1 {metrics['Accuracy@1']:.2f}% | Acc@10 {metrics['Accuracy@10']:.2f}%"


def query_metrics(indices, labels, query_idx, shown_k=TOP_K):
    query_label = labels[query_idx]
    ranked = []
    for idx in indices[query_idx]:
        if idx != query_idx:
            ranked.append(idx)
    ranked_labels = labels[ranked]

    top1 = bool(len(ranked_labels) and ranked_labels[0] == query_label)
    top10 = bool(query_label in ranked_labels[:10])
    shown_matches = int(np.sum(ranked_labels[:shown_k] == query_label))

    total_relevant = int(np.sum(labels == query_label) - 1)
    correct = 0
    precision_sum = 0.0
    for rank, lbl in enumerate(ranked_labels):
        if lbl == query_label:
            correct += 1
            precision_sum += correct / (rank + 1)
            if correct == total_relevant:
                break
    if total_relevant:
        ap = precision_sum / total_relevant
    else:
        ap = 0.0

    return {
        "top1": top1,
        "top10": top10,
        "shown_matches": shown_matches,
        "total_relevant": total_relevant,
        "ap": ap,
    }


class AllVersionsDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Vase Retrieval - All Available Versions")
        self.root.geometry("1480x900")
        self.root.configure(bg=BG)
        self.methods = load_methods()
        self.primary = self.methods[-1]
        self.query_idx = 0
        self.photos = {}
        self.frames = []
        self.build_ui()
        self.render()

    def build_ui(self):
        top = tk.Frame(self.root, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        top.pack(fill="x")
        tk.Label(top, text="Vase Retrieval - All Available Versions", bg=SURFACE, fg=TEXT, font=("Segoe UI", 16, "bold")).pack(side="left", padx=16, pady=12)

        best = max(self.methods, key=lambda m: m["metrics"]["mAP"])
        tk.Label(top, text=f"Best: {best['name']} | {metric_text(best['metrics'])}", bg=SURFACE, fg=GREEN, font=("Segoe UI", 11, "bold")).pack(side="right", padx=16)

        nav = tk.Frame(self.root, bg=BG)
        nav.pack(fill="x", padx=16, pady=10)
        tk.Label(nav, text="Query:", bg=BG, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")

        self.query_var = tk.StringVar()
        self.query_box = ttk.Combobox(nav, textvariable=self.query_var, state="readonly", width=75)
        query_names = []
        for i, p in enumerate(self.primary["paths"]):
            query_names.append(self.short_name(p, i))
        self.query_box["values"] = query_names
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

        canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=BG)
        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for method in self.methods:
            self.frames.append((method, self.make_method_frame(self.scroll_frame, method)))

    def make_method_frame(self, parent, method):
        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", pady=(0, 12))
        header = tk.Frame(frame, bg=SURFACE)
        header.pack(fill="x")
        tk.Label(header, text=method["name"], bg=SURFACE, fg=method["color"], font=("Segoe UI", 12, "bold")).pack(side="left", padx=12, pady=(10, 2))
        tk.Label(header, text=metric_text(method["metrics"]), bg=SURFACE, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side="right", padx=12)
        tk.Label(frame, text=method["note"], bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=12)
        stats = tk.Label(frame, text="", bg=SURFACE, fg=TEXT, font=("Segoe UI", 9, "bold"))
        stats.pack(anchor="w", padx=12, pady=(2, 0))
        grid = tk.Frame(frame, bg=SURFACE)
        grid.pack(fill="x", padx=10, pady=10)
        frame.grid_holder = grid
        frame.stats_label = stats
        return frame

    def short_name(self, path, index):
        return f"{index + 1:02d}. {path.parent.name} / {path.name}"

    def on_select(self, _event=None):
        self.query_idx = self.query_box.current()
        self.render()

    def prev_query(self):
        self.query_idx = (self.query_idx - 1) % len(self.primary["paths"])
        self.query_box.current(self.query_idx)
        self.render()

    def next_query(self):
        self.query_idx = (self.query_idx + 1) % len(self.primary["paths"])
        self.query_box.current(self.query_idx)
        self.render()

    def random_query(self):
        self.query_idx = random.randrange(len(self.primary["paths"]))
        self.query_box.current(self.query_idx)
        self.render()

    def mapped_query_index(self, method):
        query_path = self.primary["paths"][self.query_idx]
        for i, path in enumerate(method["paths"]):
            if path == query_path:
                return i
        return self.query_idx if self.query_idx < len(method["paths"]) else 0

    def non_self_results(self, method, qidx):
        results = []
        for idx in method["indices"][qidx]:
            if idx != qidx:
                results.append(idx)
        return results[:TOP_K]

    def render(self):
        path = self.primary["paths"][self.query_idx]
        query_photo = make_thumb(path, 200, 260)
        self.photos["query"] = query_photo
        self.query_img.configure(image=query_photo)
        self.query_label.configure(text=f"{path.parent.name}\n{path.name}")
        self.info.configure(text=f"Image {self.query_idx + 1} / {len(self.primary['paths'])}")

        for method, frame in self.frames:
            for child in frame.grid_holder.winfo_children():
                child.destroy()
            qidx = self.mapped_query_index(method)
            query_label = method["labels"][qidx]
            qm = query_metrics(method["indices"], method["labels"], qidx)
            frame.stats_label.configure(
                text=(
                    f"Current query: AP {qm['ap'] * 100:.2f}% | "
                    f"Top-1 {'correct' if qm['top1'] else 'wrong'} | "
                    f"Top-10 {'hit' if qm['top10'] else 'miss'} | "
                    f"Shown matches {qm['shown_matches']}/{min(TOP_K, len(method['indices'][qidx]) - 1)} | "
                    f"Relevant total {qm['total_relevant']}"
                ),
                fg=(GREEN if qm["top1"] else RED),
            )
            for col, ridx in enumerate(self.non_self_results(method, qidx)):
                result_path = method["paths"][ridx]
                correct = method["labels"][ridx] == query_label
                card = tk.Frame(frame.grid_holder, bg=SURFACE, highlightbackground=(GREEN if correct else RED), highlightthickness=2)
                card.grid(row=0, column=col, padx=5, pady=4, sticky="n")
                photo = make_thumb(result_path)
                self.photos[f"{method['name']}_{qidx}_{ridx}_{col}"] = photo
                tk.Label(card, image=photo, bg=SURFACE).pack()
                tk.Label(card, text=f"#{col + 1} " + ("MATCH" if correct else "DIFF"), bg=SURFACE, fg=(GREEN if correct else RED), font=("Segoe UI", 8, "bold")).pack(pady=(4, 0))
                tk.Label(card, text=result_path.parent.name, bg=SURFACE, fg=MUTED, font=("Segoe UI", 8), wraplength=THUMB_W).pack()


def main():
    root = tk.Tk()
    AllVersionsDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
