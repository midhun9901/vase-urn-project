"""
demo.py — Visual Comparison Demo for Professor
================================================
Shows Baseline (ResNet-50 raw features) vs Metric Learning (Triplet Loss)
side by side for any query vase image.

Run:  python demo.py
Requires: retrieval_indices.npy, retrieval_indices_baseline.npy,
          test_labels.npy, test_folders.txt  (all already on disk)
"""

import os, sys
import numpy as np
import tkinter as tk
from tkinter import ttk, font as tkfont
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Colours (Catppuccin Mocha palette) ────────────────────────────────────────
BG       = "#1e1e2e"
SURFACE  = "#181825"
SURFACE2 = "#313244"
OVERLAY  = "#45475a"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
BLUE     = "#89b4fa"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
RED      = "#f38ba8"
MAUVE    = "#cba6f7"
PEACH    = "#fab387"
LAVENDER = "#b4befe"

THUMB_W, THUMB_H = 160, 120      # thumbnail size
TOP_K = 9                         # show top-9 results (skip self)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load all pre-computed npy files and folder lists."""
    # test folder → one representative image path per folder
    with open(os.path.join(BASE, "test_folders.txt")) as f:
        folders = [ln.strip() for ln in f if ln.strip()]

    labels = np.load(os.path.join(BASE, "test_labels.npy"))

    # Metric-learning indices
    ml_indices = None
    ml_path = os.path.join(BASE, "retrieval_indices.npy")
    if os.path.exists(ml_path):
        ml_indices = np.load(ml_path)

    # Baseline indices
    bl_indices = None
    bl_path = os.path.join(BASE, "retrieval_indices_baseline.npy")
    if os.path.exists(bl_path):
        bl_indices = np.load(bl_path)

    # Build: for each test image index → (folder, image_path)
    # extract_features.py iterates folders sequentially
    image_paths = []   # index i → absolute image path
    image_labels = []
    for lbl, folder in enumerate(folders):
        try:
            imgs = sorted([
                os.path.join(folder, fn)
                for fn in os.listdir(folder)
                if fn.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            for p in imgs:
                image_paths.append(p)
                image_labels.append(lbl)
        except FileNotFoundError:
            pass

    return folders, np.array(image_labels), image_paths, ml_indices, bl_indices


def get_thumb(path, w=THUMB_W, h=THUMB_H):
    """Load image → resized ImageTk.PhotoImage."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((w, h), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (24, 24, 46))
        ox = (w - img.width)  // 2
        oy = (h - img.height) // 2
        canvas.paste(img, (ox, oy))
        return ImageTk.PhotoImage(canvas)
    except Exception:
        canvas = Image.new("RGB", (w, h), (49, 50, 68))
        return ImageTk.PhotoImage(canvas)


def compute_metrics(indices, labels):
    n = len(labels)
    ap_list, acc1, acc10 = [], 0, 0
    for i in range(n):
        ranked_labels = labels[indices[i][1:]]   # skip self (index 0)
        if ranked_labels[0] == labels[i]:
            acc1 += 1
        if labels[i] in ranked_labels[:10]:
            acc10 += 1
        correct, precision_sum = 0, 0
        for rank, lbl in enumerate(ranked_labels):
            if lbl == labels[i]:
                correct += 1
                precision_sum += correct / (rank + 1)
        total_relevant = np.sum(labels == labels[i]) - 1
        ap_list.append(precision_sum / total_relevant if total_relevant > 0 else 0.0)
    return np.mean(ap_list), acc1/n*100, acc10/n*100


# ──────────────────────────────────────────────────────────────────────────────
# Demo App
# ──────────────────────────────────────────────────────────────────────────────

class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vase Retrieval — Baseline vs Metric Learning")
        self.root.configure(bg=BG)
        self.root.geometry("1400x820")
        self.root.resizable(True, True)

        self._loading_label = None
        self._build_loading_screen()
        threading.Thread(target=self._load_and_build, daemon=True).start()

    # ── Loading screen ────────────────────────────────────────────────────────

    def _build_loading_screen(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(frame, text="🏺", font=("Segoe UI Emoji", 48), bg=BG, fg=MAUVE).pack()
        tk.Label(frame, text="Loading retrieval data…", font=("Segoe UI", 14),
                 bg=BG, fg=SUBTEXT).pack(pady=8)
        self._loading_frame = frame

    def _load_and_build(self):
        try:
            folders, labels, image_paths, ml_indices, bl_indices = load_data()

            # Compute metrics
            bl_metrics = compute_metrics(bl_indices, labels) if bl_indices is not None else None
            ml_metrics = compute_metrics(ml_indices, labels) if ml_indices is not None else None

            self.data = dict(
                folders=folders, labels=labels, image_paths=image_paths,
                ml_indices=ml_indices, bl_indices=bl_indices,
                bl_metrics=bl_metrics, ml_metrics=ml_metrics,
            )
            self.query_idx = 0
            self._photo_refs = {}   # keep PhotoImage alive

            self.root.after(0, self._build_main_ui)
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))

    def _show_error(self, msg):
        self._loading_frame.destroy()
        tk.Label(self.root, text=f"❌ Error loading data:\n{msg}",
                 font=("Segoe UI", 12), bg=BG, fg=RED, justify="center").place(
                     relx=0.5, rely=0.5, anchor="center")

    # ── Main UI ───────────────────────────────────────────────────────────────

    def _build_main_ui(self):
        self._loading_frame.destroy()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",   background=BG)
        style.configure("TLabel",   background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TButton",  background=SURFACE2, foreground=TEXT,
                         font=("Segoe UI", 10, "bold"), padding=8)
        style.map("TButton", background=[("active", OVERLAY)])
        style.configure("Header.TLabel", background=BG, foreground=BLUE,
                         font=("Segoe UI", 13, "bold"))
        style.configure("Score.TLabel", background=SURFACE, foreground=GREEN,
                         font=("Cascadia Code", 10, "bold"), padding=6)

        d = self.data
        n_queries = len(d["image_paths"])

        # ── TOP BAR ──
        top = tk.Frame(self.root, bg=SURFACE, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="🏺  Vase Retrieval Demo",
                 font=("Segoe UI", 16, "bold"), bg=SURFACE, fg=BLUE).pack(side="left", padx=20)

        tk.Label(top, text="Master's Project · FAU · Midhun Somanunnithan",
                 font=("Segoe UI", 10), bg=SURFACE, fg=SUBTEXT).pack(side="left", padx=4)

        # Metric badges (top right)
        badge_frame = tk.Frame(top, bg=SURFACE)
        badge_frame.pack(side="right", padx=20)

        if d["bl_metrics"]:
            bm, ba1, ba10 = d["bl_metrics"]
            self._make_badge(badge_frame, "Baseline", bm, ba1, ba10, PEACH).pack(side="right", padx=6)

        if d["ml_metrics"]:
            mm, ma1, ma10 = d["ml_metrics"]
            self._make_badge(badge_frame, "Metric Learning", mm, ma1, ma10, GREEN).pack(side="right", padx=6)

        # ── QUERY NAV BAR ──
        nav = tk.Frame(self.root, bg=BG, pady=8)
        nav.pack(fill="x", padx=20)

        tk.Label(nav, text="Query image:", font=("Segoe UI", 11, "bold"),
                 bg=BG, fg=LAVENDER).pack(side="left")

        self.query_var = tk.StringVar()
        self.query_cb = ttk.Combobox(nav, textvariable=self.query_var, state="readonly",
                                      width=55, font=("Segoe UI", 10))
        paths = d["image_paths"]
        folder_labels = [os.path.basename(os.path.dirname(p)) + " / " + os.path.basename(p)
                         for p in paths]
        self.query_cb["values"] = folder_labels
        self.query_cb.current(0)
        self.query_cb.pack(side="left", padx=10)
        self.query_cb.bind("<<ComboboxSelected>>", self._on_query_change)

        tk.Button(nav, text="◀  Prev", command=self._prev_query,
                  bg=SURFACE2, fg=TEXT, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=10, pady=4, activebackground=OVERLAY,
                  activeforeground=TEXT, bd=0, cursor="hand2").pack(side="left", padx=4)

        tk.Button(nav, text="Next  ▶", command=self._next_query,
                  bg=SURFACE2, fg=TEXT, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=10, pady=4, activebackground=OVERLAY,
                  activeforeground=TEXT, bd=0, cursor="hand2").pack(side="left", padx=4)

        tk.Button(nav, text="🎲 Random", command=self._random_query,
                  bg=SURFACE2, fg=MAUVE, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=10, pady=4, activebackground=OVERLAY,
                  activeforeground=MAUVE, bd=0, cursor="hand2").pack(side="left", padx=4)

        self.query_info = tk.Label(nav, text="", font=("Segoe UI", 10),
                                    bg=BG, fg=SUBTEXT)
        self.query_info.pack(side="left", padx=14)

        # ── MAIN PANEL ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        # --- Query image (left) ---
        left = tk.Frame(main, bg=SURFACE, width=220, bd=0, relief="flat")
        left.pack(side="left", fill="y", padx=(0, 12), pady=4)
        left.pack_propagate(False)

        tk.Label(left, text="Query", font=("Segoe UI", 11, "bold"),
                 bg=SURFACE, fg=LAVENDER).pack(pady=(12, 4))

        self.query_canvas = tk.Label(left, bg=SURFACE)
        self.query_canvas.pack(pady=4)

        self.query_name_lbl = tk.Label(left, text="", font=("Segoe UI", 9),
                                        bg=SURFACE, fg=SUBTEXT, wraplength=200)
        self.query_name_lbl.pack(pady=(0, 6), padx=6)

        tk.Label(left, text="Folder label:", font=("Segoe UI", 9, "bold"),
                 bg=SURFACE, fg=SUBTEXT).pack()
        self.query_lbl_lbl = tk.Label(left, text="", font=("Cascadia Code", 12, "bold"),
                                       bg=SURFACE, fg=YELLOW)
        self.query_lbl_lbl.pack(pady=(0, 8))

        # Legend
        sep = tk.Frame(left, bg=OVERLAY, height=1)
        sep.pack(fill="x", padx=10, pady=6)
        tk.Label(left, text="Result border colours:", font=("Segoe UI", 9, "bold"),
                 bg=SURFACE, fg=SUBTEXT).pack(pady=(4, 2))
        for colour, meaning in [(GREEN, "✔ Correct class"), (RED, "✗ Wrong class")]:
            row = tk.Frame(left, bg=SURFACE)
            row.pack(anchor="w", padx=12, pady=1)
            tk.Frame(row, bg=colour, width=14, height=14).pack(side="left")
            tk.Label(row, text=f"  {meaning}", font=("Segoe UI", 9),
                     bg=SURFACE, fg=TEXT).pack(side="left")

        # --- Results columns (right) ---
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Two side-by-side panels
        panels = tk.Frame(right, bg=BG)
        panels.pack(fill="both", expand=True)

        self.bl_frame  = self._make_results_panel(panels, "Approach 1 — Baseline", "Raw ResNet-50 (2048-dim)", PEACH)
        self.ml_frame  = self._make_results_panel(panels, "Approach 2 — Metric Learning", "Triplet Loss MLP (128-dim)", GREEN)

        self.bl_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.ml_frame.pack(side="left", fill="both", expand=True)

        # First render
        self._render_query(0)

    def _make_badge(self, parent, title, mAP, acc1, acc10, colour):
        f = tk.Frame(parent, bg="#2a2a3e", bd=1, relief="solid")
        tk.Label(f, text=title, font=("Segoe UI", 9, "bold"),
                 bg="#2a2a3e", fg=colour).pack(padx=10, pady=(4, 0))
        tk.Label(f, text=f"mAP {mAP*100:.1f}%  |  Acc@1 {acc1:.1f}%  |  Acc@10 {acc10:.1f}%",
                 font=("Cascadia Code", 8), bg="#2a2a3e", fg=TEXT).pack(padx=10, pady=(0, 4))
        return f

    def _make_results_panel(self, parent, title, subtitle, colour):
        frame = tk.Frame(parent, bg=SURFACE, bd=0)

        # Header
        hdr = tk.Frame(frame, bg=SURFACE)
        hdr.pack(fill="x", pady=(10, 2), padx=10)
        tk.Label(hdr, text=title, font=("Segoe UI", 12, "bold"),
                 bg=SURFACE, fg=colour).pack(side="left")
        tk.Label(hdr, text=f"  ({subtitle})", font=("Segoe UI", 9),
                 bg=SURFACE, fg=SUBTEXT).pack(side="left")

        # Thumbnail grid — will be populated dynamically
        grid = tk.Frame(frame, bg=SURFACE)
        grid.pack(fill="both", expand=True, padx=6, pady=4)
        frame._grid = grid
        frame._colour = colour
        return frame

    # ── Query navigation ──────────────────────────────────────────────────────

    def _on_query_change(self, _=None):
        self._render_query(self.query_cb.current())

    def _prev_query(self):
        n = len(self.data["image_paths"])
        self._render_query((self.query_idx - 1) % n)

    def _next_query(self):
        n = len(self.data["image_paths"])
        self._render_query((self.query_idx + 1) % n)

    def _random_query(self):
        import random
        n = len(self.data["image_paths"])
        self._render_query(random.randint(0, n - 1))

    # ── Main render ──────────────────────────────────────────────────────────

    def _render_query(self, idx):
        self.query_idx = idx
        self.query_cb.current(idx)
        d = self.data
        path = d["image_paths"][idx]
        label = int(d["labels"][idx])

        # Query image
        big_thumb = get_thumb(path, 200, 180)
        self._photo_refs["query"] = big_thumb
        self.query_canvas.config(image=big_thumb)
        self.query_name_lbl.config(text=os.path.basename(path))
        self.query_lbl_lbl.config(text=f"#{label:03d}")
        self.query_info.config(
            text=f"Image {idx+1} / {len(d['image_paths'])}  ·  folder: {os.path.basename(os.path.dirname(path))}"
        )

        # Results
        if d["bl_indices"] is not None:
            self._render_results(self.bl_frame, d["bl_indices"][idx], label, "bl")
        if d["ml_indices"] is not None:
            self._render_results(self.ml_frame, d["ml_indices"][idx], label, "ml")

    def _render_results(self, panel, ranked_indices, query_label, prefix):
        grid = panel._grid
        # Clear previous widgets
        for w in grid.winfo_children():
            w.destroy()

        d = self.data
        show_indices = [i for i in ranked_indices if i != self.query_idx][:TOP_K]

        COLS = 3
        for pos, rid in enumerate(show_indices):
            r, c = divmod(pos, COLS)
            result_path  = d["image_paths"][rid]
            result_label = int(d["labels"][rid])
            is_correct   = (result_label == query_label)
            border_col   = GREEN if is_correct else RED

            cell = tk.Frame(grid, bg=border_col, bd=2)
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

            inner = tk.Frame(cell, bg=SURFACE2)
            inner.pack(fill="both", expand=True, padx=2, pady=2)

            ph = get_thumb(result_path, THUMB_W, THUMB_H)
            key = f"{prefix}_{pos}"
            self._photo_refs[key] = ph

            lbl_img = tk.Label(inner, image=ph, bg=SURFACE2, cursor="hand2")
            lbl_img.pack()

            rank_text = f"#{pos+1}  folder {result_label:03d}"
            status_text = "✔ Match" if is_correct else "✗ No match"
            status_col = GREEN if is_correct else RED

            info_row = tk.Frame(inner, bg=SURFACE2)
            info_row.pack(fill="x", padx=4)
            tk.Label(info_row, text=rank_text, font=("Segoe UI", 8),
                     bg=SURFACE2, fg=SUBTEXT).pack(side="left")
            tk.Label(info_row, text=status_text, font=("Segoe UI", 8, "bold"),
                     bg=SURFACE2, fg=status_col).pack(side="right")

            # Click to preview full image
            result_path_copy = result_path
            lbl_img.bind("<Button-1>", lambda e, p=result_path_copy: self._preview(p))

        # Make grid columns equal-width
        for c in range(COLS):
            grid.columnconfigure(c, weight=1)

    def _preview(self, path):
        """Open a larger preview popup."""
        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win.configure(bg=BG)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((600, 500), Image.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            lbl = tk.Label(win, image=ph, bg=BG)
            lbl._img = ph
            lbl.pack(padx=16, pady=16)
        except Exception as e:
            tk.Label(win, text=str(e), bg=BG, fg=RED).pack(padx=20, pady=20)
        tk.Label(win, text=path, font=("Segoe UI", 8), bg=BG, fg=SUBTEXT).pack(pady=(0, 8))


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("ERROR: Pillow is required.  Install with:  pip install Pillow")
        sys.exit(1)

    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()
