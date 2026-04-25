import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import time
import os
import queue as Queue

# ── CONFIG ─────────────────────────────────────────────────────────────────────
LOCAL_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = {
    "v5_dinov2/train.py":            "python v5_dinov2/train.py",
    "v5_dinov2/evaluate.py":         "python v5_dinov2/evaluate.py",
    "v5_dinov2/extract_features.py": "python v5_dinov2/extract_features.py",
    "v5_dinov2/retrieve.py":         "python v5_dinov2/retrieve.py",
    "tools/split_data.py":           "python tools/split_data.py",
}

# Full pipeline in execution order
PIPELINE = [
    "tools/split_data.py",
    "v5_dinov2/extract_features.py",
    "v5_dinov2/train.py",
    "v5_dinov2/retrieve.py",
    "v5_dinov2/evaluate.py",
]
LOG_DIR = os.path.join(LOCAL_PROJECT, "logs")
# ───────────────────────────────────────────────────────────────────────────────


def _fmt_size(n):
    if n < 1024:       return f"{n} B"
    if n < 1024**2:    return f"{n//1024} KB"
    if n < 1024**3:    return f"{n//1024**2} MB"
    return f"{n//1024**3} GB"


def _fmt_elapsed(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        lines = []
        for i, line in enumerate(out.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5:
                name, util, mem_used, mem_total, temp = parts
                lines.append(
                    f"GPU {i}: {name}\n"
                    f"  Util: {util}%   Temp: {temp}°C\n"
                    f"  VRAM: {mem_used} / {mem_total} MiB\n"
                )
        return "\n".join(lines) if lines else "No NVIDIA GPU found."
    except FileNotFoundError:
        return "nvidia-smi not found. No NVIDIA driver or CPU-only machine."
    except Exception as e:
        return f"nvidia-smi error: {e}"


class ProcessEntry:
    """Tracks a single subprocess run."""
    def __init__(self, pid, label, proc, log_path, start_time):
        self.pid        = pid
        self.label      = label
        self.proc       = proc
        self.log_path   = log_path
        self.start_time = start_time
        self.state      = "RUNNING"   # RUNNING | DONE | FAILED | KILLED
        self.exit_code  = None


class LocalManager:
    def __init__(self, root):
        self.root        = root
        self.root.title("Local Training Manager — Midhun")
        self.root.geometry("1200x750")
        self.root.configure(bg="#1e1e2e")

        self.processes    = {}    # pid -> ProcessEntry
        self.auto_refresh = True
        self.log_queue    = Queue.Queue()   # lines from active subprocesses

        os.makedirs(LOG_DIR, exist_ok=True)
        self._build_ui()
        self._start_auto_refresh()
        self._poll_log_queue()

    # ── UI ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",      background="#1e1e2e")
        style.configure("TLabel",      background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TButton",     background="#313244", foreground="#cdd6f4", font=("Segoe UI", 10), padding=6)
        style.configure("TCombobox",   fieldbackground="#313244", background="#313244",
                         foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Treeview",    background="#181825", foreground="#cdd6f4",
                         fieldbackground="#181825", font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#313244", foreground="#89b4fa",
                         font=("Segoe UI", 10, "bold"))
        style.configure("TEntry",      fieldbackground="#313244", foreground="#cdd6f4",
                         font=("Segoe UI", 10))
        style.map("TButton", background=[("active", "#45475a")])

        # ── TOP BAR ─────────────────────────────────────────────────────────────
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Local Training Manager",
                  font=("Segoe UI", 14, "bold"), foreground="#89b4fa").pack(side="left")

        self.status_lbl = ttk.Label(top, text="● Ready", foreground="#a6e3a1",
                                     font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side="left", padx=20)

        for txt, cmd in [("View Results",  self.view_results),
                          ("Kill Selected", self.kill_selected),
                          ("Refresh",       self.refresh_all),
                          ("Run Pipeline",  self.run_pipeline),
                          ("Run",           self.run_script)]:
            ttk.Button(top, text=txt, command=cmd).pack(side="right", padx=4)

        # ── SCRIPT LAUNCHER BAR ─────────────────────────────────────────────────
        launch = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        launch.pack(fill="x")

        ttk.Label(launch, text="Script:").pack(side="left")
        self.script_var = tk.StringVar(value=list(SCRIPTS.keys())[0])
        cb = ttk.Combobox(launch, textvariable=self.script_var,
                          values=list(SCRIPTS.keys()), width=22, state="readonly")
        cb.pack(side="left", padx=(4, 12))

        ttk.Label(launch, text="Extra args:").pack(side="left")
        self.args_var = tk.StringVar()
        args_entry = ttk.Entry(launch, textvariable=self.args_var, width=50)
        args_entry.pack(side="left", padx=4)

        # ── MAIN AREA ────────────────────────────────────────────────────────────
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Left — local file browser
        left = ttk.Frame(main, width=300)
        left.pack(side="left", fill="both", padx=(0, 6))
        left.pack_propagate(False)

        ttk.Label(left, text="Project Files", foreground="#a6e3a1",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))

        self.file_tree = ttk.Treeview(left, show="tree headings")
        self.file_tree["columns"] = ("size",)
        self.file_tree.column("#0",   width=190)
        self.file_tree.column("size", width=70, anchor="e")
        self.file_tree.heading("#0",   text="Name")
        self.file_tree.heading("size", text="Size")
        self.file_tree.pack(fill="both", expand=True)
        self.file_tree.bind("<Double-1>", self._on_file_double_click)
        self._populate_file_tree()

        # Right panel
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # Process queue table
        ttk.Label(right, text="Running Processes", foreground="#a6e3a1",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))

        proc_frame = ttk.Frame(right)
        proc_frame.pack(fill="x")

        self.proc_tree = ttk.Treeview(proc_frame, columns=("pid", "script", "state", "elapsed", "log"),
                                       show="headings", height=5)
        for col, w, label in [("pid", 60, "PID"), ("script", 180, "Script"),
                                ("state", 90, "State"), ("elapsed", 90, "Elapsed"),
                                ("log", 220, "Log File")]:
            self.proc_tree.column(col, width=w, anchor="center")
            self.proc_tree.heading(col, text=label)
        self.proc_tree.pack(fill="x", side="left", expand=True)

        proc_vsb = ttk.Scrollbar(proc_frame, orient="vertical", command=self.proc_tree.yview)
        self.proc_tree.configure(yscrollcommand=proc_vsb.set)
        proc_vsb.pack(side="right", fill="y")

        self.proc_tree.tag_configure("running", foreground="#a6e3a1")
        self.proc_tree.tag_configure("done",    foreground="#89b4fa")
        self.proc_tree.tag_configure("failed",  foreground="#f38ba8")
        self.proc_tree.tag_configure("killed",  foreground="#fab387")

        # Tabs
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True, pady=(8, 0))

        # Tab 1 — Live Log
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="  Live Log  ")

        log_top = ttk.Frame(log_frame)
        log_top.pack(fill="x", pady=(2, 2))
        ttk.Label(log_top, text="stdout / stderr from active run",
                  foreground="#cdd6f4").pack(side="left")
        ttk.Button(log_top, text="Clear", command=self._clear_log).pack(side="right", padx=4)

        self.log_box = scrolledtext.ScrolledText(
            log_frame, bg="#181825", fg="#a6e3a1",
            font=("Cascadia Code", 9), insertbackground="white")
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

        # Tab 2 — GPU Monitor
        gpu_frame = ttk.Frame(self.notebook)
        self.notebook.add(gpu_frame, text="  GPU Monitor  ")

        gpu_top = ttk.Frame(gpu_frame)
        gpu_top.pack(fill="x", pady=(2, 2))
        ttk.Button(gpu_top, text="Refresh GPU", command=self._refresh_gpu).pack(side="right", padx=4)

        self.gpu_box = scrolledtext.ScrolledText(
            gpu_frame, bg="#181825", fg="#cba6f7",
            font=("Cascadia Code", 9), insertbackground="white", height=8)
        self.gpu_box.pack(fill="both", expand=True)
        self.gpu_box.configure(state="disabled")

        # Tab 3 — Notifications
        notif_frame = ttk.Frame(self.notebook)
        self.notebook.add(notif_frame, text="  Notifications  ")

        notif_top = ttk.Frame(notif_frame)
        notif_top.pack(fill="x", pady=(4, 2))
        ttk.Label(notif_top, text="All Notifications", foreground="#a6e3a1",
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(notif_top, text="Clear", command=self._clear_notifications).pack(side="right", padx=4)

        self.notif_box = scrolledtext.ScrolledText(
            notif_frame, bg="#181825", fg="#fab387",
            font=("Cascadia Code", 9), insertbackground="white")
        self.notif_box.pack(fill="both", expand=True)
        self.notif_box.configure(state="disabled")

        # Initial GPU read
        threading.Thread(target=self._refresh_gpu, daemon=True).start()

    # ── FILE BROWSER ────────────────────────────────────────────────────────────

    def _populate_file_tree(self):
        self.file_tree.delete(*self.file_tree.get_children())
        self._file_paths = {}

        def add_node(parent, path, label):
            node = self.file_tree.insert(parent, "end", text=label, values=("",))
            self._file_paths[node] = path
            try:
                entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir():
                        sub = self.file_tree.insert(node, "end", text=entry.name + "/", values=("",))
                        self._file_paths[sub] = entry.path
                        # lazy placeholder
                        self.file_tree.insert(sub, "end", text="...", values=("",))
                    else:
                        size = _fmt_size(entry.stat().st_size)
                        fn = self.file_tree.insert(node, "end", text=entry.name, values=(size,))
                        self._file_paths[fn] = entry.path
            except PermissionError:
                pass
            return node

        root_node = add_node("", LOCAL_PROJECT, os.path.basename(LOCAL_PROJECT) + "/")
        self.file_tree.item(root_node, open=True)

    def _on_file_double_click(self, event):
        node = self.file_tree.focus()
        path = self._file_paths.get(node, "")
        if os.path.isdir(path):
            # Expand if has placeholder
            children = self.file_tree.get_children(node)
            if children and self.file_tree.item(children[0])["text"] == "...":
                self.file_tree.delete(children[0])
                try:
                    entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
                    for entry in entries:
                        if entry.name.startswith("."):
                            continue
                        if entry.is_dir():
                            sub = self.file_tree.insert(node, "end", text=entry.name + "/", values=("",))
                            self._file_paths[sub] = entry.path
                            self.file_tree.insert(sub, "end", text="...", values=("",))
                        else:
                            size = _fmt_size(entry.stat().st_size)
                            fn = self.file_tree.insert(node, "end", text=entry.name, values=(size,))
                            self._file_paths[fn] = entry.path
                except PermissionError:
                    pass
        elif os.path.isfile(path):
            threading.Thread(target=self._show_file, args=(path,), daemon=True).start()

    def _show_file(self, path):
        try:
            with open(path, "r", errors="replace") as f:
                content = f.read(10000)
            self.root.after(0, lambda: self._show_file_popup(path, content))
        except Exception as e:
            self._log(f"Could not read {path}: {e}")

    def _show_file_popup(self, path, content):
        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win.geometry("750x550")
        win.configure(bg="#1e1e2e")
        txt = scrolledtext.ScrolledText(win, bg="#181825", fg="#cdd6f4",
                                         font=("Cascadia Code", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("end", content)
        txt.configure(state="disabled")

    # ── ACTIONS ─────────────────────────────────────────────────────────────────

    def run_script(self):
        script_name = self.script_var.get()
        extra_args  = self.args_var.get().strip()
        base_cmd    = SCRIPTS.get(script_name, f"python {script_name}")
        cmd         = f"{base_cmd} {extra_args}".strip()

        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        log_file   = os.path.join(LOG_DIR, f"{script_name.replace('.py','')}_{timestamp}.log")

        self._log(f"Launching: {cmd}")
        self._log(f"Log → {log_file}")

        def _run():
            try:
                proc = subprocess.Popen(
                    cmd, shell=True, cwd=LOCAL_PROJECT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=1, universal_newlines=True
                )
                entry = ProcessEntry(
                    pid=proc.pid, label=script_name,
                    proc=proc, log_path=log_file,
                    start_time=time.time()
                )
                self.processes[proc.pid] = entry
                self.root.after(0, lambda: self._refresh_proc_table())
                self.root.after(0, lambda: self._notify(
                    "Run Started", f"{script_name} launched (PID {proc.pid})"))

                with open(log_file, "w") as lf:
                    for line in proc.stdout:
                        lf.write(line)
                        self.log_queue.put(line)

                proc.wait()
                entry.exit_code = proc.returncode
                entry.state = "DONE" if proc.returncode == 0 else "FAILED"

                msg = f"{script_name} finished (exit {proc.returncode})"
                self.root.after(0, lambda: self._notify(
                    "Run Finished" if proc.returncode == 0 else "Run Failed!", msg))
                self.root.after(0, lambda: self._refresh_proc_table())

            except Exception as e:
                self.root.after(0, lambda: self._notify("Launch Error", str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def run_pipeline(self):
        if not messagebox.askyesno(
            "Run Full Pipeline",
            "Run all scripts in order?\n\n" +
            "\n".join(f"  {i+1}. {s}" for i, s in enumerate(PIPELINE))
        ):
            return

        def _run_all():
            self.root.after(0, lambda: self._notify(
                "Pipeline Started", f"Running {len(PIPELINE)} steps in order"))
            for i, script_name in enumerate(PIPELINE):
                step = f"[{i+1}/{len(PIPELINE)}] {script_name}"
                self.log_queue.put(f"\n{'='*50}\n{step}\n{'='*50}\n")
                self.root.after(0, lambda s=step: self.status_lbl.config(
                    text=f"● {s}", foreground="#f9e2af"))

                cmd = SCRIPTS[script_name]
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file  = os.path.join(LOG_DIR, f"{script_name.replace('.py','')}_{timestamp}.log")

                try:
                    proc = subprocess.Popen(
                        cmd, shell=True, cwd=LOCAL_PROJECT,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=1, universal_newlines=True
                    )
                    entry = ProcessEntry(
                        pid=proc.pid, label=script_name,
                        proc=proc, log_path=log_file,
                        start_time=time.time()
                    )
                    self.processes[proc.pid] = entry
                    self.root.after(0, self._refresh_proc_table)

                    with open(log_file, "w") as lf:
                        for line in proc.stdout:
                            lf.write(line)
                            self.log_queue.put(line)

                    proc.wait()
                    entry.exit_code = proc.returncode
                    entry.state = "DONE" if proc.returncode == 0 else "FAILED"
                    self.root.after(0, self._refresh_proc_table)

                    if proc.returncode != 0:
                        self.root.after(0, lambda s=script_name, c=proc.returncode: self._notify(
                            "Pipeline ABORTED",
                            f"{s} failed (exit {c}) — stopping pipeline."))
                        self.root.after(0, lambda: self.status_lbl.config(
                            text="● Pipeline Failed", foreground="#f38ba8"))
                        return

                except Exception as e:
                    self.root.after(0, lambda err=str(e): self._notify("Pipeline Error", err))
                    self.root.after(0, lambda: self.status_lbl.config(
                        text="● Pipeline Error", foreground="#f38ba8"))
                    return

            self.root.after(0, lambda: self._notify(
                "Pipeline Complete!", "All steps finished successfully."))
            self.root.after(0, lambda: self.status_lbl.config(
                text="● Pipeline Done", foreground="#a6e3a1"))

        threading.Thread(target=_run_all, daemon=True).start()

    def kill_selected(self):
        sel = self.proc_tree.selection()
        if not sel:
            messagebox.showinfo("Kill", "Select a process row first.")
            return
        item   = sel[0]
        values = self.proc_tree.item(item)["values"]
        pid    = int(values[0])
        entry  = self.processes.get(pid)
        if not entry or entry.state != "RUNNING":
            messagebox.showinfo("Kill", f"PID {pid} is not running.")
            return
        if not messagebox.askyesno("Kill Process", f"Kill {entry.label} (PID {pid})?"):
            return
        try:
            entry.proc.kill()
            entry.state = "KILLED"
            self._log(f"Killed PID {pid}")
            self._refresh_proc_table()
        except Exception as e:
            messagebox.showerror("Kill Error", str(e))

    def view_results(self):
        """Open the most recent log file."""
        logs = sorted(
            [f for f in os.listdir(LOG_DIR) if f.endswith(".log")],
            reverse=True
        ) if os.path.isdir(LOG_DIR) else []
        if not logs:
            messagebox.showinfo("No Results", "No log files found yet.")
            return
        path = os.path.join(LOG_DIR, logs[0])
        threading.Thread(target=self._show_file, args=(path,), daemon=True).start()

    def refresh_all(self):
        self._refresh_proc_table()
        threading.Thread(target=self._refresh_gpu, daemon=True).start()
        self._populate_file_tree()

    # ── PROCESS TABLE ────────────────────────────────────────────────────────────

    def _refresh_proc_table(self):
        self.proc_tree.delete(*self.proc_tree.get_children())
        for pid, entry in sorted(self.processes.items(), key=lambda x: -x[1].start_time):
            elapsed = _fmt_elapsed(time.time() - entry.start_time)
            tag = entry.state.lower()
            self.proc_tree.insert("", "end",
                values=(pid, entry.label, entry.state, elapsed, os.path.basename(entry.log_path)),
                tags=(tag,))

    # ── LOG ─────────────────────────────────────────────────────────────────────

    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg.rstrip("\n") + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _poll_log_queue(self):
        """Drain the log queue into the UI — safe to call from main thread."""
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._log(line)
        except Queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    # ── GPU ─────────────────────────────────────────────────────────────────────

    def _refresh_gpu(self):
        text = run_nvidia_smi()
        def _update():
            self.gpu_box.configure(state="normal")
            self.gpu_box.delete("1.0", "end")
            self.gpu_box.insert("end", f"[{time.strftime('%H:%M:%S')}]\n{text}")
            self.gpu_box.configure(state="disabled")
        self.root.after(0, _update)

    # ── NOTIFICATIONS ────────────────────────────────────────────────────────────

    def _notify(self, title, msg):
        self._log(f">> {title}: {msg}")
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {title}\n  {msg}\n{'─'*50}\n"
        self.notif_box.configure(state="normal")
        self.notif_box.insert("1.0", entry)
        self.notif_box.configure(state="disabled")
        self.notebook.select(2)
        messagebox.showinfo(title, msg)

    def _clear_notifications(self):
        self.notif_box.configure(state="normal")
        self.notif_box.delete("1.0", "end")
        self.notif_box.configure(state="disabled")

    # ── AUTO REFRESH ─────────────────────────────────────────────────────────────

    def _start_auto_refresh(self):
        def _loop():
            while self.auto_refresh:
                time.sleep(10)
                self.root.after(0, self._refresh_proc_table)
                threading.Thread(target=self._refresh_gpu, daemon=True).start()
        threading.Thread(target=_loop, daemon=True).start()


# ── ENTRY POINT ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = LocalManager(root)
    root.mainloop()
