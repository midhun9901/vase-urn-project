import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import paramiko
import threading
import time
import os

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SSH_HOST    = "tinyx.nhr.fau.de"
SSH_USER    = "iwi5419h"
SSH_KEY     = os.path.expanduser("~/.ssh/id_ed25519_nhr_fau")
PROXY_HOST  = "csnhr.nhr.fau.de"
HPC_PROJECT = "/home/woody/iwi5/iwi5419h/vase_project"
HPC_WORK    = "/home/woody/iwi5/iwi5419h"
JOB_SCRIPT  = "job.sh"
# ───────────────────────────────────────────────────────────────────────────────


def make_ssh():
    """Open SSH connection via ProxyJump (csnhr → tinygpu)."""
    proxy_client = paramiko.SSHClient()
    proxy_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    proxy_client.connect(PROXY_HOST, username=SSH_USER, key_filename=SSH_KEY)

    proxy_transport = proxy_client.get_transport()
    dest_addr  = (SSH_HOST, 22)
    local_addr = ("127.0.0.1", 0)
    channel    = proxy_transport.open_channel("direct-tcpip", dest_addr, local_addr)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, username=SSH_USER, key_filename=SSH_KEY, sock=channel)
    return client, proxy_client


def run_cmd(client, cmd):
    """Run a command and return stdout + stderr."""
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out + err


class HPCManager:
    def __init__(self, root):
        self.root = root
        self.root.title("HPC Job Manager — Midhun")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e2e")

        self.client       = None
        self.proxy        = None
        self.connected    = False
        self.auto_refresh = False
        self.job_states   = {}   # jobid -> last known state

        self._build_ui()

    # ── UI LAYOUT ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",       background="#1e1e2e")
        style.configure("TLabel",       background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TButton",      background="#313244", foreground="#cdd6f4", font=("Segoe UI", 10), padding=6)
        style.configure("Treeview",     background="#181825", foreground="#cdd6f4", fieldbackground="#181825", font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#313244", foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#45475a")])

        # ── TOP BAR ─────────────────────────────────────────────────────────────
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="HPC Job Manager", font=("Segoe UI", 14, "bold"),
                  foreground="#89b4fa").pack(side="left")

        self.status_lbl = ttk.Label(top, text="● Disconnected", foreground="#f38ba8",
                                    font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side="left", padx=20)

        for txt, cmd in [("Connect",      self.connect),
                          ("Refresh All",  self.refresh_all),
                          ("Submit Job",   self.submit_job),
                          ("View Results", self.view_results)]:
            ttk.Button(top, text=txt, command=cmd).pack(side="right", padx=4)

        # ── MAIN AREA ────────────────────────────────────────────────────────────
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Left — file explorer
        left = ttk.Frame(main, width=320)
        left.pack(side="left", fill="both", padx=(0, 6))
        left.pack_propagate(False)

        ttk.Label(left, text="Files on HPC", foreground="#a6e3a1",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))

        self.tree = ttk.Treeview(left, show="tree headings")
        self.tree["columns"] = ("size",)
        self.tree.column("#0",    width=200)
        self.tree.column("size",  width=80, anchor="e")
        self.tree.heading("#0",   text="Name")
        self.tree.heading("size", text="Size")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        # Right panel with tabs
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # Queue status (always visible at top)
        ttk.Label(right, text="Job Queue", foreground="#a6e3a1",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))

        self.queue_box = scrolledtext.ScrolledText(
            right, height=8, bg="#181825", fg="#cdd6f4",
            font=("Cascadia Code", 9), insertbackground="white")
        self.queue_box.pack(fill="x")

        # Tabs for Log and Notifications
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True, pady=(8, 0))

        # Tab 1 — Output / Live Log
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="  Output / Live Log  ")
        self.log_box = scrolledtext.ScrolledText(
            log_frame, bg="#181825", fg="#a6e3a1",
            font=("Cascadia Code", 9), insertbackground="white")
        self.log_box.pack(fill="both", expand=True)

        # Tab 2 — Notifications
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

    # ── HELPERS ─────────────────────────────────────────────────────────────────

    def log(self, msg, color=None):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_queue(self, text):
        self.queue_box.configure(state="normal")
        self.queue_box.delete("1.0", "end")
        self.queue_box.insert("end", text)
        self.queue_box.configure(state="disabled")

    # ── ACTIONS ─────────────────────────────────────────────────────────────────

    def connect(self):
        self.log("Connecting to TinyGPU via csnhr...")
        def _connect():
            try:
                self.client, self.proxy = make_ssh()
                self.connected = True
                self.status_lbl.config(text="● Connected", foreground="#a6e3a1")
                self.log("Connected!")
                self.refresh_all()
                self._start_auto_refresh()
            except Exception as e:
                self.log(f"Connection failed: {e}")
                self.status_lbl.config(text="● Error", foreground="#f38ba8")
        threading.Thread(target=_connect, daemon=True).start()

    def refresh_all(self):
        if not self.connected:
            self.log("Not connected. Click Connect first.")
            return
        threading.Thread(target=self._do_refresh, daemon=True).start()

    def _do_refresh(self):
        self._refresh_queue()
        self._refresh_files()

    def _refresh_queue(self):
        try:
            out = run_cmd(self.client, "squeue -u $USER")
            display = out if out.strip() else "No jobs in queue."
            self.root.after(0, lambda: self.set_queue(display))
            self._check_job_transitions(out)
        except Exception as e:
            self.root.after(0, lambda: self.set_queue(f"Error: {e}"))

    def _check_job_transitions(self, squeue_out):
        """Detect job state changes and notify user."""
        current_states = {}
        for line in squeue_out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                jobid, state = parts[0], parts[4]
                current_states[jobid] = state

        for jobid, state in current_states.items():
            prev = self.job_states.get(jobid)
            if prev != state:
                if state == "R" and prev == "PD":
                    self.root.after(0, lambda j=jobid: self._notify(
                        "Job Started!", f"Job {j} is now RUNNING on the A100 GPU!"))
                elif state not in ("PD", "R"):
                    self.root.after(0, lambda j=jobid, s=state: self._notify(
                        "Job Status Changed", f"Job {j} changed to: {s}"))

        # Detect jobs that disappeared (finished)
        for jobid in list(self.job_states):
            if self.job_states[jobid] in ("PD", "R") and jobid not in current_states:
                self.root.after(0, lambda j=jobid: self._on_job_finished(j))

        self.job_states = current_states

    def _add_notification(self, title, msg):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {title}\n  {msg}\n{'─'*50}\n"
        self.notif_box.configure(state="normal")
        self.notif_box.insert("1.0", entry)  # newest at top
        self.notif_box.configure(state="disabled")
        # Switch to notifications tab
        self.notebook.select(1)

    def _clear_notifications(self):
        self.notif_box.configure(state="normal")
        self.notif_box.delete("1.0", "end")
        self.notif_box.configure(state="disabled")

    def _notify(self, title, msg):
        self.log(f">> {title}: {msg}")
        self._add_notification(title, msg)
        messagebox.showinfo(title, msg)

    def _on_job_finished(self, jobid):
        self.log(f">> Job {jobid} FINISHED! Fetching results...")
        messagebox.showinfo("Job Complete!", f"Job {jobid} finished!\nFetching results now...")
        self.view_results()

    def _refresh_files(self):
        try:
            sftp = self.client.open_sftp()
            self.root.after(0, lambda: self._populate_tree(sftp))
        except Exception as e:
            self.log(f"File refresh error: {e}")

    def _populate_tree(self, sftp):
        self.tree.delete(*self.tree.get_children())

        def add_dir(parent, path):
            try:
                items = sorted(sftp.listdir_attr(path), key=lambda x: x.filename)
                for item in items:
                    full = path + "/" + item.filename
                    is_dir = item.st_mode and (item.st_mode & 0o40000)
                    size = "" if is_dir else _fmt_size(item.st_size)
                    node = self.tree.insert(parent, "end", text=item.filename,
                                            values=(size,), tags=("dir" if is_dir else "file",))
                    if is_dir:
                        self.tree.insert(node, "end", text="Loading...")
                        self.tree._full_path = getattr(self.tree, "_full_path", {})
                        self.tree._full_path[node] = full
            except Exception:
                pass

        root_node = self.tree.insert("", "end", text="vase_project", values=("",))
        add_dir(root_node, HPC_PROJECT)
        self.tree.item(root_node, open=True)
        self.tree._sftp = sftp

    def on_tree_double_click(self, event):
        node = self.tree.focus()
        children = self.tree.get_children(node)
        # If first child is "Loading..." placeholder, expand it
        if children and self.tree.item(children[0])["text"] == "Loading...":
            self.tree.delete(children[0])
            path = getattr(self.tree, "_full_path", {}).get(node)
            if path and hasattr(self.tree, "_sftp"):
                def add_dir(parent, p):
                    try:
                        items = sorted(self.tree._sftp.listdir_attr(p), key=lambda x: x.filename)
                        for item in items:
                            full = p + "/" + item.filename
                            is_dir = item.st_mode and (item.st_mode & 0o40000)
                            size = "" if is_dir else _fmt_size(item.st_size)
                            n = self.tree.insert(parent, "end", text=item.filename,
                                                 values=(size,))
                            if is_dir:
                                self.tree.insert(n, "end", text="Loading...")
                                self.tree._full_path[n] = full
                    except Exception:
                        pass
                add_dir(node, path)
        else:
            # It's a file — show contents
            path = getattr(self.tree, "_full_path", {}).get(node)
            if path:
                threading.Thread(target=self._show_file, args=(path,), daemon=True).start()

    def _show_file(self, path):
        try:
            sftp = self.tree._sftp
            with sftp.open(path, "r") as f:
                content = f.read(8000).decode(errors="replace")
            self.root.after(0, lambda: self._show_file_popup(path, content))
        except Exception as e:
            self.log(f"Could not read file: {e}")

    def _show_file_popup(self, path, content):
        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win.geometry("700x500")
        win.configure(bg="#1e1e2e")
        txt = scrolledtext.ScrolledText(win, bg="#181825", fg="#cdd6f4",
                                         font=("Cascadia Code", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("end", content)
        txt.configure(state="disabled")

    def submit_job(self):
        if not self.connected:
            self.log("Not connected.")
            return
        if not messagebox.askyesno("Submit Job", "Submit job.sh to TinyGPU A100?"):
            return
        def _submit():
            self.log("Submitting job...")
            out = run_cmd(self.client, f"cd {HPC_PROJECT} && sbatch.tinygpu {JOB_SCRIPT}")
            self.log(out)
            time.sleep(2)
            self._refresh_queue()
        threading.Thread(target=_submit, daemon=True).start()

    def view_results(self):
        if not self.connected:
            self.log("Not connected.")
            return
        def _results():
            # Find latest .out file
            out = run_cmd(self.client, f"ls -t {HPC_PROJECT}/vase_*.out 2>/dev/null | head -1")
            latest = out.strip()
            if not latest:
                self.log("No output files found yet. Job may still be running.")
                return
            content = run_cmd(self.client, f"cat {latest}")
            self.root.after(0, lambda: self._show_file_popup(latest, content))
        threading.Thread(target=_results, daemon=True).start()

    def _start_auto_refresh(self):
        self.auto_refresh = True
        def _loop():
            while self.auto_refresh and self.connected:
                time.sleep(30)
                if self.connected:
                    self._refresh_queue()
        threading.Thread(target=_loop, daemon=True).start()


def _fmt_size(n):
    if n < 1024:         return f"{n} B"
    if n < 1024**2:      return f"{n//1024} KB"
    if n < 1024**3:      return f"{n//1024**2} MB"
    return f"{n//1024**3} GB"


if __name__ == "__main__":
    root = tk.Tk()
    app = HPCManager(root)
    root.mainloop()
