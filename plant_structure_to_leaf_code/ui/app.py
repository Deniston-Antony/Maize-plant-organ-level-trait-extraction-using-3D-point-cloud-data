"""Tkinter front end for the maize plant-structure pipeline.

Tabs
  Setup   : load scans (one LAS per date), optional ground truth, output folder, run
  Results : ground-truth summary table + built-in 3D viewer (matplotlib, rotatable)
A log panel at the bottom shows progress and a live timer (elapsed, time left,
expected end). Long jobs run in a worker thread.
"""

from __future__ import annotations

import csv
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from pipeline.config import (  # noqa: E402
    ASSETS_DIR, DATASET_DIR, OUTPUT_ROOT, REFERENCE_DATE, date_from_name, pretty_date,
)
from pipeline.progress import ProgressInfo, fmt_seconds  # noqa: E402
from ui.help_content import GT_HELP_TEXT, GT_HELP_TITLE, make_sample_image  # noqa: E402
from ui.step_viewer import StepViewer  # noqa: E402
from pipeline.names import nice  # noqa: E402

RUN_STEPS = [("1  Soil and plants only", "soil"), ("2  + alignment of the dates", "alignment"),
             ("3  + stems and plant clusters", "stems"), ("4  + plant instances, stem / leaves, structure", "plants"),
             ("5  everything (+ tracking, growth, report)", "all")]

SAMPLE_GT = CODE_DIR.parent / "P147" / "manual_segmented_leaf_file" / "iou_evaluation" / "LtR_10th_Plant_230621pc_kinematic_calibration_asc"
SAMPLE_PNG = ASSETS_DIR / "gt_sample.png"
AUTO = "auto"
CHECK_ON, CHECK_OFF = "\u2611", "\u2610"


class Tooltip:
    def __init__(self, widget, text: str):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", relief="solid",
                 borderwidth=1, padx=6, pady=3).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maize plant structure pipeline")
        self.geometry("1250x900")
        self.minsize(1000, 700)
        self.log_q: queue.Queue = queue.Queue()
        self.busy = False
        self.use_gt = tk.BooleanVar(value=False)
        self.ref_date = tk.StringVar(value=REFERENCE_DATE)
        self.out_root = tk.StringVar(value=str(OUTPUT_ROOT))
        self.run_name = tk.StringVar(value=datetime.now().strftime("run_%Y%m%d_%H%M"))
        self.last_run: Path | None = None
        self.gt_rows: list[dict] = []
        self._t_start: float | None = None      # monotonic start of the running job
        self._pinfo: ProgressInfo | None = None  # last progress report
        self._pinfo_t = 0.0                      # when it arrived (monotonic)

        style = ttk.Style(self)
        style.configure("Help.TButton", font=("Helvetica", 13, "bold"), padding=(4, 0))
        style.configure("Run.TButton", font=("Helvetica", 12, "bold"))

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = ttk.Notebook(paned)
        paned.add(self.nb, weight=4)
        self._build_setup_tab()
        self.viewer = StepViewer(self.nb, log=self.log)
        self.nb.add(self.viewer, text="  Step viewer  ")
        self._build_results_tab()

        logf = ttk.Frame(paned)
        paned.add(logf, weight=1)
        top = ttk.Frame(logf)
        top.pack(fill="x")
        ttk.Label(top, text="Log").pack(side="left")
        self.progress = ttk.Progressbar(top, mode="indeterminate", length=220, maximum=100)
        self.progress.pack(side="right")
        self.status = ttk.Label(top, text="ready")
        self.status.pack(side="right", padx=8)
        self.timer = ttk.Label(top, text="", font=("Menlo", 12, "bold"))
        self.timer.pack(side="right", padx=12)
        self.log_box = ScrolledText(logf, height=9, font=("Menlo", 11))
        self.log_box.pack(fill="both", expand=True)

        self._toggle_gt()
        self._autoload_datasets()
        self._update_run_hint()
        latest = self._latest_run()
        if latest:
            try:
                self.viewer.set_run(latest)
                self.last_run = latest
                self.log(f"Step viewer: showing the latest run '{latest.name}' (use 'Open run...' for another).")
            except Exception as e:  # noqa: BLE001
                self.log(f"Step viewer: could not load {latest.name}: {e}")
        self.after(100, self._poll_log)

    # ------------------------------------------------------------------ setup tab
    def _build_setup_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="  Setup  ")

        # 1. scans
        f1 = ttk.LabelFrame(tab, text=" 1. Point clouds - one LAS/LAZ per date; click the box to choose which dates to run ",
                            padding=6)
        f1.pack(fill="x", pady=4)
        self.scan_tree = self._tree(f1, ("run", "date", "file", "size"), (50, 110, 600, 90), height=5,
                                    headings=("Run", "Date", "File", "Size"))
        self.scan_tree.bind("<Button-1>", self._toggle_run_cell)
        b = ttk.Frame(f1)
        b.pack(fill="x", pady=(4, 0))
        ttk.Button(b, text="Tick all", command=lambda: self._tick_all(True)).pack(side="left")
        ttk.Button(b, text="Untick all", command=lambda: self._tick_all(False)).pack(side="left", padx=(4, 12))
        ttk.Button(b, text="Add scan file(s)...", command=self.add_scans).pack(side="left")
        ttk.Button(b, text="Add dataset folder...", command=self.add_scan_folder).pack(side="left", padx=4)
        ttk.Button(b, text="Remove selected", command=lambda: self._remove_sel(self.scan_tree)).pack(side="left")
        ttk.Label(b, text="   Reference date (one of the ticked dates):").pack(side="left")
        self.ref_combo = ttk.Combobox(b, textvariable=self.ref_date, width=10, state="readonly")
        self.ref_combo.pack(side="left", padx=4)

        # 2. ground truth
        f2 = ttk.LabelFrame(tab, text=" 2. Ground truth (optional) ", padding=6)
        f2.pack(fill="both", expand=True, pady=4)
        row = ttk.Frame(f2)
        row.pack(fill="x")
        ttk.Checkbutton(row, text="Compare predictions with ground truth", variable=self.use_gt,
                        command=self._toggle_gt).pack(side="left")
        self.gt_buttons = []
        for txt, cmd in (("Add GT folder...", self.add_gt_folder), ("Add GT file(s)...", self.add_gt_files)):
            w = ttk.Button(row, text=txt, command=cmd)
            w.pack(side="left", padx=(12 if not self.gt_buttons else 4, 0))
            self.gt_buttons.append(w)
        hb = ttk.Button(row, text="?", width=2, style="Help.TButton", command=self.show_gt_help)
        hb.pack(side="left", padx=4)
        Tooltip(hb, "How to name and label ground-truth files\n(click for instructions and a sample)")
        for txt, cmd in (("Check selected", self.validate_gt), ("Remove selected", lambda: self._remove_sel(self.gt_tree))):
            w = ttk.Button(row, text=txt, command=cmd)
            w.pack(side="left", padx=4)
            self.gt_buttons.append(w)
        self.gt_tree = self._tree(f2, ("name", "date", "type", "status"), (360, 110, 200, 420), height=6)
        self.gt_tree.bind("<Double-1>", self._edit_gt_date)
        self.gt_hint = ttk.Label(f2, foreground="#666",
                                 text="Double-click a date to change it. 'auto' = taken from the name, or every date is tried.")
        self.gt_hint.pack(anchor="w")

        # 3. output
        f3 = ttk.LabelFrame(tab, text=" 3. Output ", padding=6)
        f3.pack(fill="x", pady=4)
        r = ttk.Frame(f3)
        r.pack(fill="x")
        ttk.Label(r, text="Output folder:").pack(side="left")
        ttk.Entry(r, textvariable=self.out_root, width=80).pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(r, text="Browse...", command=self._browse_out).pack(side="left")
        r2 = ttk.Frame(f3)
        r2.pack(fill="x", pady=(4, 0))
        ttk.Label(r2, text="Run name:").pack(side="left")
        ttk.Entry(r2, textvariable=self.run_name, width=30).pack(side="left", padx=4)
        ttk.Label(r2, foreground="#666", text=(
            "-> <run>/<date>/01_soil, 02_stems_groups, 03_plants/groupNN/groupNN_plantMM ;  "
            "<run>/ground_truth, tracking, growth")).pack(side="left", padx=8)

        # run buttons
        rb = ttk.Frame(tab)
        rb.pack(fill="x", pady=8)
        ttk.Label(rb, text="Run up to step:").pack(side="left")
        self.stop_step = tk.StringVar(value=RUN_STEPS[-1][0])
        ttk.Combobox(rb, textvariable=self.stop_step, values=[n for n, _ in RUN_STEPS], width=40,
                     state="readonly").pack(side="left", padx=(4, 10))
        ttk.Button(rb, text="Run ticked dates", style="Run.TButton", command=self.run_pipeline).pack(side="left")
        self.run_hint = ttk.Label(rb, foreground="#666", text="")
        self.run_hint.pack(side="left", padx=8)
        ttk.Button(rb, text="Ground-truth check on an existing run...", command=self.run_gt_on_existing).pack(side="left", padx=8)

    # ------------------------------------------------------------------ results tab
    def _build_results_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="  Results  ")
        cols = ("gt", "date", "plant", "method", "plant_iou", "stem_iou", "leaf_iou", "mean_iou",
                "leaves", "height_err", "stem_len_err", "radius_err")
        widths = (230, 70, 230, 110, 70, 70, 70, 70, 70, 85, 90, 85)
        heads = ("Ground truth", "Date", "Matched plant instance", "Matched by", "Plant IoU", "Stem IoU", "Leaf IoU",
                 "Mean IoU", "Leaves GT/pred", "Height err cm", "Stem len err cm", "Radius err mm")
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="x")
        f_pl = ttk.LabelFrame(pw, text=" Plant instances - availability per date (select one for its 3D view) ", padding=4)
        f_gt = ttk.LabelFrame(pw, text=" Ground-truth check ", padding=4)
        pw.add(f_pl, weight=3)
        pw.add(f_gt, weight=2)
        self.plant_tree = self._tree(f_pl, ("plant",), (230,), height=9, headings=("Plant instance",))
        self.plant_tree.bind("<<TreeviewSelect>>", lambda _: self.show_plant_3d())
        self.res_tree = self._tree(f_gt, cols, widths, height=9, headings=heads)
        self.res_tree.bind("<<TreeviewSelect>>", lambda _: self.show_selected_3d())
        b = ttk.Frame(tab)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="Per-leaf details", command=self.show_leaf_details).pack(side="left")
        ttk.Button(b, text="Comparison picture", command=self.show_comparison_png).pack(side="left", padx=4)
        ttk.Button(b, text="Interactive 3D (PyVista)", command=self.open_pyvista).pack(side="left")
        ttk.Button(b, text="Open run folder", command=self.open_run_folder).pack(side="left", padx=4)
        ttk.Button(b, text="Open report", command=self.open_report).pack(side="left")
        ttk.Button(b, text="Load results of a run...", command=self.load_results_dialog).pack(side="left")
        self.view_frame = ttk.LabelFrame(tab, text=" Plant structure (your structure viewer) - drag: rotate, two fingers: zoom ", padding=4)
        self.view_frame.pack(fill="both", expand=True)
        self.canvas = None

    # ------------------------------------------------------------------ helpers
    def _tree(self, parent, cols, widths, height=5, headings=None):
        fr = ttk.Frame(parent)
        fr.pack(fill="both", expand=True)
        t = ttk.Treeview(fr, columns=cols, show="headings", height=height)
        for c, w, h in zip(cols, widths, headings or [c.capitalize() for c in cols]):
            t.heading(c, text=h)
            t.column(c, width=w, anchor="w", stretch=True)
        sb = ttk.Scrollbar(fr, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return t

    def log(self, msg: str):
        self.log_q.put(str(msg))

    def _poll_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                if callable(msg):
                    msg()
                    continue
                self.log_box.insert("end", msg + "\n")
                self.log_box.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _set_busy(self, busy: bool, text: str = ""):
        self.busy = busy
        self.status.config(text=text or ("working..." if busy else "ready"))
        if busy:
            self._t_start, self._pinfo = time.monotonic(), None
            self.progress.config(mode="indeterminate", value=0)
            self.progress.start(12)
            self._tick()
        else:
            self.progress.stop()
            if self._t_start is not None:
                self.progress.config(mode="determinate", value=100 if self._pinfo else 0)
                self.timer.config(text=f"done in {fmt_seconds(time.monotonic() - self._t_start)}", foreground="")
            self._t_start = None

    # live timer --------------------------------------------------------------
    def report_progress(self, info: ProgressInfo):
        """Thread-safe callback handed to the pipeline."""
        self.log_q.put(lambda: self._apply_progress(info))

    def _apply_progress(self, info: ProgressInfo):
        if self._t_start is None:
            return
        self._pinfo, self._pinfo_t = info, time.monotonic()
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.config(mode="determinate")
        self.progress.config(value=100 * info.fraction)
        self.status.config(text=f"{info.stage}  {100 * info.fraction:.0f} %")

    def _tick(self):
        if self._t_start is None:
            return
        elapsed = time.monotonic() - self._t_start
        txt = f"elapsed {fmt_seconds(elapsed)}"
        colour = ""
        info = self._pinfo
        if info and info.remaining_s is not None:
            left = max(0.0, info.remaining_s - (time.monotonic() - self._pinfo_t))  # count down between reports
            end = datetime.now() + timedelta(seconds=left)
            txt += f" | left ~{fmt_seconds(left)} | ends ~{end:%H:%M:%S}"
            if info.budget_s:
                txt += f" | budget {fmt_seconds(info.budget_s)}"
                colour = "#c0392b" if elapsed + left > info.budget_s * 1.05 else "#1e8449"
        elif info and info.budget_s:
            txt += f" | budget {fmt_seconds(info.budget_s)}"
        self.timer.config(text=txt, foreground=colour)
        self.after(250, self._tick)

    def _in_thread(self, fn, done=None, label="working..."):
        if self.busy:
            messagebox.showinfo("Busy", "Another job is still running.")
            return
        self._set_busy(True, label)

        def work():
            try:
                res = fn()
                self.log_q.put(lambda: done(res) if done else None)
            except Exception as e:
                import traceback
                self.log(traceback.format_exc())
                self.log_q.put(lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.log_q.put(lambda: self._set_busy(False))

        threading.Thread(target=work, daemon=True).start()

    def _remove_sel(self, tree):
        for i in tree.selection():
            tree.delete(i)
        self._refresh_dates()

    def _browse_out(self):
        d = filedialog.askdirectory(initialdir=self.out_root.get())
        if d:
            self.out_root.set(d)

    # ------------------------------------------------------------------ scans
    def _add_scan(self, p: Path):
        if any(self.scan_tree.set(i, "file") == str(p) for i in self.scan_tree.get_children()):
            return
        d = date_from_name(p.name) or "?"
        size = f"{p.stat().st_size / 1e9:.2f} GB"
        # only the reference date is ticked by default; tick the others you want
        tick = CHECK_ON if d == REFERENCE_DATE else CHECK_OFF
        self.scan_tree.insert("", "end", values=(tick, d, str(p), size))
        self._refresh_dates()

    def _toggle_run_cell(self, event):
        if self.scan_tree.identify_region(event.x, event.y) != "cell" or self.scan_tree.identify_column(event.x) != "#1":
            return
        iid = self.scan_tree.identify_row(event.y)
        if iid:
            v = self.scan_tree.set(iid, "run")
            self.scan_tree.set(iid, "run", CHECK_OFF if v == CHECK_ON else CHECK_ON)
            self._update_run_hint()

    def _tick_all(self, on: bool):
        for i in self.scan_tree.get_children():
            self.scan_tree.set(i, "run", CHECK_ON if on else CHECK_OFF)
        self._update_run_hint()

    def ticked_scans(self) -> list[tuple[str, str]]:
        return [(self.scan_tree.set(i, "file"), self.scan_tree.set(i, "date")) for i in self.scan_tree.get_children()
                if self.scan_tree.set(i, "run") == CHECK_ON]

    def _auto_run_name(self) -> str:
        dates = sorted(d for _, d in self.ticked_scans())
        return datetime.now().strftime("run_%Y%m%d_%H%M") + ("_" + "_".join(dates) if dates else "")

    def _refresh_reference(self):
        """Reference date = one of the ticked dates (29.06 when ticked, else the middle one)."""
        dates = sorted(d for _, d in self.ticked_scans() if d != "?")
        if not hasattr(self, "ref_combo"):
            return
        self.ref_combo["values"] = dates
        if dates and self.ref_date.get() not in dates:
            self.ref_date.set(REFERENCE_DATE if REFERENCE_DATE in dates else dates[len(dates) // 2])

    def _update_run_hint(self):
        self._refresh_reference()
        # run name follows the ticked dates until the user types an own name
        if self.run_name.get() == getattr(self, "_last_auto_name", self.run_name.get()):
            self._last_auto_name = self._auto_run_name()
            self.run_name.set(self._last_auto_name)
        n = len(self.ticked_scans())
        if hasattr(self, "run_hint"):
            est = f" (about {max(1, round(2.1 * n))} min; less when soil results can be reused)" if n else ""
            self.run_hint.config(text=f"{n} of {len(self.scan_tree.get_children())} dates ticked{est}")

    def _autoload_datasets(self):
        if DATASET_DIR.is_dir():
            for p in sorted(DATASET_DIR.glob("*.la[sz]")):
                self._add_scan(p)
            n = len(self.scan_tree.get_children())
            if n:
                self.log(f"Found {n} scans in {DATASET_DIR.name}/ (remove the ones you do not want).")

    def add_scans(self):
        for f in filedialog.askopenfilenames(filetypes=[("LAS/LAZ", "*.las *.laz"), ("All", "*.*")],
                                             initialdir=str(DATASET_DIR)):
            self._add_scan(Path(f))

    def add_scan_folder(self):
        d = filedialog.askdirectory(initialdir=str(DATASET_DIR))
        if d:
            for p in sorted(Path(d).glob("*.la[sz]")):
                self._add_scan(p)

    def scan_dates(self) -> list[str]:
        return sorted({self.scan_tree.set(i, "date") for i in self.scan_tree.get_children()} - {"?"})

    def _refresh_dates(self):
        self._update_run_hint()

    # ------------------------------------------------------------------ ground truth
    def _toggle_gt(self):
        state = "normal" if self.use_gt.get() else "disabled"
        for w in self.gt_buttons:
            w.configure(state=state)

    def _add_gt(self, p: Path):
        if any(self.gt_tree.set(i, "name") == str(p) for i in self.gt_tree.get_children()):
            return
        if p.is_dir():
            kind = "folder (one file per organ)"
        elif p.suffix.lower() == ".bin":
            kind = "CloudCompare .bin"
        else:
            kind = f"single file ({p.suffix.lower()})"
        d = date_from_name(p.name)
        self.gt_tree.insert("", "end", values=(str(p), d or AUTO, kind, "not checked"))

    def add_gt_folder(self):
        d = filedialog.askdirectory(title="Ground-truth folder (one plant)", initialdir=str(SAMPLE_GT.parent))
        if d:
            self._add_gt(Path(d))

    def add_gt_files(self):
        files = filedialog.askopenfilenames(
            title="Ground-truth file(s) - one plant per file",
            filetypes=[("Point clouds", "*.las *.laz *.ply *.asc *.txt *.csv *.xyz *.bin"), ("All", "*.*")],
            initialdir=str(SAMPLE_GT.parent))
        for f in files:
            self._add_gt(Path(f))

    def gt_items(self, only_selected=False) -> list[tuple[str, str, str | None]]:
        ids = self.gt_tree.selection() if only_selected else self.gt_tree.get_children()
        out = []
        for i in ids:
            d = self.gt_tree.set(i, "date")
            out.append((i, self.gt_tree.set(i, "name"), None if d == AUTO else d))
        return out

    def _edit_gt_date(self, event):
        item = self.gt_tree.identify_row(event.y)
        col = self.gt_tree.identify_column(event.x)
        if not item or col != "#2":
            return
        x, y, w, h = self.gt_tree.bbox(item, col)
        cb = ttk.Combobox(self.gt_tree, values=[AUTO] + self.scan_dates(), width=10)
        cb.set(self.gt_tree.set(item, "date"))
        cb.place(x=x, y=y, width=w, height=h)
        cb.focus_set()

        def commit(_=None):
            v = cb.get().strip()
            if v != AUTO and not (len(v) == 6 and v.isdigit()):
                messagebox.showwarning("Date", "Use YYMMDD, e.g. 230621, or 'auto'.")
            else:
                self.gt_tree.set(item, "date", v)
            cb.destroy()

        cb.bind("<<ComboboxSelected>>", commit)
        cb.bind("<Return>", commit)
        cb.bind("<FocusOut>", commit)

    def validate_gt(self):
        items = self.gt_items(only_selected=True) or self.gt_items()
        if not items:
            messagebox.showinfo("Ground truth", "Add a ground-truth folder or file first.")
            return

        def work():
            from pipeline.ground_truth import load_ground_truth
            for iid, path, date in items:
                try:
                    gt = load_ground_truth(path, date, log=self.log)
                    self.log(gt.describe())
                    status = (f"OK - {len(gt.leaf_ids)} leaves"
                              + (", no stem" if gt.labelled and not gt.has_stem else "")
                              + (", unlabelled" if not gt.labelled else "")
                              + (f", {sum(gt.fragments.values())} leaf pieces merged" if gt.fragments else "")
                              + ("" if gt.utm_like else ", SHIFTED coords"))
                    self.log_q.put(lambda i=iid, s=status, d=gt.date: (
                        self.gt_tree.set(i, "status", s),
                        self.gt_tree.set(i, "date", d or AUTO) if self.gt_tree.set(i, "date") == AUTO else None))
                except Exception as e:
                    self.log(f"x {path}: {e}")
                    self.log_q.put(lambda i=iid, s=str(e): self.gt_tree.set(i, "status", "ERROR: " + s[:120]))

        self._in_thread(work, label="checking ground truth...")

    def show_gt_help(self):
        win = tk.Toplevel(self)
        win.title(GT_HELP_TITLE)
        win.geometry("1180x760")
        left = ttk.Frame(win, padding=8)
        left.pack(side="left", fill="both", expand=True)
        txt = ScrolledText(left, wrap="word", font=("Menlo", 11))
        txt.insert("1.0", GT_HELP_TEXT)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)
        right = ttk.Frame(win, padding=8)
        right.pack(side="right", fill="y")
        img_lbl = ttk.Label(right, text="rendering sample...")
        img_lbl.pack()

        def show_img():
            if SAMPLE_PNG.exists():
                win._img = tk.PhotoImage(file=str(SAMPLE_PNG))
                img_lbl.configure(image=win._img, text="")
            else:
                img_lbl.configure(text="(sample ground truth not found on this machine)")

        if SAMPLE_PNG.exists():
            show_img()
        else:
            def work():
                make_sample_image(SAMPLE_PNG, SAMPLE_GT)
            def done(_):
                if win.winfo_exists():
                    show_img()
            self._in_thread(work, done, label="rendering sample...")

    # ------------------------------------------------------------------ runs
    def run_pipeline(self):
        scans = self.ticked_scans()
        if not scans:
            messagebox.showinfo("Run", "Tick at least one date in section 1 (click the box in the 'Run' column).")
            return
        self._refresh_reference()
        if len(scans) == 1:
            self.log("Only one date: plants are taken from that date alone (no cross-date pooling).")
        if self.run_name.get() == getattr(self, "_last_auto_name", None):
            self._last_auto_name = self._auto_run_name()        # time of the run, ticked dates at the end
            self.run_name.set(self._last_auto_name)
        try:
            from pipeline import runner  # added with the processing stages
        except ImportError:
            messagebox.showinfo(
                "Not available yet",
                "The processing stages could not be loaded (see the log).")
            return
        root = Path(self.out_root.get()) / self.run_name.get()
        gt = [(p, d) for _, p, d in self.gt_items()] if self.use_gt.get() else []

        def work():
            return runner.run(scans, root, reference_date=self.ref_date.get(), gt_items=gt if stop == "all" else [],
                              log=self.log, progress=self.report_progress, stop_after=stop)

        stop = dict(RUN_STEPS)[self.stop_step.get()]
        self.log(f"Running {len(scans)} date(s): {', '.join(sorted(d for _, d in scans))} - up to step '{stop}'")

        self.last_run = root

        def done(_):
            self.viewer.set_run(root, stop)
            self.nb.select(self.viewer)
            if stop == "all":
                self.load_results(root)
                self.nb.select(self.viewer)

        self._in_thread(work, done, label="running pipeline...")

    def _latest_run(self) -> Path | None:
        root = Path(self.out_root.get())
        runs = [p for p in root.glob("*") if p.is_dir() and any(p.glob("*/03_plants/plants_index.csv"))]
        return max(runs, key=lambda p: p.stat().st_mtime) if runs else None

    def run_gt_on_existing(self):
        if not self.use_gt.get():
            self.use_gt.set(True)
            self._toggle_gt()
        items = self.gt_items()
        if not items:
            messagebox.showinfo("Ground truth", "Add at least one ground-truth folder or file first (section 2).")
            return
        latest = self._latest_run()
        d = filedialog.askdirectory(title="Run folder with predictions",
                                    initialdir=str(latest or self.out_root.get()))
        if not d:
            return
        run_root = Path(d)

        def work():
            from pipeline.ground_truth import run_ground_truth_check
            return run_ground_truth_check(run_root, [(p, dt) for _, p, dt in items], log=self.log,
                                          progress=self.report_progress)

        self.last_run = run_root
        self._in_thread(work, lambda _: self.load_results(run_root), label="checking ground truth...")

    # ------------------------------------------------------------------ results
    def load_results_dialog(self):
        d = filedialog.askdirectory(title="Run folder", initialdir=str(self._latest_run() or self.out_root.get()))
        if d:
            self.load_results(Path(d))

    def load_results(self, run_root: Path):
        self.last_run = run_root
        try:
            self.viewer.set_run(run_root)
        except Exception as e:  # noqa: BLE001
            self.log(f"Step viewer: {e}")
        self._load_plants(run_root)
        f = run_root / "ground_truth" / "gt_summary.csv"
        self.res_tree.delete(*self.res_tree.get_children())
        self.gt_rows = []
        if not f.exists():
            self.log(f"No ground-truth summary in {run_root} (plants table loaded)")
            self.nb.select(1)
            return
        with open(f) as fh:
            self.gt_rows = list(csv.DictReader(fh))

        def num(r, k, scale=1.0, fmt="{:.3f}"):
            try:
                v = float(r.get(k, "nan")) * scale
                return "" if v != v else fmt.format(v)
            except ValueError:
                return ""

        for r in self.gt_rows:
            self.res_tree.insert("", "end", values=(
                r.get("gt_name", ""), r.get("matched_date", ""), nice(r.get("matched_plant", "")) or "NOT FOUND",
                r.get("match_method", ""), num(r, "plant_iou"), num(r, "stem_iou"),
                num(r, "mean_leaf_iou_with_fragments"), num(r, "mean_iou"),
                f"{r.get('n_gt_leaves', '')}/{r.get('n_pred_leaves', '')}",
                num(r, "err_plant_height_m", 100, "{:.1f}"), num(r, "err_stem_length_m", 100, "{:.1f}"),
                num(r, "err_stem_radius_m", 1000, "{:.1f}")))
        self.nb.select(1)
        self.log(f"Loaded {len(self.gt_rows)} ground-truth results from {run_root.name}")

    def _selected_gt_dir(self) -> Path | None:
        sel = self.res_tree.selection()
        if not sel or self.last_run is None:
            return None
        name = self.res_tree.set(sel[0], "gt")
        return self.last_run / "ground_truth" / name

    def show_selected_3d(self):
        gdir = self._selected_gt_dir()
        if gdir is None or not (gdir / "viewer_data.npz").exists():
            return
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from ui.mpl_toolbar import HelpToolbar as NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from pipeline import reference_structure as ref

        d = np.load(gdir / "viewer_data.npz")
        c0 = d["gt_xyz"].mean(axis=0)
        lut = ref._COLOR_LUT
        fig = Figure(figsize=(10, 5))
        rng = np.random.default_rng(0)
        axes = []
        for k, (xyz, lab, ttl) in enumerate([
            (d["gt_xyz"], d["gt_labels"], f"Ground truth ({gdir.name})"),
            (d["pred_xyz"], d["pred_labels"], f"Prediction {d['pred_id']} (GT colours, grey = unmatched)"),
        ]):
            i = rng.choice(len(xyz), min(len(xyz), 15000), replace=False)
            cols = np.full((len(i), 3), 0.6)
            ok = (lab[i] >= 0) & (lab[i] < len(lut))
            cols[ok] = lut[lab[i][ok]]
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            p = xyz[i] - c0
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=cols, s=0.5)
            ax.set_title(ttl, fontsize=9)
            ax.set_box_aspect(np.ptp(d["gt_xyz"] - c0, axis=0) + 1e-3)
            axes.append(ax)

        def sync(_):  # rotate both views together
            a, b = axes
            src = a if a.button_pressed is not None else b
            dst = b if src is a else a
            dst.view_init(elev=src.elev, azim=src.azim)
            canvas.draw_idle()

        for w in self.view_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.view_frame)
        NavigationToolbar2Tk(canvas, self.view_frame).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.mpl_connect("motion_notify_event", sync)
        canvas.draw()
        self.canvas = canvas

    def show_leaf_details(self):
        gdir = self._selected_gt_dir()
        if gdir is None:
            messagebox.showinfo("Details", "Select a row first.")
            return
        win = tk.Toplevel(self)
        win.title(f"Details - {gdir.name}")
        win.geometry("1100x650")
        for title, fname in (("IoU per organ", "iou_per_class.csv"), ("Plant traits", "trait_comparison_plant.csv"),
                             ("Leaf traits (matched leaves)", "trait_comparison_leaf.csv")):
            f = gdir / fname
            fr = ttk.LabelFrame(win, text=f" {title} ", padding=4)
            fr.pack(fill="both", expand=True, padx=6, pady=3)
            if not f.exists() or f.stat().st_size == 0:
                ttk.Label(fr, text="(none)").pack()
                continue
            with open(f) as fh:
                rows = list(csv.reader(fh))
            t = self._tree(fr, tuple(rows[0]), tuple(110 for _ in rows[0]), height=6)
            for r in rows[1:]:
                t.insert("", "end", values=r)

    def show_comparison_png(self):
        gdir = self._selected_gt_dir()
        if gdir is None or not (gdir / "comparison.png").exists():
            messagebox.showinfo("Picture", "Select a row with a matched plant first.")
            return
        win = tk.Toplevel(self)
        win.title(f"Comparison - {gdir.name}")
        win._img = tk.PhotoImage(file=str(gdir / "comparison.png"))
        ttk.Label(win, image=win._img).pack()

    def open_pyvista(self):
        gdir = self._selected_gt_dir()
        if gdir is None or not (gdir / "viewer_data.npz").exists():
            messagebox.showinfo("3D", "Select a row with a matched plant first.")
            return
        subprocess.Popen([sys.executable, str(CODE_DIR / "ui" / "viewer3d.py"), str(gdir / "viewer_data.npz")])

    # plants table -------------------------------------------------------------
    def _load_plants(self, run_root: Path):
        av = run_root / "tracking" / "availability.csv"
        tr = run_root / "tracking" / "traits_by_date.csv"
        self.plant_tree.delete(*self.plant_tree.get_children())
        self.plant_traits = {}
        if not av.exists():
            return
        with open(av) as fh:
            rows = list(csv.DictReader(fh))
        if tr.exists():
            with open(tr) as fh:
                for r in csv.DictReader(fh):
                    self.plant_traits.setdefault(r["plant_id"], {})[r["date"]] = r
        date_cols = [c for c in rows[0] if c not in ("plant_id", "group", "missing_on", "n_dates_present", "n_dates_measured")]
        cols = ("plant", *[f"d{i}" for i in range(len(date_cols))], "height", "leaves", "stage")
        heads = ("Plant instance", *date_cols, "Height m (last)", "Leaves (last)", "Stage est.")
        self.plant_tree.configure(columns=cols)
        for c, h in zip(cols, heads):
            self.plant_tree.heading(c, text=h)
            self.plant_tree.column(c, width=230 if c == "plant" else 95, anchor="w")
        for r in rows:
            last = {}
            for d, t in sorted(self.plant_traits.get(r["plant_id"], {}).items()):
                if t.get("present") == "True" and t.get("status") == "ok":
                    last = t
            def f(v, n=2):
                try:
                    return f"{float(v):.{n}f}"
                except (TypeError, ValueError):
                    return ""
            short = {"present": "yes", "present (stem hidden)": "yes (stem hidden)", "missing": "MISSING",
                     "not visible (covered by canopy)": "covered", "too small to measure": "too small"}
            self.plant_tree.insert("", "end", iid=r["plant_id"], values=(
                nice(r["plant_id"]), *[short.get(r[c], r[c]) for c in date_cols],
                f(last.get("plant_height_m")), f(last.get("n_leaves"), 0), last.get("v_stage_estimate", "")))
        self.log(f"Loaded {len(rows)} plant instances from {run_root.name}")

    def show_plant_3d(self, date: str | None = None):
        """The plant in the structure viewer of the user's script (same view as the Step viewer),
        with a choice of the dates on which it was measured."""
        sel = self.plant_tree.selection()
        if not sel or not self.last_run:
            return
        pid = sel[0]
        files = sorted(self.last_run.glob(f"*/03_plants/{pid.split('_')[0]}/{pid}/{pid}_labelled.npz"))
        if not files:
            self.log(f"No structure saved for {nice(pid)}")
            return
        dates = [f.parts[-5] for f in files]
        date = date if date in dates else dates[-1]
        for w in self.view_frame.winfo_children():
            w.destroy()
        top = ttk.Frame(self.view_frame)
        top.pack(fill="x")
        ttk.Label(top, text=f"{nice(pid)}   date:").pack(side="left")
        dv = tk.StringVar(value=date)
        cb = ttk.Combobox(top, textvariable=dv, values=dates, width=10, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _: self.show_plant_3d(dv.get()))
        t = self.plant_traits.get(pid, {}).get(date, {})
        try:
            ttk.Label(top, text=f"   height {float(t.get('plant_height_m')):.2f} m, {float(t.get('n_leaves')):.0f} leaves, "
                                f"leaf area {float(t.get('total_leaf_area_m2')) * 1e4:.0f} cm\u00b2").pack(side="left")
        except (TypeError, ValueError):
            pass
        from ui.user_structure_view import UserStructureView, layer_vars
        if getattr(self.viewer, "_struct_vis", None) is None:
            self.viewer._struct_vis = layer_vars()          # shared with the Step viewer
        f = files[dates.index(date)]
        UserStructureView(self.view_frame, f, f"{nice(pid)}  {date}", log=self.log,
                          vis=self.viewer._struct_vis).pack(fill="both", expand=True)

    def open_report(self):
        """Rebuild the report from the run's files (adds the summary / leaf width to older runs), then open it."""
        run = self.last_run
        if not run or not (run / "run_config.json").exists():
            messagebox.showinfo("Report", "No run loaded yet.")
            return
        try:
            import json
            from pipeline.report import write_report
            cfg = json.loads((run / "run_config.json").read_text())
            dates = sorted(d for _, d in cfg.get("scans", []))
            write_report(run, dates, cfg.get("reference_date", ""), None)
        except Exception as e:  # noqa: BLE001
            self.log(f"Report could not be rebuilt ({e}); opening the saved one.")
        if (run / "report.html").exists():
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(run / "report.html")])
        else:
            messagebox.showinfo("Report", "No report.html in the current run yet.")

    def open_run_folder(self):
        if self.last_run and self.last_run.exists():
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(self.last_run)])


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
