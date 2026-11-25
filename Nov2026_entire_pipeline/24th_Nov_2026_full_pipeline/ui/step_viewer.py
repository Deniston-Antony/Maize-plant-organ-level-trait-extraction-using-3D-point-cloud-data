"""Step viewer: look at the output of every pipeline step inside the app.

  1 Soil & plants      soil/plant split per date (figure, numbers, 3D of the plant points)
  2 Alignment          shift of each date onto the reference date
  3 Stems & plant clusters   stems found, plant clusters along the row
  4 Plants             every plant separated (one colour per plant), status per plant
  5 Plant structure    one plant: stem, individual leaves, stem axis, leaf midribs, all traits
  6 Growth             availability per plant and date, growth curves
  7 Compare dates      the same plants on several dates, overlapped
  8 Cross-date matching  stage 9: a plant cluster, or one plant's stem and leaves
                       (coloured by leaf ID), on every date + the matching tables

Figures are the PNGs the pipeline writes; 3D views are drawn from the saved
point files (matplotlib, rotate with the mouse).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

from pipeline.names import cluster, instance, nice, raw

STEP_NAMES = [("soil", "1  Soil & plants"), ("alignment", "2  Alignment"), ("stems", "3  Stems & plant clusters"),
              ("plants", "4  Plants"), ("structure", "5  Plant structure"), ("growth", "6  Growth"),
              ("compare", "7  Compare dates"), ("cross", "8  Cross-date matching")]


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _fmt(v, scale=1.0, nd=2):
    try:
        x = float(v) * scale
        return "" if x != x else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return "" if v is None else str(v)


class ImageView(ttk.Frame):
    """Scrollable picture. Zoom: two fingers up/down on the trackpad (or Ctrl + wheel);
    double-click = fit to width. The zoom (% of the picture's own size) is shown top right."""

    def __init__(self, master):
        super().__init__(master)
        self.cv = tk.Canvas(self, background="#ffffff", highlightthickness=0)
        ys = ttk.Scrollbar(self, orient="vertical", command=self.cv.yview)
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.cv.xview)
        self.cv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.cv.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.img = None
        self.tkimg = None
        self.scale = None            # None = fit to width
        self.cv.bind("<Configure>", lambda _: self._draw())
        self.cv.bind("<Double-Button-1>", self._fit)
        self.cv.bind("<MouseWheel>", lambda e: self.cv.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.cv.bind("<Control-MouseWheel>", lambda e: self._zoom(1.15 if e.delta > 0 else 1 / 1.15))
        try:
            self.cv.bind("<TouchpadScroll>", self._touchpad)
        except tk.TclError:
            pass
        self.cv.bind("<ButtonPress-1>", lambda e: self.cv.scan_mark(e.x, e.y))
        self.cv.bind("<B1-Motion>", lambda e: self.cv.scan_dragto(e.x, e.y, gain=1))
        self._pending = None

    def show(self, path: Path | None, missing_text: str = "No figure for this step yet - run the pipeline up to this step."):
        from PIL import Image
        self.img = Image.open(path) if path and Path(path).exists() else None
        self.missing = missing_text
        self.scale = None
        self._draw()

    def _fit(self, _=None):
        self.scale = None
        self._draw()

    def _fit_scale(self) -> float:
        w = max(200, self.cv.winfo_width())
        return min(1.0, w / self.img.width) if self.img is not None else 1.0

    def _zoom(self, f):
        if self.img is None:
            return
        self.scale = min(4.0, max(0.05, (self.scale or self._fit_scale()) * f))
        if self._pending is None:          # several trackpad steps -> one redraw
            self._pending = self.after(30, self._draw)

    def _touchpad(self, e):
        try:
            dx, dy = (float(v) for v in self.tk.splitlist(self.tk.call("tk::PreciseScrollDeltas", e.delta)))
        except (tk.TclError, ValueError):
            return
        if dy:
            self._zoom(min(1.2, max(0.83, 1.0 + 0.02 * dy)))

    def _draw(self):
        from PIL import Image, ImageTk
        self._pending = None
        self.cv.delete("all")
        if self.img is None:
            self.cv.create_text(20, 20, anchor="nw", text=getattr(self, "missing", ""), fill="#666", font=("Helvetica", 13),
                                width=max(200, self.cv.winfo_width() - 40))
            return
        sc = self.scale or self._fit_scale()
        im = self.img
        if abs(sc - 1.0) > 1e-3:
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(im)
        self.cv.create_image(0, 0, anchor="nw", image=self.tkimg)
        self.cv.configure(scrollregion=(0, 0, im.width, im.height))
        w = max(200, self.cv.winfo_width())
        x0, y0 = self.cv.canvasx(w - 8), self.cv.canvasy(8)
        self.cv.create_text(x0, y0, anchor="ne", fill="#444", font=("Helvetica", 12, "bold"),
                            text=f"zoom {sc * 100:.0f} %")
        self.cv.create_text(x0, y0 + 18, anchor="ne", fill="#888", font=("Helvetica", 10),
                            text="two fingers: zoom   drag: move   double-click: fit")


class StepViewer(ttk.Frame):
    def __init__(self, master, log=print):
        super().__init__(master, padding=6)
        self.log = log
        self.run_root: Path | None = None
        self.dates: list[str] = []
        self.step = tk.StringVar(value="soil")
        self.date = tk.StringVar()
        self.trait = tk.StringVar(value="plant_height_m")
        self.show_points = tk.BooleanVar(value=True)
        self.canvas = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="Open run...", command=self.open_dialog).pack(side="left")
        self.run_lbl = ttk.Label(top, text="no run loaded", foreground="#666")
        self.run_lbl.pack(side="left", padx=8)
        ttk.Label(top, text="Date:").pack(side="left", padx=(16, 2))
        self.date_cb = ttk.Combobox(top, textvariable=self.date, width=10, state="readonly")
        self.date_cb.pack(side="left")
        self.date_cb.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        ttk.Button(top, text="Open folder of this step", command=self.open_folder).pack(side="right")

        steps = ttk.Frame(self)
        steps.pack(fill="x", pady=6)
        for key, name in STEP_NAMES:
            ttk.Radiobutton(steps, text=name, value=key, variable=self.step, command=self.refresh,
                            style="Toolbutton").pack(side="left", padx=2)

        self.pw = ttk.PanedWindow(self, orient="horizontal")
        self.pw.pack(fill="both", expand=True)
        self.left = ttk.Frame(self.pw, width=430)
        self.right = ttk.Frame(self.pw)
        self.pw.add(self.left, weight=1)
        self.pw.add(self.right, weight=3)
        self.image = ImageView(self.right)

    # ------------------------------------------------------------------ run / navigation
    def open_dialog(self):
        d = filedialog.askdirectory(title="Run folder")
        if d:
            self.set_run(Path(d))

    def set_run(self, root: Path, step: str | None = None):
        self.run_root = Path(root)
        cfg = self.run_root / "run_config.json"
        if cfg.exists():
            self.dates = [d for _, d in json.loads(cfg.read_text())["scans"]]
            self.ref = json.loads(cfg.read_text()).get("reference_date", "")
        else:
            self.dates = sorted(p.name for p in self.run_root.iterdir() if p.is_dir() and p.name.isdigit())
            self.ref = ""
        self.date_cb["values"] = self.dates
        if self.dates and self.date.get() not in self.dates:
            self.date.set(self.ref if self.ref in self.dates else self.dates[0])
        self.run_lbl.config(text=f"{self.run_root.name}  ({len(self.dates)} dates, reference {self.ref})")
        if step:
            self.step.set({"all": "structure"}.get(step, step))
        self.refresh()

    def open_folder(self):
        if not self.run_root:
            return
        d = self.date.get()
        sub = {"soil": f"{d}/01_soil", "alignment": "alignment", "stems": f"{d}/02_stems_groups",
               "plants": f"{d}/03_plants", "structure": f"{d}/03_plants", "growth": "growth",
               "compare": "tracking", "cross": "cross_date_matching"}[self.step.get()]
        p = self.run_root / sub
        if p.exists():
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])

    def _clear_right(self):
        """Empty the right panel but keep the shared image widget."""
        for w in self.right.winfo_children():
            if w is not self.image:
                w.destroy()
        self.image.pack_forget()

    def _clear(self):
        if str(self.left) not in self.pw.panes():          # step 7 hides the left panel
            self.pw.insert(0, self.left, weight=1)
        for w in self.left.winfo_children():
            w.destroy()
        for w in self.right.winfo_children():
            if w is not self.image:
                w.destroy()
        self.image.pack_forget()

    def _table(self, parent, cols, heads, widths, rows, height=10):
        f = ttk.Frame(parent)
        f.pack(fill="both", expand=True, pady=2)
        t = ttk.Treeview(f, columns=cols, show="headings", height=height)
        for c, h, w in zip(cols, heads, widths):
            t.heading(c, text=h)
            t.column(c, width=w, anchor="w")
        for r in rows:
            t.insert("", "end", values=r)
        sb = ttk.Scrollbar(f, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return t

    def _kv(self, parent, title, pairs):
        lf = ttk.LabelFrame(parent, text=f" {title} ", padding=4)
        lf.pack(fill="x", pady=3)
        for k, v in pairs:
            r = ttk.Frame(lf)
            r.pack(fill="x")
            ttk.Label(r, text=k, width=30, foreground="#555").pack(side="left")
            ttk.Label(r, text=v, font=("Menlo", 11)).pack(side="left")

    def refresh(self):
        if not self.run_root:
            return
        self._clear()
        getattr(self, f"_step_{self.step.get()}")()

    def _show_image(self, path: Path):
        self.image.pack(fill="both", expand=True)
        self.image.show(path)

    # ------------------------------------------------------------------ 1 soil
    def _step_soil(self):
        d = self.date.get()
        sd = self.run_root / d / "01_soil"
        js = json.loads((sd / "soil_summary.json").read_text()) if (sd / "soil_summary.json").exists() else {}
        self._kv(self.left, f"Soil and plants {d}", [
            ("points in the scan", f"{js.get('n_points', 0):,}"),
            ("plant points", f"{js.get('n_plant', 0):,}  ({_fmt(js.get('plant_share'), 100, 1)} %)"),
            (f"ground outliers removed (band {_fmt(js.get('ground_band_height_m'), 100, 0)} cm)",
             f"{js.get('n_ground_outliers_removed', 0):,}"),
            ("  below the ground", f"{js.get('n_removed_below_ground', 0):,}"),
            ("  isolated points near the ground", f"{js.get('n_removed_isolated_in_band', 0):,}"),
            ("  low blobs (clods, debris)", f"{js.get('n_low_blobs_removed', 0):,} blobs, "
                                            f"{js.get('n_removed_low_blob_points', 0):,} points"),
            ("ground hidden under plants", f"{_fmt(js.get('dtm_cells_under_plants_filled_share'), 100, 0)} % of cells"),
            ("hidden ground taken from", js.get("ground_borrowed_from", "-")),
            ("points moved plant -> soil by that", f"{js.get('plant_points_removed_by_borrow', 0):,}"),
            ("time", f"{js.get('seconds', '')} s"),
        ])
        b = ttk.Frame(self.left)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="3D view (rotate)", command=lambda: self._soil_3d()).pack(side="left")
        ttk.Button(b, text="Check figure", command=lambda: self._soil_figure()).pack(side="left", padx=4)
        ttk.Button(b, text="Open plants PLY in CloudCompare", command=lambda: self._cloudcompare(sd / f"{d}_plants.ply")).pack(side="left", padx=4)
        ttk.Button(b, text="Soil PLY", command=lambda: self._cloudcompare(sd / f"{d}_soil.ply")).pack(side="left", padx=4)
        ttk.Button(b, text="Removed outliers PLY", command=lambda: self._cloudcompare(sd / f"{d}_ground_outliers.ply")).pack(side="left", padx=4)
        rows = [(r.get("date"), f"{int(float(r.get('n_plant', 0))):,}", _fmt(r.get("plant_share"), 100, 1),
                 _fmt(r.get("dtm_cells_under_plants_filled_share"), 100, 0))
                for r in _read_csv(self.run_root / "soil_summary.csv")]
        ttk.Label(self.left, text="All dates of this run:").pack(anchor="w", pady=(8, 0))
        self._table(self.left, ("d", "n", "s", "h"), ("Date", "Plant points", "Plant %", "Ground hidden %"), (80, 110, 70, 110), rows, 6)
        self._soil_3d()

    def _soil_3d(self):
        """Rotatable 3D view of the separation: plant / soil / removed points."""
        from ui.soil_view import SoilView
        d = self.date.get()
        sd = self.run_root / d / "01_soil"
        self._clear_right()
        if not (sd / "plants.npz").exists():
            ttk.Label(self.right, text="No soil result for this date.").pack()
            return
        SoilView(self.right, sd, d, f"{d}  soil / plant separation", log=self.log).pack(fill="both", expand=True)

    def _soil_figure(self):
        d = self.date.get()
        self._clear_right()
        self._show_image(self.run_root / d / "01_soil" / "soil_check.png")

    def _cloudcompare(self, path: Path):
        app = Path("/Applications/CloudCompare.app")
        if path.exists() and app.exists():
            subprocess.Popen(["open", "-a", str(app), str(path)])
        else:
            self.log(f"Cannot open {path.name} (file or CloudCompare missing)")

    # ------------------------------------------------------------------ 2 alignment
    def _step_alignment(self):
        rows = _read_csv(self.run_root / "alignment" / "alignment_summary.csv")
        ttk.Label(self.left, wraplength=400, text=(
            "How far each scan had to be moved to sit on the reference date "
            f"({self.ref}). Click a date to see its check figure on the right.")).pack(anchor="w", pady=(0, 4))
        from pipeline.tolerance import alignment_tolerance
        tol = alignment_tolerance(self.run_root)

        def pm(v):
            return f"\u00b1{v:.1f}" if v == v else "not known"
        t = self._table(self.left, ("d", "x", "y", "z", "txy", "tz", "c"),
                        ("Date", "Shift x cm", "Shift y cm", "Shift z cm", "Dev./err. x/y cm", "Dev./err. z cm", "Confidence"),
                        (70, 70, 70, 70, 100, 90, 80),
                        [(r["date"], r["dx_cm"], r["dy_cm"], r["dz_cm"], pm(tol.get(r["date"], {}).get("xy_cm", float("nan"))),
                          pm(tol.get(r["date"], {}).get("z_cm", float("nan"))), r["confidence"]) for r in rows], 6)
        for iid in t.get_children():                       # highlight the date shown on the right
            if t.item(iid, "values")[0] == self.date.get():
                t.selection_set(iid)
        t.bind("<<TreeviewSelect>>", lambda e: self._pick_date(t.item(t.focus(), "values")[0] if t.focus() else None))
        ttk.Label(self.left, foreground="#888", wraplength=400, text=(
            "x / y = sideways shift, z = height shift. Dev./err. = deviation / error (\u00b1): how much the 1.5 m row sections "
            "disagree about the shift (x/y) and about the soil height after the shift (z). Confidence: high = the soil and stem bases agree; "
            "medium = most parts agree; low = not reliable (e.g. 01.06: seedlings and bare soil give little to match), "
            "plant instances of that date may be matched to the wrong neighbour.")).pack(anchor="w", pady=(4, 0))
        r = next((r for r in rows if r["date"] == self.date.get()), None)
        if r:
            msg = tk.Message(self.left, text=f"How: {r['method']}\n\n{r['notes']}", width=400)
            msg.pack(fill="x", pady=6)
        self.image.pack(fill="both", expand=True)
        self.image.show(self.run_root / "alignment" / f"align_{self.date.get()}.png",
                        f"{self.date.get()} is the reference date: the other dates are moved onto it. "
                        "Choose another date to see its alignment." if self.date.get() == self.ref
                        else "No alignment figure yet - run the pipeline up to step 2.")

    def _pick_date(self, d):
        if d and d != self.date.get() and d in self.dates:
            self.date.set(d)
            self.refresh()

    # ------------------------------------------------------------------ 3 stems & groups
    def _step_stems(self):
        d = self.date.get()
        sd = self.run_root / d / "02_stems_groups"
        js = json.loads((sd / "stems_summary.json").read_text()) if (sd / "stems_summary.json").exists() else {}
        self._kv(self.left, f"Stems {d}", [
            ("stems found", str(js.get("n_stems_total", ""))),
            ("recovered at reference positions", str(js.get("n_guided", ""))),
            ("off the row (weeds, leaf tips)", str(js.get("n_off_row", ""))),
            ("low blobs removed (clods)", f"{js.get('n_blobs_removed', '')}  ({_fmt(js.get('points_removed_share'), 100, 1)} % of points)"),
            ("plant clusters (whole run)", str(js.get("n_groups", ""))),
        ])
        rows = [(cluster(r["group"]), r.get("n_plants", r.get("n_stems", "")), r.get("flag", "")) for r in _read_csv(self.run_root / "groups_reference.csv")]
        ttk.Label(self.left, text="Plant clusters (named from the reference date, south to north):").pack(anchor="w", pady=(6, 0))
        tg = self._table(self.left, ("g", "n", "f"), ("Plant cluster", "Plant instances", "Check"), (120, 100, 200), rows, 14)
        b = ttk.Frame(self.left)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="Stems figure", command=lambda: self._figure(sd / "stems_check.png")).pack(side="left")
        ttk.Button(b, text="3D view of the plant clusters", command=lambda: self._plants_3d()).pack(side="left", padx=4)
        ttk.Label(self.left, text="Click a plant cluster to see it in 3D (needs the plants step).", foreground="#888").pack(anchor="w")
        tg.bind("<<TreeviewSelect>>", lambda e: self._group_pick(raw(tg.item(tg.focus(), "values")[0]) if tg.focus() else None))
        self._show_image(sd / "stems_check.png")

    def _group_pick(self, group):
        if not group:
            return
        v = getattr(self, "_plants_view", None)
        if v is not None and v.winfo_exists():
            v.show_group(group)
        elif (self.run_root / self.date.get() / "03_plants" / "plants_index.csv").exists():
            self._plants_3d(group)

    # ------------------------------------------------------------------ 4 plants
    def _plant_rows(self, d):
        return {r["plant_id"]: r for r in _read_csv(self.run_root / d / "03_plants" / "plants_traits.csv")}

    def _step_plants(self):
        d = self.date.get()
        pr = self._plant_rows(d)
        rows = []
        for pid, r in sorted(pr.items()):
            st = "ok" if r.get("status") == "ok" else r.get("presence_visibility", r.get("status", ""))
            rows.append((nice(pid), st, _fmt(r.get("plant_height_m")), _fmt(r.get("n_leaves"), 1, 0)))
        ttk.Label(self.left, text=f"Plant instances on {d} (double-click one for its structure):").pack(anchor="w")
        t = self._table(self.left, ("p", "s", "h", "l"), ("Plant instance", "Status", "Height m", "Leaves"), (230, 120, 70, 60), rows, 24)
        t.bind("<Double-1>", lambda e: self._goto_structure(raw(t.item(t.focus(), "values")[0]) if t.focus() else None))
        t.bind("<<TreeviewSelect>>", lambda e: self._plants_pick(raw(t.item(t.focus(), "values")[0]) if t.focus() else None))
        b = ttk.Frame(self.left)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="3D view of plant instances and clusters", command=self._plants_3d).pack(side="left")
        ttk.Button(b, text="Check figure", command=lambda: self._figure(self.run_root / d / "03_plants" / "plants_check.png")).pack(side="left", padx=4)
        ttk.Label(self.left, text="Click a plant instance: its plant cluster is shown in 3D.", foreground="#888").pack(anchor="w")
        self._plants_3d()

    def _plants_3d(self, group=None):
        """Rotatable 3D view of the separated plants, coloured by plant or group."""
        from ui.plants_view import PlantsView
        d = self.date.get()
        self._clear_right()
        self._plants_view = PlantsView(self.right, self.run_root / d, d, f"{d}  plant instances", log=self.log, group=group)
        self._plants_view.pack(fill="both", expand=True)

    def _plants_pick(self, pid):
        v = getattr(self, "_plants_view", None)
        if pid and v is not None and v.winfo_exists():
            v.show_group(pid.split("_")[0])

    def _figure(self, path):
        self._clear_right()
        self._plants_view = None
        self._show_image(path)

    def _goto_structure(self, pid):
        if pid:
            self._want_plant = pid
            self.step.set("structure")
            self.refresh()

    # ------------------------------------------------------------------ 5 plant structure
    def _step_structure(self):
        d = self.date.get()
        pr = self._plant_rows(d)
        ok = sorted(p for p, r in pr.items() if r.get("status") == "ok")
        top = ttk.Frame(self.left)
        top.pack(fill="x")
        ttk.Label(top, text=f"Plant instances with structure on {d}: {len(ok)}").pack(side="left")
        ttk.Button(top, text="Open in a separate window",
                   command=self._open_user_viewer).pack(side="right", padx=4)
        nav = ttk.Frame(self.left)
        nav.pack(fill="x", pady=(2, 2))
        ttk.Button(nav, text="< Previous plant instance", command=lambda: self._step_plant(-1)).pack(side="left")
        ttk.Button(nav, text="Next plant instance >", command=lambda: self._step_plant(1)).pack(side="left", padx=4)
        ttk.Label(nav, text="(or click one / use the arrow keys)", foreground="#888").pack(side="left")
        holder = ttk.Frame(self.left)
        holder.pack(fill="x")
        tree = ttk.Treeview(holder, columns=("g", "p", "n", "h"), show="headings", height=8, selectmode="browse")
        for c, h_, w_ in (("g", "Plant cluster", 120), ("p", "Plant instance", 130), ("n", "Leaves", 60), ("h", "Height m", 70)):
            tree.heading(c, text=h_)
            tree.column(c, width=w_, anchor="w")
        sb = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        for p in ok:
            r = pr[p]
            tree.insert("", "end", iid=p, values=(cluster(p.split("_")[0]), instance(p), _fmt(r.get("n_leaves"), 1, 0),
                                                   _fmt(r.get("plant_height_m"))))
        self._plant_list, self._plant_tree = ok, tree
        tree.bind("<<TreeviewSelect>>", lambda _: self._draw_plant(tree.selection()[0] if tree.selection() and tree.selection()[0] in ok else None))
        self.info = ttk.Frame(self.left)
        self.info.pack(fill="both", expand=True)
        want = getattr(self, "_want_plant", None) or (ok[0] if ok else None)
        self._want_plant = None
        if want in ok:
            tree.selection_set(want)
            tree.see(want)
        tree.focus_set()

    def _step_plant(self, k):
        ok = getattr(self, "_plant_list", [])
        if not ok:
            return
        cur = getattr(self, "_cur_plant", None)
        i = (ok.index(cur) + k) % len(ok) if cur in ok else 0
        self._plant_tree.selection_set(ok[i])
        self._plant_tree.focus(ok[i])
        self._plant_tree.see(ok[i])

    def _open_user_viewer(self):
        """The plant in the PyVista viewer of the user's structure script (separate window)."""
        pid = getattr(self, "_cur_plant", None)
        if not pid:
            self.log("Select a plant instance first.")
            return
        script = Path(__file__).resolve().parent.parent / "show_in_my_viewer.py"
        subprocess.Popen([sys.executable, str(script), str(self.run_root), self.date.get(), pid])
        self.log(f"Opening {nice(pid)} in the structure viewer of your script (keys 1-8 toggle layers) ...")

    def _draw_plant(self, pid):
        if not pid:
            return
        self._cur_plant = pid
        d = self.date.get()
        pdir = self.run_root / d / "03_plants" / pid.split("_")[0] / pid
        z = np.load(pdir / f"{pid}_labelled.npz")
        xyz, lab, base = z["xyz"], z["labels"], z["base"]
        tj = json.loads((pdir / f"{pid}_traits.json").read_text())
        traits, leaves = tj["traits"], tj["leaves"]
        for w in self.info.winfo_children():
            w.destroy()
        self._kv(self.info, f"{nice(pid)} on {d}", [
            ("plant height", f"{_fmt(traits.get('plant_height_m'))} m"),
            ("stem length (to top of whorl)", f"{_fmt(traits.get('stem_length_m'))} m"),
            ("stem diameter (median)", f"{_fmt(traits.get('stem_diameter_mm'), 1, 1)} mm"),
            ("stem found as", str(traits.get("stem_status", ""))),
            ("visible leaves", _fmt(traits.get("n_leaves"), 1, 0)),
            ("total leaf area", f"{_fmt(traits.get('total_leaf_area_m2'), 1e4, 0)} cm²"),
            ("mean leaf length", f"{_fmt(traits.get('mean_leaf_length_m'))} m"),
            ("mean leaf angle to stem", f"{_fmt(traits.get('mean_leaf_angle_to_stem_deg'), 1, 0)} °"),
        ])
        rows = [(lf["leaf_label"], _fmt(lf.get("leaf_length_m")), _fmt(lf.get("leaf_max_width_m"), 100, 1),
                 _fmt(lf.get("leaf_area_m2"), 1e4, 0), _fmt(lf.get("leaf_angle_to_stem_deg"), 1, 0),
                 _fmt(lf.get("insertion_height_m"))) for lf in sorted(leaves, key=lambda l_: l_.get("insertion_height_m", 0))]
        ttk.Label(self.info, text="Individual leaves (bottom to top):").pack(anchor="w")
        self._table(self.info, ("n", "l", "w", "a", "g", "h"), ("Leaf", "Length m", "Width cm", "Area cm²", "Angle °", "Attached at m"),
                    (45, 70, 70, 70, 60, 90), rows, 8)
        # 3D: the PyVista scene of the user's structure script, inside the app
        from ui.user_structure_view import UserStructureView
        self._clear_right()
        from ui.user_structure_view import layer_vars
        if getattr(self, "_struct_vis", None) is None:
            self._struct_vis = layer_vars()          # kept from plant to plant
        UserStructureView(self.right, pdir / f"{pid}_labelled.npz", f"{nice(pid)}  {d}", log=self.log,
                          vis=self._struct_vis).pack(fill="both", expand=True)
        from ui.user_structure_view import prefetch
        ok = getattr(self, "_plant_list", [])
        if pid in ok:
            i = ok.index(pid)
            nxt = [ok[(i + k) % len(ok)] for k in (1, 2, -1)]
            prefetch([self.run_root / d / "03_plants" / q.split("_")[0] / q / f"{q}_labelled.npz" for q in nxt])
        ttk.Label(self.info, foreground="#555", text="3D = your structure viewer (your script's layers and colours). "
                  "Drag to rotate, wheel to zoom; boxes 1-8 = your keys 1-8.", wraplength=400).pack(anchor="w")

    # ------------------------------------------------------------------ 7 compare dates
    def _step_compare(self):
        """Overlapped plants of several dates + one linked 3D view per date (whole top area)."""
        from ui.compare_view import CompareDates
        self.pw.forget(self.left)
        self._clear_right()
        CompareDates(self.right, self.run_root, self.dates, getattr(self, "ref", ""), log=self.log).pack(
            fill="both", expand=True)

    # ------------------------------------------------------------------ 8 cross-date matching (stage 9)
    def _step_cross(self):
        """A plant cluster or one plant (stem + leaves by leaf ID) on every date, linked 3D views + table."""
        from ui.cross_date_view import CrossDateView
        self.pw.forget(self.left)
        self._clear_right()
        CrossDateView(self.right, self.run_root, self.dates, getattr(self, "ref", ""), log=self.log).pack(
            fill="both", expand=True)

    # ------------------------------------------------------------------ 6 growth
    def _step_growth(self):
        av = _read_csv(self.run_root / "tracking" / "availability.csv")
        if not av:
            ttk.Label(self.left, text="No tracking yet - run the pipeline up to the last step.").pack(anchor="w")
            self._show_image(None)
            return
        dcols = [c for c in av[0] if c not in ("plant_id", "group", "missing_on", "n_dates_present", "n_dates_measured")]
        short = {"present": "yes", "present (stem hidden)": "yes*", "not visible (covered by canopy)": "covered",
                 "too small to measure": "small", "missing": "MISSING", "present (no structure)": "yes (no struct.)"}
        rows = [(nice(r["plant_id"]), *[short.get(r[c], r[c]) for c in dcols]) for r in av]
        ttk.Label(self.left, text="Availability (yes* = stem hidden; double-click: growth of that plant instance)").pack(anchor="w")
        t = self._table(self.left, ("p", *[f"d{i}" for i in range(len(dcols))]), ("Plant instance", *[c[:5] for c in dcols]),
                        (230, *[62] * len(dcols)), rows, 18)
        t.bind("<Double-1>", lambda e: self._plant_growth(raw(t.item(t.focus(), "values")[0]) if t.focus() else None))
        r = ttk.Frame(self.left)
        r.pack(fill="x", pady=4)
        ttk.Label(r, text="Figure:").pack(side="left")
        traits = ["plant_height_m", "stem_length_m", "stem_diameter_mm", "n_leaves", "total_leaf_area_m2",
                  "mean_leaf_length_m", "mean_leaf_angle_to_stem_deg"]
        cb = ttk.Combobox(r, textvariable=self.trait, values=traits, width=28, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _: self._growth_figure())
        ttk.Label(self.left, foreground="#888", wraplength=400,
                  text="The figure is interactive: the toolbar under it zooms / moves the plots, and its "
                       "'Configure subplots' button (sliders icon) opens the spacing tool with a ? for every setting.").pack(anchor="w")
        self._growth_figure()

    def _growth_figure(self):
        """Growth of every plant, one panel per group, drawn live (toolbar with the explained subplot tool)."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from pipeline.tracking import GROWTH_TRAITS
        from ui.mpl_toolbar import HelpToolbar
        key = self.trait.get()
        rows = _read_csv(self.run_root / "growth" / f"{key}.csv")
        self._clear_right()
        if not rows:
            self._show_image(self.run_root / "growth" / f"{key}.png")
            return
        label = next((lb for k, lb, _ in GROWTH_TRAITS if k == key), key)
        dates = [c for c in rows[0] if c not in ("plant_id", "group")]
        groups = sorted({r["group"] for r in rows})
        nc = min(5, max(1, len(groups)))
        nr = -(-len(groups) // nc)
        fig = Figure(figsize=(3.2 * nc, 2.3 * nr))
        axs = fig.subplots(nr, nc, sharex=True, sharey=True, squeeze=False)
        x = np.arange(len(dates))
        for gi, g in enumerate(groups):
            ax = axs[gi // nc, gi % nc]
            for r in [r_ for r_ in rows if r_["group"] == g]:
                v = np.array([float(r[d]) if r[d] not in ("", None) else np.nan for d in dates])
                ok = np.isfinite(v)
                if ok.any():
                    ax.plot(x[ok], v[ok], "o-", ms=3, lw=1, label=instance(r["plant_id"]))
            ax.set_title(cluster(g), fontsize=9)
            if ax.lines:
                ax.legend(fontsize=6, loc="upper left")
        for k in range(len(groups), nr * nc):
            axs[k // nc, k % nc].axis("off")
        for ax in axs[-1]:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d[4:6]}.{d[2:4]}" for d in dates], fontsize=7, rotation=45)
        fig.suptitle(f"{label} per plant instance (gaps = plant instance missing or not measured)", fontsize=11)
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.9, wspace=0.15, hspace=0.45)
        canvas = FigureCanvasTkAgg(fig, master=self.right)
        HelpToolbar(canvas, self.right).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _plant_growth(self, pid):
        if not pid:
            return
        rows = [r for r in _read_csv(self.run_root / "tracking" / "traits_by_date.csv") if r["plant_id"] == pid]
        from matplotlib.figure import Figure
        keys = [("plant_height_m", "height m", 1), ("stem_length_m", "stem length m", 1), ("n_leaves", "visible leaves", 1),
                ("total_leaf_area_m2", "leaf area cm²", 1e4), ("stem_diameter_mm", "stem diameter mm", 1),
                ("mean_leaf_angle_to_stem_deg", "leaf angle °", 1)]
        fig = Figure(figsize=(9, 5))
        for k, (key, lab, sc) in enumerate(keys):
            ax = fig.add_subplot(2, 3, k + 1)
            xs, ys, flags = [], [], []
            for i, r in enumerate(rows):
                try:
                    v = float(r.get(key, "nan")) * sc
                except ValueError:
                    v = float("nan")
                if r.get("present") == "True" and v == v:
                    xs.append(i)
                    ys.append(v)
                    flags.append(bool(r.get("check")))
            ax.plot(xs, ys, "o-", color="#2f6b3a")
            for x_, y_, f_ in zip(xs, ys, flags):
                if f_:
                    ax.plot(x_, y_, "o", color="#c0392b", ms=9, mfc="none")
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([r["date"][2:] for r in rows], fontsize=7)
            ax.set_title(lab, fontsize=9)
        fig.suptitle(f"{nice(pid)}: growth (red ring = height check failed on that date)", fontsize=10)
        fig.tight_layout()
        self._embed(fig)

    # ------------------------------------------------------------------ drawing helpers
    def _embed(self, fig):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from ui.mpl_toolbar import HelpToolbar as NavigationToolbar2Tk
        for w in self.right.winfo_children():
            if w is not self.image:
                w.destroy()
        self.image.pack_forget()
        canvas = FigureCanvasTkAgg(fig, master=self.right)
        NavigationToolbar2Tk(canvas, self.right).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self.canvas = canvas

    def _mpl_3d(self, layers, title, lines=()):
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        allp = []
        for pts, col, cmap in layers:
            if cmap:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, cmap=cmap, s=0.3)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=0.6, alpha=0.25 if lines else 1.0)
            allp.append(pts)
        for pts, col, label, tag in lines:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, lw=3.5 if label else 2.2, label=label)
            if tag:
                ax.text(*pts[-1], tag, fontsize=10, fontweight="bold")
            allp.append(pts)
        if allp:
            P = np.vstack(allp)
            ax.set_box_aspect(np.ptp(P, axis=0) + 1e-3)
        ax.set_title(title, fontsize=9)
        if lines:
            ax.legend(loc="upper left", fontsize=8)
        self._embed(fig)
