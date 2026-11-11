"""Step 7 - compare dates: the same plants on several dates, on top of each other.

Choose the dates and a group (or one plant). The structure of each plant
(from the user's structure script: stem points, leaf clusters, stem
skeleton, leaf midribs) is moved into the reference date's frame with the
date alignment, so the plants of all dates stand on the same spot.

Views: one with all selected dates overlapped (one colour per date) and one
per date (stem dark grey, each leaf its own colour, skeleton on top). All
views move together: rotating or zooming one turns every date the same way.
"""

from __future__ import annotations

import csv
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from pipeline.names import cluster, nice, raw
from ui.offscreen_view import CameraLink, CheckMenu, OffscreenView

ALL = "all plant instances of the cluster"

DATE_COLOURS = [(0.12, 0.47, 0.71), (1.0, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
                (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50)]
STEM_GREY = (0.25, 0.25, 0.25)


def pretty(d: str) -> str:
    return f"{d[4:6]}.{d[2:4]}.20{d[:2]}" if len(d) == 6 else d


def plant_structure(run: Path, date: str, pid: str, al) -> dict | None:
    """Structure of one plant on one date, in the reference frame (local metres)."""
    from pipeline.config import PROJECT_ORIGIN
    from ui.user_structure_view import user_result
    npz = run / date / "03_plants" / pid.split("_")[0] / pid / f"{pid}_labelled.npz"
    if not npz.exists():
        return None
    res, _ = user_result(npz)                                  # cached scene of the user's script
    centre = (np.load(npz)["xyz"] - np.asarray(PROJECT_ORIGIN)).mean(axis=0)

    def move(a):
        a = np.asarray(a, np.float64).reshape(-1, 3) + centre
        return al.apply(a) if (al is not None and len(a)) else a
    return {"pid": pid, "stem": move(res.segmentation.stem_points),
            "leaves": [move(c) for c in res.leaf_clusters.clusters],
            "axis": move(res.smoothed_stem), "paths": [move(p) for p in res.smoothed_leaf_paths]}


class Panel(OffscreenView):
    """One linked 3D view (no own buttons; the compare bar controls all panels)."""
    loading_text = "Loading ..."

    def __init__(self, master, title: str):
        super().__init__(master, title)
        self.make_view()

    def draw(self, items: list[tuple[dict, str]], mode: str, show_pts: bool, show_skel: bool, legend=None):
        """items: (structure, colour key) ; mode 'date' = one colour per item, 'structure' = stem grey / leaves."""
        import pyvista as pv
        from pipeline.user_code import structure_module
        ref = structure_module()
        pl = self.new_plotter()
        for st, col in items:
            if show_pts:
                if mode == "date":
                    pts = np.vstack([st["stem"], *st["leaves"]]) if st["leaves"] else st["stem"]
                    if len(pts):
                        pl.add_points(pv.PolyData(pts), color=col, point_size=2)
                else:
                    if len(st["stem"]):
                        pl.add_points(pv.PolyData(st["stem"]), color=STEM_GREY, point_size=2)
                    cc = ref.cluster_colors(len(st["leaves"]))
                    for c, cl in zip(st["leaves"], cc):
                        if len(c):
                            pl.add_points(pv.PolyData(c), color=tuple(cl), point_size=2)
            if show_skel:
                lw = 6
                if len(st["axis"]) >= 2:
                    pl.add_mesh(pv.lines_from_points(st["axis"]), color=col if mode == "date" else "blue",
                                line_width=lw, render_lines_as_tubes=True)
                cc = ref.cluster_colors(len(st["paths"]))
                for p, cl in zip(st["paths"], cc):
                    if len(p) >= 2:
                        pl.add_mesh(pv.lines_from_points(p), color=col if mode == "date" else tuple(cl),
                                    line_width=lw, render_lines_as_tubes=True)
        pl.add_text(self.title, font_size=11)
        self.legend = legend or []                        # drawn on the picture (see _draw_scale)
        self.message = "" if items else "not present on this date"
        self._render()

    def _draw_scale(self, img) -> None:
        """Scale bar + a legend box (one line per date, spaced by the text height) + message."""
        super()._draw_scale(img)
        from PIL import ImageDraw
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=16)
        except Exception:  # noqa: BLE001
            font = None
        dr = ImageDraw.Draw(img)
        w, h = img.size
        if getattr(self, "legend", None):
            labs = [lab for lab, _ in self.legend]
            tw = max(dr.textlength(t, font=font) for t in labs) if hasattr(dr, "textlength") else 9 * max(map(len, labs))
            lh, sq, pad = 22, 14, 8
            bw, bh = int(tw + sq + 3 * pad), lh * len(labs) + pad
            x0, y0 = w - bw - 10, 34
            dr.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(150, 150, 150))
            for i, (lab, col) in enumerate(self.legend):
                y = y0 + pad // 2 + i * lh
                c = tuple(int(255 * v) for v in col)
                dr.rectangle([x0 + pad, y + 3, x0 + pad + sq, y + 3 + sq], fill=c)
                dr.text((x0 + 2 * pad + sq, y), lab, fill=(30, 30, 30), font=font)
        if getattr(self, "message", ""):
            tw = dr.textlength(self.message, font=font) if hasattr(dr, "textlength") else 9 * len(self.message)
            dr.text(((w - tw) / 2, h / 2), self.message, fill=(140, 140, 140), font=font)


class CompareDates(ttk.Frame):
    def __init__(self, master, run: Path, dates: list[str], reference: str, log=print):
        super().__init__(master)
        self.run, self.dates, self.reference, self.log = Path(run), dates, reference, log
        from pipeline.align import load_alignments
        af = self.run / "alignment" / "alignment.json"
        self.als = load_alignments(af) if af.exists() else {}
        reg = self.run / "tracking" / "plant_register.csv"
        self.register = list(csv.DictReader(open(reg))) if reg.exists() else []
        groups = sorted({r["group"] for r in self.register})
        have = [d for d in dates if (self.run / d / "03_plants" / "plants_index.csv").exists()]

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.dvars = {d: tk.BooleanVar(value=d in have) for d in dates}
        dm = CheckMenu(bar, "Dates", [(pretty(d) + ("" if d in have else "  (no plant instances)"), self.dvars[d]) for d in dates],
                       self.update_views)
        for i, d in enumerate(dates):
            if d not in have:
                dm.set_item(i, enabled=False)
        dm.pack(side="left")
        ttk.Label(bar, text="   Plant cluster:").pack(side="left")
        self.group = tk.StringVar(value=cluster(groups[0]) if groups else "")
        g_cb = ttk.Combobox(bar, textvariable=self.group, values=[cluster(g) for g in groups], width=17, state="readonly")
        g_cb.pack(side="left")
        g_cb.bind("<<ComboboxSelected>>", lambda _: (self._fill_plants(), self.update_views()))
        ttk.Label(bar, text=" Plant instance:").pack(side="left")
        self.plant = tk.StringVar()
        self.p_cb = ttk.Combobox(bar, textvariable=self.plant, width=34, state="readonly")
        self.p_cb.pack(side="left")
        self.p_cb.bind("<<ComboboxSelected>>", lambda _: self.update_views())
        self.show_pts, self.show_skel = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        CheckMenu(bar, "Show", [("points", self.show_pts), ("skeleton", self.show_skel)],
                  self.update_views).pack(side="left", padx=(10, 0))
        ttk.Button(bar, text="Reset view", command=self._reset).pack(side="right")
        from ui.offscreen_view import size_sliders
        size_sliders(bar, lambda: [p._apply_point_size() for p in self.panels])
        self.info = ttk.Label(self, foreground="#888",
                              text="Drag in any view to rotate, two fingers to zoom: all dates move together. "
                                   "Plant instances are placed with the date alignment (reference date "
                                   f"{pretty(reference)}).")
        self.info.pack(anchor="w")
        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(fill="both", expand=True)
        self.panels: list[Panel] = []
        self._job = 0
        self._fill_plants()
        if not have:
            ttk.Label(self.grid_frame, text="No separated plant instances in this run yet.").pack()
        else:
            self.update_views()

    def _fill_plants(self):
        g = raw(self.group.get())
        pids = [r["plant_id"] for r in self.register if r["group"] == g]
        self.p_cb.configure(values=(ALL, *[nice(p) for p in pids]))
        self.plant.set(ALL)

    def _pids(self) -> list[str]:
        if self.plant.get() and self.plant.get() != ALL:
            return [raw(self.plant.get())]
        return [r["plant_id"] for r in self.register if r["group"] == raw(self.group.get())]

    def _reset(self):
        if self.panels:
            self.panels[0]._reset()

    def update_views(self):
        dates = [d for d in self.dates if self.dvars[d].get()]
        pids = self._pids()
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.panels = []
        if not dates or not pids:
            ttk.Label(self.grid_frame, text="Select at least one date and a plant cluster.").pack()
            return
        n = len(dates) + 1
        ncol = 3 if n > 4 else min(n, 2 if n <= 2 else 3)
        nrow = -(-n // ncol)
        for c in range(ncol):
            self.grid_frame.columnconfigure(c, weight=1, uniform="col")
        for r in range(nrow):
            self.grid_frame.rowconfigure(r, weight=1, uniform="row")
        link = CameraLink()
        name = nice(pids[0]) if len(pids) == 1 else self.group.get()
        titles = [f"{name}: all selected dates overlapped"] + [f"{name}  {pretty(d)}" for d in dates]
        for i, t in enumerate(titles):
            p = Panel(self.grid_frame, t)
            p.grid(row=i // ncol, column=i % ncol, sticky="nsew", padx=2, pady=2)
            link.add(p)
            self.panels.append(p)
        self.link = link
        self._job += 1
        job = self._job
        q: queue.Queue = queue.Queue()

        def work():
            out = {}
            for d in dates:
                al = self.als.get(d)
                out[d] = [s for s in (plant_structure(self.run, d, pid, al) for pid in pids) if s is not None]
            return out

        def run():
            try:
                q.put((True, work()))
            except Exception as e:  # noqa: BLE001
                q.put((False, e))
        threading.Thread(target=run, daemon=True).start()

        def poll():
            if job != self._job or not self.winfo_exists():
                return
            try:
                ok, val = q.get_nowait()
            except queue.Empty:
                self.after(80, poll)
                return
            if not ok:
                self.log(f"Compare dates: could not load the plant instances ({val})")
                return
            self._draw(dates, val)
        self.after(80, poll)

    def _draw(self, dates, data):
        sp, ss = self.show_pts.get(), self.show_skel.get()
        cols = {d: DATE_COLOURS[i % len(DATE_COLOURS)] for i, d in enumerate(self.dates)}
        over = [(st, cols[d]) for d in dates for st in data[d]]
        legend = [[pretty(d), cols[d]] for d in dates if data[d]]
        self.panels[0].draw(over, "date", sp, ss, legend=legend)
        for p, d in zip(self.panels[1:], dates):
            p.draw([(st, cols[d]) for st in data[d]], "structure", sp, ss)
        # same start view for every panel: fitted to all dates
        p0 = self.panels[0].plotter
        p0.view_isometric()
        p0.reset_camera()
        p0.camera.Zoom(1.2)
        for p in self.panels:
            if p.plotter is not None:
                p.plotter.camera_position = p0.camera_position
                p.plotter.camera.SetViewAngle(p0.camera.GetViewAngle())
                p._home = p0.camera_position
                p._render()
        n = sum(len(v) for v in data.values())
        self.info.configure(text=f"{n} plant instance structures on {len(dates)} dates. Drag in any view to rotate, two fingers "
                                 f"to zoom: all dates move together (aligned to {pretty(self.reference)}).")
