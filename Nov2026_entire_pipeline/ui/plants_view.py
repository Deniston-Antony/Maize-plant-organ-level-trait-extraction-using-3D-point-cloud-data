"""Rotatable 3D view of the separated plants of one date and their groups.

Every plant of 03_plants is drawn in its own colour (or coloured by group,
or stem / leaves). Choose one group, or all; the plant names can be shown at
the stem bases, and the soil can be added for context. Large dates are
shown as an even sample per plant so rotating stays fast.
"""

from __future__ import annotations

import csv
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from pipeline.names import cluster, instance, raw
from ui.offscreen_view import CheckMenu, OffscreenView

ALL = "all plant clusters"

MAX_ALL = 1_500_000      # points shown for all groups together
MAX_GROUP = 1_500_000    # points shown for one group
_cache: dict = {}


def _palette(n: int) -> np.ndarray:
    import matplotlib
    base = [matplotlib.colormaps["tab10"](i)[:3] for i in range(10)] + \
           [matplotlib.colormaps["Dark2"](i)[:3] for i in range(8)] + \
           [matplotlib.colormaps["Set1"](i)[:3] for i in range(9)]
    return np.array([base[i % len(base)] for i in range(max(n, 1))])


def load_plants(plants_dir: Path) -> dict:
    """{plant_id: {"xyz" (local), "labels", "base"}} for every saved plant of a date (cached)."""
    from pipeline.config import PROJECT_ORIGIN
    O = np.asarray(PROJECT_ORIGIN)
    idx = plants_dir / "plants_index.csv"
    key = (str(plants_dir), idx.stat().st_mtime if idx.exists() else 0)
    if key in _cache:
        return _cache[key]
    out = {}
    if idx.exists():
        for r in csv.DictReader(open(idx)):
            f = idx.parent / r["file"]
            if not f.exists():
                continue
            z = np.load(f)
            out[r["plant_id"]] = {"xyz": (z["xyz"] - O).astype(np.float32), "labels": z["labels"],
                                  "base": z["base"] - O}
    _cache.clear()
    _cache[key] = out
    return out


class PlantsView(OffscreenView):
    loading_text = "Loading the separated plants ..."

    def __init__(self, master, date_dir: Path, date: str, title: str, log=print, group: str | None = None):
        super().__init__(master, title, log)
        self.date_dir, self.date = Path(date_dir), date
        self.plants: dict = {}
        self.group = tk.StringVar(value=cluster(group) if group else ALL)
        self.colour = tk.StringVar(value="by plant instance")
        self.names = tk.BooleanVar(value=True)
        self.soil = tk.BooleanVar(value=False)
        bar = self.layer_bar()
        ttk.Label(bar, text="Plant cluster:").pack(side="left")
        self.g_cb = ttk.Combobox(bar, textvariable=self.group, width=18, state="readonly", values=(ALL,))
        self.g_cb.pack(side="left")
        self.g_cb.bind("<<ComboboxSelected>>", lambda _: self._build(new_view=True))
        ttk.Label(bar, text="  Colour:").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.colour, width=17, state="readonly",
                          values=("by plant instance", "by plant cluster", "stem / leaves"))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _: self._build())
        CheckMenu(bar, "Show", [("names", self.names), ("soil", self.soil)], self._build).pack(side="left", padx=(8, 0))
        self.info = ttk.Label(bar, text="")
        self.info.pack(side="left", padx=8)
        self.make_view()
        self.in_background(self._read, lambda r: self._loaded(*r), self._failed)

    def show_group(self, group: str) -> None:
        """Show one group (used when a plant is picked in the table)."""
        if self.plants and group != raw(self.group.get()):
            self.group.set(cluster(group))
            self._build(new_view=True)
        elif not self.plants:
            self.group.set(cluster(group))

    def _read(self):
        plants = load_plants(self.date_dir / "03_plants")
        soil = None
        sp = self.date_dir / "01_soil" / f"{self.date}_soil.ply"
        if sp.exists():
            from pipeline.config import PROJECT_ORIGIN
            from pipeline.io_utils import read_cloud
            soil = (read_cloud(sp).xyz - np.asarray(PROJECT_ORIGIN)).astype(np.float32)
        return plants, soil

    def _failed(self, e):
        msg = f"Could not load the plants: {e}"
        self.show_message(msg)
        self.log(msg)

    def _loaded(self, plants, soil):
        self.plants, self.soil_xyz = plants, soil
        if not plants:
            self.show_message("No separated plant instances for this date yet (run up to the plants step).")
            return
        groups = sorted({p.split("_")[0] for p in plants})
        self.g_cb.configure(values=(ALL, *[cluster(g) for g in groups]))
        if raw(self.group.get()) not in groups:
            self.group.set(ALL)
        self._build(new_view=True)

    def _build(self, new_view=False):
        if not self.plants:
            return
        import pyvista as pv
        cam = self.plotter.camera_position if (self.plotter is not None and not new_view) else None
        g = raw(self.group.get())
        pids = sorted(p for p in self.plants if g == ALL or p.startswith(g + "_"))
        total = sum(len(self.plants[p]["xyz"]) for p in pids)
        budget = MAX_ALL if g == ALL else MAX_GROUP
        keep = min(1.0, budget / max(total, 1))
        rng = np.random.default_rng(0)
        groups = sorted({p.split("_")[0] for p in self.plants})
        pcol, gcol = _palette(len(pids)), _palette(len(groups))
        pl = self.new_plotter()
        pts, cols = [], []
        for i, p in enumerate(pids):
            d = self.plants[p]
            n = len(d["xyz"])
            sel = np.arange(n) if keep >= 1 else np.sort(rng.choice(n, max(1, int(n * keep)), replace=False))
            xyz = d["xyz"][sel].astype(np.float64)
            mode = self.colour.get()
            if mode == "by plant cluster":
                c = np.tile(gcol[groups.index(p.split("_")[0])], (len(sel), 1))
            elif mode == "stem / leaves":
                lab = d["labels"][sel]
                c = np.where((lab == 0)[:, None], [0.15, 0.15, 0.15], [0.20, 0.70, 0.20])
                c[lab < 0] = (0.75, 0.75, 0.75)
            else:
                c = np.tile(pcol[i], (len(sel), 1))
            pts.append(xyz)
            cols.append(c)
        if pts:
            cloud = pv.PolyData(np.vstack(pts))
            cloud["rgb"] = (np.vstack(cols) * 255).astype(np.uint8)
            pl.add_points(cloud, scalars="rgb", rgb=True, point_size=2)
        if self.soil.get() and self.soil_xyz is not None and pts:
            allp = np.vstack(pts)
            lo, hi = allp.min(axis=0) - 0.1, allp.max(axis=0) + 0.1
            m = np.all((self.soil_xyz[:, :2] >= lo[:2]) & (self.soil_xyz[:, :2] <= hi[:2]), axis=1)
            if m.any():
                pl.add_points(pv.PolyData(self.soil_xyz[m].astype(np.float64)), color=(0.55, 0.40, 0.25), point_size=2)
        if self.names.get() and pids:
            if g == ALL:     # one name per plant cluster (plant names would overlap)
                gl = sorted({p.split("_")[0] for p in pids})
                pos = []
                for gg in gl:
                    mem = [p for p in pids if p.startswith(gg + "_")]
                    xy = np.mean([self.plants[p]["base"][:2] for p in mem], axis=0)
                    zt = max(float(self.plants[p]["xyz"][:, 2].max()) for p in mem) + 0.03
                    pos.append([*xy, zt])
                pl.add_point_labels(np.array(pos), [cluster(x) for x in gl], font_size=11, point_size=0, show_points=False,
                                    shape_opacity=0.6, always_visible=True)
            else:                     # name above each plant (stem bases of one group are too close together)
                pos = np.array([[*self.plants[p]["base"][:2], float(self.plants[p]["xyz"][:, 2].max()) + 0.03]
                                for p in pids])
                pl.add_point_labels(pos, [instance(p) for p in pids], font_size=14, point_size=6,
                                    point_color="black", shape_opacity=0.6, always_visible=True)
        pl.add_text(f"{self.title}   {cluster(g)}", font_size=10)
        self.info.configure(text=f"{len(pids)} plant instances, {total:,} points" + (f" ({keep:.0%} shown)" if keep < 1 else ""))
        if cam is not None:
            pl.camera_position = cam
        else:
            pl.view_isometric()
            pl.reset_camera()
            pl.camera.Zoom(1.6 if g == ALL else 1.2)
            self._home = pl.camera_position
        self._render()
