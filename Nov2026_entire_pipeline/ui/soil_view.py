"""Rotatable 3D view of the soil / plant separation of one date.

Layers: plant points (plants.npz), soil points (<date>_soil.ply) and, when
the ground-outlier clean-up is on, the removed points. Colour by class
(plant green, soil brown, removed red), by height above the ground, or by
the scan's own colour. Large clouds are shown as an even sample so rotating
stays fast; the counts of all points are given in the bar.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from ui.offscreen_view import CheckMenu, OffscreenView

KEEP_PLANTS = 8_000_000   # plant points kept in memory per date (even sample above this)
MAX_SHOWN = {"plants": 1_500_000, "soil": 800_000, "outliers": 300_000}
CLASS_COLOUR = {"plants": (0.10, 0.65, 0.15), "soil": (0.55, 0.40, 0.25), "outliers": (0.90, 0.10, 0.10)}
NAMES = {"plants": "plant points", "soil": "soil points", "outliers": "removed outliers"}


def _sample(n: int, k: int) -> np.ndarray:
    return np.arange(n) if n <= k else np.sort(np.random.default_rng(0).choice(n, k, replace=False))


def load_layers(soil_dir: Path, date: str) -> dict:
    """{name: {"xyz", "hag", "rgb", "n_total"}} in the project frame (local metres)."""
    from pipeline.config import PROJECT_ORIGIN
    from pipeline.io_utils import read_cloud
    from pipeline.soil import DTM
    O = np.asarray(PROJECT_ORIGIN)
    out = {}
    f = soil_dir / "plants.npz"
    if f.exists():
        z = np.load(f)
        n = len(z["hag"])
        # memory: keep at most KEEP_PLANTS points (an even sample; a 1 m part still has ~1 M points).
        # 11.07 has 20 M plant points = 400 MB at full size; the sample is float32 / uint8 only.
        i = _sample(n, KEEP_PLANTS)
        rgb = z["rgb"] if len(z["rgb"]) == n else None
        out["plants"] = {"xyz": np.ascontiguousarray(z["xyz"][i], np.float32),
                         "hag": np.ascontiguousarray(z["hag"][i], np.float32),
                         "rgb": rgb[i] if rgb is not None else None, "n_total": n}
        del z
    dtm = DTM.load(soil_dir / "dtm.npz") if (soil_dir / "dtm.npz").exists() else None
    for key, fn in (("soil", f"{date}_soil.ply"), ("outliers", f"{date}_ground_outliers.ply")):
        p = soil_dir / fn
        if not p.exists():
            continue
        c = read_cloud(p)
        if len(c.xyz) == 0:
            continue
        xyz = (c.xyz - O).astype(np.float32)
        hag = (xyz[:, 2] - dtm.height(xyz[:, 0], xyz[:, 1])).astype(np.float32) if dtm is not None else np.zeros(len(xyz), np.float32)
        rgb = (c.rgb * 255).astype(np.uint8) if c.rgb is not None else None
        out[key] = {"xyz": xyz, "hag": hag, "rgb": rgb, "n_total": len(c.xyz)}
    # along-row coordinate (row direction from the plant points)
    ref = out.get("plants") or next(iter(out.values()), None)
    if ref is not None:
        q = np.asarray(ref["xyz"][:: max(1, len(ref["xyz"]) // 200_000), :2], np.float64)
        o = q.mean(axis=0)
        w, v = np.linalg.eigh(np.cov((q - o).T))
        along = v[:, int(np.argmax(w))]
        for lay in out.values():
            lay["s"] = (np.asarray(lay["xyz"][:, :2], np.float64) - o) @ along
    return out


class SoilView(OffscreenView):
    loading_text = "Loading the soil and plant points ..."

    def __init__(self, master, soil_dir: Path, date: str, title: str, log=print):
        super().__init__(master, title, log)
        self.soil_dir, self.date = Path(soil_dir), date
        self.layers: dict = {}
        self.vis = {k: tk.BooleanVar(value=True) for k in NAMES}
        self.colour = tk.StringVar(value="plant / soil")
        self.bar = self.layer_bar()
        self.show_menu = CheckMenu(self.bar, "Show", [(name, self.vis[k]) for k, name in NAMES.items()],
                                   self._apply_visibility)
        self.show_menu.pack(side="left")
        self.keys = list(NAMES)
        ttk.Label(self.bar, text="   Colour:").pack(side="left")
        self.cb = ttk.Combobox(self.bar, textvariable=self.colour, width=20, state="readonly",
                               values=("plant / soil", "height above ground", "scan colour"))
        self.cb.pack(side="left")
        self.cb.bind("<<ComboboxSelected>>", lambda _: self._build())
        ttk.Label(self.bar, text="   Part of the row:").pack(side="left")
        self.section = tk.StringVar(value="whole row")
        self.sec_cb = ttk.Combobox(self.bar, textvariable=self.section, width=14, state="readonly", values=("whole row",))
        self.sec_cb.pack(side="left")
        self.sec_cb.bind("<<ComboboxSelected>>", lambda _: self._build(new_view=True))
        self.make_view()
        self.in_background(lambda: load_layers(self.soil_dir, self.date), self._loaded, self._failed)

    def _failed(self, e):
        msg = f"Could not load the soil / plant points: {e}"
        self.show_message(msg)
        self.log(msg)

    def _loaded(self, layers):
        self.layers = layers
        s_all = np.concatenate([v["s"] for v in layers.values() if "s" in v]) if layers else np.zeros(1)
        lo, hi = np.floor(s_all.min()), np.ceil(s_all.max())
        secs = [f"{a - lo:.0f}-{a - lo + 1:.0f} m along the row" for a in np.arange(lo, hi)]
        self._sec_lo = {t: float(a) for t, a in zip(secs, np.arange(lo, hi))}
        self.sec_cb.configure(values=("whole row", *secs))
        for i, k in enumerate(self.keys):
            if k not in layers:
                self.vis[k].set(False)
                self.show_menu.set_item(i, label=f"{NAMES[k]} (none)", enabled=False)
        if not any(v["rgb"] is not None for v in layers.values()):
            self.cb.configure(values=("plant / soil", "height above ground"))
        self._build()

    def _part(self, v, k):
        """Points of one layer in the chosen part of the row (sampled if too many)."""
        sec = self.section.get()
        m = np.ones(len(v["xyz"]), bool) if sec == "whole row" else \
            (v["s"] >= self._sec_lo[sec]) & (v["s"] < self._sec_lo[sec] + 1.0)
        idx = np.where(m)[0]
        idx = idx[_sample(len(idx), MAX_SHOWN[k])]
        return {"xyz": np.asarray(v["xyz"][idx], np.float64), "hag": np.asarray(v["hag"][idx], np.float64),
                "rgb": (v["rgb"][idx] / 255.0) if v["rgb"] is not None else None,
                "n_in": int(round(m.sum() * v["n_total"] / max(1, len(v["xyz"]))))}   # points in the scan

    def _build(self, new_view=False):
        if not self.layers:
            return
        cam = self.plotter.camera_position if (self.plotter is not None and not new_view) else None
        import pyvista as pv
        pl = self.new_plotter()
        self.actors = {}
        mode = self.colour.get()
        parts = {k: self._part(v, k) for k, v in self.layers.items()}
        for i, k in enumerate(self.keys):
            if k in parts:
                n, shown = parts[k]["n_in"], len(parts[k]["xyz"])
                self.show_menu.set_item(i, label=f"{NAMES[k]} ({n:,}" + (f", {shown:,} shown)" if shown < n else ")"))
        top = float(np.percentile(parts["plants"]["hag"], 99)) if "plants" in parts and len(parts["plants"]["hag"]) else 0.5
        for k, v in parts.items():
            if not len(v["xyz"]):
                continue
            cloud = pv.PolyData(v["xyz"])
            if mode == "height above ground":
                cloud["height above ground (m)"] = v["hag"]
                a = pl.add_points(cloud, scalars="height above ground (m)", cmap="viridis", clim=(-0.03, top),
                                  point_size=2, show_scalar_bar=(k == "plants"))
            elif mode == "scan colour" and v["rgb"] is not None:
                cloud["rgb"] = (np.asarray(v["rgb"]) * 255).astype(np.uint8)
                a = pl.add_points(cloud, scalars="rgb", rgb=True, point_size=2)
            else:
                a = pl.add_points(cloud, color=CLASS_COLOUR[k], point_size=2)
            self.actors[k] = a
        pl.add_text(f"{self.title}   ({self.section.get()})", font_size=10)
        if cam is not None:
            pl.camera_position = cam
        else:
            pl.view_isometric()
            pl.reset_camera()
            pl.camera.Zoom(1.8 if self.section.get() == "whole row" else 1.3)
            self._home = pl.camera_position
        self._apply_visibility()

    def _apply_visibility(self):
        if self.plotter is None:
            return
        for k, a in self.actors.items():
            a.SetVisibility(self.vis[k].get())
        self._render()
