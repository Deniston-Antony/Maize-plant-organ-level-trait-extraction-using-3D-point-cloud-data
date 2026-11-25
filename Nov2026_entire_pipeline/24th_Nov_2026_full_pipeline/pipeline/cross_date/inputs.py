"""Step 1: read a finished pipeline run and put every date into one frame.

Frame = reference-date coordinates (the pipeline's local coordinates, relative
to PROJECT_ORIGIN), shifted with the FINAL alignment (alignment.json, written
after the stage-4 stem refinement - alignment_summary.csv is stale for some
dates). Height = height above ground (hag) of each date, so ground-level
differences between dates drop out.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.align import load_alignments, row_frame, Alignment
from pipeline.config import PROJECT_ORIGIN
from pipeline.stems import fit_row_centre

from .config import ClusterConfig


DISPLAY_VOXEL = 0.007      # thinning of the plant points saved for the app's 3D views (m)


@dataclass
class Frame:
    origin: np.ndarray     # row frame origin (local xy)
    along: np.ndarray      # unit vector along the row (points north)
    right: np.ndarray      # unit vector across the row (to the right looking north)
    off_row: object        # function xy -> distance from the row centre line
    row_knots: np.ndarray

    def st(self, xy: np.ndarray) -> np.ndarray:
        """(along, across) coordinates of xy points."""
        d = xy - self.origin
        return np.column_stack([d @ self.along, d @ self.right])


@dataclass
class DateLow:
    date: str
    xyz: np.ndarray        # low-layer points in the reference frame (local, aligned)
    hag: np.ndarray
    s: np.ndarray          # along the row
    t: np.ndarray          # across the row
    n_plant_points: int
    column_filter: bool = False   # True if only upright columns were kept
    n_before_column: int = 0


@dataclass
class Run:
    path: Path
    dates: list
    reference: str
    alignments: dict       # date -> Alignment
    frame: Frame
    low: dict              # date -> DateLow
    tall: dict             # date -> (s, hag) of all plant points in the row band, thinned (for clod test)
    register: list         # rows of tracking/plant_register.csv (+ s, t)
    display: dict = None   # date -> (xyz float32, hag float16) all plant points, thinned, reference frame (app views)


def _thin(xyz: np.ndarray, voxel: float) -> np.ndarray:
    k = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return np.sort(idx)


def column_mask(xyz: np.ndarray, hag: np.ndarray, cell: float, min_extent: float) -> np.ndarray:
    """True for points in xy cells whose points span >= min_extent in height (upright structures)."""
    k = np.floor(xyz[:, :2] / cell).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1 if len(inv) else 0
    zmin, zmax = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(zmin, inv, hag)
    np.maximum.at(zmax, inv, hag)
    return (zmax - zmin)[inv] >= min_extent


def load_run(run_dir: Path, cfg: ClusterConfig, log=print, alignments: dict | None = None,
             display: bool = True) -> Run:
    """alignments: use these (e.g. during stage 4, before alignment.json is final) instead of the file.
    display: also keep every plant point thinned to 7 mm (for the app's 3D views)."""
    rc = json.loads((run_dir / "run_config.json").read_text())
    dates = [d for _, d in rc["scans"]]
    ref = rc["reference_date"]
    als = dict(alignments) if alignments is not None else load_alignments(run_dir / "alignment" / "alignment.json")
    als.setdefault(ref, Alignment(ref, ref, [0.0, 0.0], [0.0, 1.0]))

    raw = {}
    for d in dates:
        z = np.load(run_dir / d / "01_soil" / "plants.npz")
        raw[d] = (z["xyz"], z["hag"])
        log(f"  {d}: {len(z['xyz']):,} plant points")

    # row frame exactly as stage 4 builds it (reference date plant points)
    origin, along = row_frame(raw[ref][0][:, :2].astype(np.float64))
    right = np.array([along[1], -along[0]])

    display_on = bool(display)
    low_all, tall, display = {}, {}, {}
    for d in dates:
        xyz, hag = raw[d]
        if display_on:   # every plant point, thinned (1-D voxel key: fast on 20 M points), for the app's 3D views
            k = np.floor(xyz / DISPLAY_VOXEL).astype(np.int64)
            k -= k.min(axis=0)
            key = (k[:, 0] * (k[:, 1].max() + 1) + k[:, 1]) * (k[:, 2].max() + 1) + k[:, 2]
            _, first = np.unique(key, return_index=True)
            display[d] = (als[d].apply(xyz[first].astype(np.float64)).astype(np.float32), hag[first].astype(np.float16))
        m = (hag >= cfg.low_hag[0]) & (hag <= cfg.low_hag[1])
        p = als[d].apply(xyz[m].astype(np.float64))
        low_all[d] = (p, hag[m].astype(np.float64))
        # all plant points (thinned 1 cm) for the clod / height test
        sub = np.arange(0, len(xyz), max(1, len(xyz) // 3_000_000))
        q = als[d].apply(xyz[sub].astype(np.float64))
        keep = _thin(q, 0.01)
        tall[d] = (q[keep], hag[sub][keep].astype(np.float64))

    # row centre line fitted to the low layer of all dates (unweighted, thinned)
    pool = np.vstack([low_all[d][0][:: max(1, len(low_all[d][0]) // 100_000), :2] for d in dates])
    off, knots = fit_row_centre(pool, np.ones(len(pool)), origin, along, max_off=cfg.row_band)
    frame = Frame(origin, along, right, off, knots)

    low = {}
    for d in dates:
        p, h = low_all[d]
        band = off(p[:, :2]) <= cfg.row_band
        p, h = p[band], h[band]
        keep = _thin(p, cfg.thin_voxel)
        p, h = p[keep], h[keep]
        q, qh = tall[d]
        b = off(q[:, :2]) <= cfg.row_band
        tall[d] = (frame.st(q[b, :2])[:, 0], qh[b])
        # column filter, except on a seedling date
        tall_h = qh[b][qh[b] > cfg.low_hag[0]]
        typical = float(np.percentile(tall_h, 95)) if tall_h.size else 0.0
        n0, use_col = len(p), typical >= cfg.seedling_height
        if use_col:
            col = column_mask(p, h, cfg.column_cell, cfg.column_min_extent)
            p, h = p[col], h[col]
        st = frame.st(p[:, :2])
        low[d] = DateLow(d, p, h, st[:, 0], st[:, 1], len(raw[d][0]), use_col, n0)
        log(f"  {d}: {n0:,} low-layer points in the row band (after {cfg.thin_voxel*1000:.0f} mm thinning)"
            + (f", {len(p):,} in upright columns" if use_col else
               f"; plants mostly below {100 * cfg.seedling_height:.0f} cm -> seedling date, no column filter"))

    rf = run_dir / "tracking" / "plant_register.csv"
    reg = list(csv.DictReader(open(rf))) if rf.exists() else []      # none yet during stage 4
    O = np.asarray(PROJECT_ORIGIN)
    if reg:
        xy = np.array([[float(r["x_utm"]) - O[0], float(r["y_utm"]) - O[1]] for r in reg])
        st = frame.st(xy)
        for r, (s, t) in zip(reg, st):
            r["s"], r["t"] = float(s), float(t)
    return Run(run_dir, dates, ref, als, frame, low, tall, reg, display if display_on else None)


def light_run(run_dir: Path, out: Path) -> Run:
    """Run without the point data (frame from a finished 01_clusters/summary.json): enough for the
    leaf steps when steps 1-6 were already done (stage 6b) and are reused."""
    s = json.loads((out / "01_clusters" / "summary.json").read_text())
    als = load_alignments(run_dir / "alignment" / "alignment.json")
    ref = s["reference_date"]
    als.setdefault(ref, Alignment(ref, ref, [0.0, 0.0], [0.0, 1.0]))
    o, a = np.asarray(s["row_frame"]["origin_local_xy"]), np.asarray(s["row_frame"]["along"])
    frame = Frame(o, a, np.array([a[1], -a[0]]), None, None)
    rf = run_dir / "tracking" / "plant_register.csv"
    reg = list(csv.DictReader(open(rf))) if rf.exists() else []
    return Run(run_dir, s["dates"], ref, als, frame, {}, {}, reg, None)
