"""Read the separated plants of every date (stage 5-6 outputs) into the
reference frame.

Per plant: points (5 mm thinned) with the structure labels (0 stem, 1..n leaf
pieces, <0 not counted), height above that date's ground, the stem centre line
of the user's structure code, and one record per leaf piece (junction, midrib
path junction -> tip, length, area, insertion height, points).

Frames: the per-plant files are in UTM of the date's own frame. Local =
UTM - PROJECT_ORIGIN; the date's alignment (alignment.json) maps local -> the
reference frame; an accepted fine shift of the plant's cluster (step 4) is
added on top.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pipeline.config import PROJECT_ORIGIN
from pipeline.soil import DTM

O = np.asarray(PROJECT_ORIGIN)


@dataclass
class Piece:
    label: int
    junction: np.ndarray        # reference frame
    path: np.ndarray            # midrib, junction -> tip, reference frame
    length: float
    area: float                 # m2
    insertion_h: float          # height of the junction above the stem base (m), from the structure code
    junction_hag: float         # height of the junction above ground (m)
    n_points: int


@dataclass
class Plant:
    date: str
    pid: str
    xyz: np.ndarray             # thinned points, reference frame
    labels: np.ndarray
    hag: np.ndarray
    centerline: np.ndarray      # stem centre line, reference frame, bottom -> top
    centerline_hag: np.ndarray
    base: np.ndarray            # stem centre line at its lowest point (xy) + ground z
    height: float               # plant height (m) from the traits
    stem_length: float
    pieces: list = field(default_factory=list)
    fine: tuple = (0.0, 0.0)    # fine shift applied (m)
    traits: dict = field(default_factory=dict)

    def lean(self, max_h: float = 0.30) -> np.ndarray:
        """Horizontal lean of the lower stem (dx/dz, dy/dz), from a line fit."""
        m = self.centerline_hag <= max_h
        c = self.centerline[m] if m.sum() >= 3 else self.centerline
        if len(c) < 3 or np.ptp(c[:, 2]) < 0.03:
            return np.zeros(2)
        A = np.column_stack([c[:, 2] - c[:, 2].mean(), np.ones(len(c))])
        sx = np.linalg.lstsq(A, c[:, 0], rcond=None)[0][0]
        sy = np.linalg.lstsq(A, c[:, 1], rcond=None)[0][0]
        return np.array([sx, sy])

    def shift(self, dx: float, dy: float) -> None:
        d = np.array([dx, dy, 0.0])
        self.xyz = self.xyz + d
        self.centerline = self.centerline + d
        self.base = self.base + d
        for p in self.pieces:
            p.junction = p.junction + d
            p.path = p.path + d
        self.fine = (self.fine[0] + dx, self.fine[1] + dy)


def _thin_idx(xyz: np.ndarray, voxel: float) -> np.ndarray:
    k = np.floor(xyz / voxel).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return np.sort(i)


def load_plants(run_dir: Path, date: str, alignment, voxel: float = 0.005, log=print) -> dict:
    """{plant_id: Plant} for every plant of one date whose structure succeeded."""
    pdir = run_dir / date / "03_plants"
    idx = pdir / "plants_index.csv"
    if not idx.exists():
        return {}
    dtm = DTM.load(run_dir / date / "01_soil" / "dtm.npz")
    traits = {r["plant_id"]: r for r in csv.DictReader(open(pdir / "plants_traits.csv"))}
    out = {}
    for r in csv.DictReader(open(idx)):
        pid = r["plant_id"]
        stem = pdir / r["file"].replace("_labelled.npz", "")
        f_l, f_s, f_t = Path(str(stem) + "_labelled.npz"), Path(str(stem) + "_skeleton.npz"), Path(str(stem) + "_traits.json")
        if not (f_l.exists() and f_s.exists() and f_t.exists()):
            continue
        L, S = np.load(f_l), np.load(f_s)
        tj = json.loads(f_t.read_text())
        loc = L["xyz"] - O
        keep = _thin_idx(loc, voxel)
        loc, lab = loc[keep], L["labels"][keep]
        hag = loc[:, 2] - dtm.height(loc[:, 0], loc[:, 1])
        cl = np.asarray(S["centerline"]) - O
        if len(cl) == 0:
            continue
        if cl[0, 2] > cl[-1, 2]:
            cl = cl[::-1]
        cl_hag = cl[:, 2] - dtm.height(cl[:, 0], cl[:, 1])
        z0 = float(dtm.height(cl[:1, 0], cl[:1, 1])[0])
        base_loc = np.array([cl[0, 0], cl[0, 1], z0])
        paths = np.split(np.asarray(S["leaf_points"]) - O, np.cumsum(S["leaf_sizes"])[:-1]) if len(S["leaf_sizes"]) else []
        leaves = {l["leaf_label"]: l for l in tj.get("leaves", [])}
        pieces = []
        for lab_i, p in zip(S["leaf_labels"], paths):
            l = leaves.get(int(lab_i))
            if l is None or len(p) == 0:
                continue
            j = p[0]
            jh = float(j[2] - dtm.height(j[None, 0], j[None, 1])[0])
            pieces.append(Piece(int(lab_i), alignment.apply(j[None])[0], alignment.apply(p),
                                float(l["leaf_length_m"]), float(l["leaf_area_m2"]),
                                float(l["insertion_height_m"]), jh, int(l["n_points"])))
        t = tj.get("traits", {})
        out[pid] = Plant(date, pid, alignment.apply(loc), lab, hag, alignment.apply(cl), cl_hag,
                         alignment.apply(base_loc[None])[0], float(t.get("plant_height_m", np.nan)),
                         float(t.get("stem_length_m", np.nan)), pieces, traits=traits.get(pid, {}))
    log(f"  {date}: {len(out)} separated plants with structure")
    return out
