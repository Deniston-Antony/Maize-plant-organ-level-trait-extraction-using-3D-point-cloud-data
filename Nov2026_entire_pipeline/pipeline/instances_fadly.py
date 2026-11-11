"""Stage 5 (Nov2026 pipeline): plant instances with the plant_instance_fadly code.

The code in ../plant_instance_fadly/ (copied unchanged from
Plant_instance_fadly_nov2026/plant_instances_from_las) is used the way its
README describes for a soil-removed cloud (run_from_las.py --prepared-ground-ply):

  1. thin the plant points to 4 mm (first point per voxel, as step1_downsample.py)
  2. stem / leaf / unknown per point with the bundled ExtraTrees model
     (rfpipe features -> forest -> neighbourhood smoothing -> confidence threshold)
  3. stem-location hints from the predicted stem points (run_from_las.stem_location_hints)
  4. candidate stem axes + stem-guided plant groups (gap_segmentation.segment)

Soil removal stays the user's classify_plants.py (stage 1-2), so steps 1-2 of
the README (LAS downsampling + its own CSF) are not run here.

The README's manual review (fill plant_id in stem_axes_to_review.csv, then
segment_confirmed.py) is filled in automatically: each candidate axis is matched
to the nearest plant of the run's plant register (same plant names on every
date) within 9 cm, the code's own prior-to-axis gate. Unmatched axes are left
blank, which the README treats as rejected. All other settings are the code's own.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .config import CODE_DIR, PROJECT_ORIGIN

FADLY_DIR = CODE_DIR / "plant_instance_fadly"


@dataclass
class FadlyInstanceConfig:
    # Every setting of the instance code itself is used unchanged: step1_downsample.VOXEL (4 mm),
    # the model's own smoothing / confidence threshold, gap_segmentation.Settings() defaults.
    labels: str = "accepted"           # "accepted" = segment_confirmed.py plant_id (conservative: no gap
                                       # paths, no near-ties); "candidate" = plants_candidate_including_gaps
    match_max_dist: float = 0.09       # automatic review: axis base -> register plant. 0.09 m is the code's
                                       # own gate for matching a stem-location prior to an axis (make_sources)
    # small-plant dates only (chosen in the app): the code's height settings (slice_height,
    # minimum_track_span, track_max_gap) times height_scale, and this graph voxel. 1.0 / None = code's own.
    height_scale: float = 1.0
    graph_voxel: float | None = None
    # only for the availability text of plants without an axis (not part of the instance code)
    seed_search: float = 0.03
    hidden_min_points: int = 200
    hidden_low: float = 0.20
    hidden_max_low: int = 20

    def to_dict(self) -> dict:
        return {"method": "plant_instance_fadly (Nov2026)", **asdict(self)}

    @classmethod
    def small_plants(cls) -> "FadlyInstanceConfig":
        """Seedling dates (tested on 01.06: best of x1/2, x1/3, x1/4 with 8 / 4 mm graph voxels)."""
        return cls(height_scale=0.25, graph_voxel=0.004)

    def settings(self, G):
        """gap_segmentation.Settings(): the code's defaults, height values scaled for small plants."""
        from dataclasses import replace
        s = G.Settings()
        if self.height_scale != 1.0:
            s = replace(s, slice_height=s.slice_height * self.height_scale,
                        minimum_track_span=s.minimum_track_span * self.height_scale,
                        track_max_gap=s.track_max_gap * self.height_scale)
        if self.graph_voxel:
            s = replace(s, graph_voxel=self.graph_voxel)
        return s


_MODULES: dict = {}


def _fadly():
    """Import the copied code once (its folders go to the END of sys.path, so our
    own 'pipeline' package is never shadowed by its plain 'pipeline' folder)."""
    if _MODULES:
        return _MODULES
    for p in (FADLY_DIR, FADLY_DIR / "pipeline"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import rfpipe as R
    import gap_segmentation as G
    import cloud_io as C
    import run_from_las as RL
    import step1_downsample as S1
    from sklearn.exceptions import InconsistentVersionWarning
    R.MODEL_FILE = FADLY_DIR / "model" / "et_model.joblib"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = R.load_model()
    import sklearn
    _MODULES.update(R=R, G=G, C=C, RL=RL, S1=S1, bundle=bundle,
                    version_note=(f"model saved with scikit-learn 1.7.2, loaded with {sklearn.__version__}"
                                  if any(issubclass(w.category, InconsistentVersionWarning) for w in caught) else ""))
    return _MODULES


def _downsample(utm: np.ndarray, las_mins: np.ndarray, voxel: float):
    """First point per voxel, keys as step1_downsample.py: floor((x - LAS header min) / VOXEL)
    in map coordinates. Returns (indices of kept points, voxel of every point)."""
    k = np.floor((utm - las_mins) / voxel).astype(np.int64)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return first, inv.ravel()


def _predict(pts: np.ndarray, log) -> np.ndarray:
    """rfpipe.predict_cloud on an array instead of a PLY file (same steps, same functions)."""
    M = _fadly()
    R, b = M["R"], M["bundle"]
    C = b["config"]
    X = np.hstack([R.compute_scale(pts, k) for k in b.get("ks", R.SCALES)])
    P = R.smooth_proba(pts, b["model"].predict_proba(X), k=C["smooth_k"], rounds=C["smooth_rounds"])
    ids = b["class_ids"]
    lab = np.array([ids[i] for i in P.argmax(axis=1)])
    return np.where(P.max(axis=1) >= C["conf_threshold"], lab, R.CLASS_UNKNOWN).astype(np.int8)


def _match(axes: list[dict], plant_xy: np.ndarray, cfg: FadlyInstanceConfig):
    """Automatic review: axis_id -> register index (or None), one axis per plant by optimal
    assignment within match_max_dist (the code uses the same assignment for its priors)."""
    if not axes or not len(plant_xy):
        return {int(a["axis_id"]): None for a in axes}, {}
    B = np.array([a["base_xy"] for a in axes])
    D = np.linalg.norm(B[:, None, :] - plant_xy[None, :, :], axis=2)
    r, c = linear_sum_assignment(np.where(D <= cfg.match_max_dist, D, 1e6))
    match = {int(a["axis_id"]): None for a in axes}
    dist = {}
    for i, j in zip(r, c):
        if D[i, j] <= cfg.match_max_dist:
            match[int(axes[i]["axis_id"])] = int(j)
            dist[int(axes[i]["axis_id"])] = float(D[i, j])
    return match, dist


def separate_fadly(xyz: np.ndarray, hag: np.ndarray, plant_xy: np.ndarray, out_dir: Path,
                   las_path: str | Path, cfg: FadlyInstanceConfig | None = None, log=print):
    """Same interface as separate.separate: xyz/hag plant points of one date (frame of
    this date, relative to PROJECT_ORIGIN), plant_xy (P, 2) register positions in that
    frame. Returns (label per point = register index or -1, info per register plant)."""
    cfg = cfg or FadlyInstanceConfig()
    M = _fadly()
    G, Cio, RL = M["G"], M["C"], M["RL"]
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    P = len(plant_xy)

    import laspy
    O = np.asarray(PROJECT_ORIGIN)
    with laspy.open(las_path) as fh:
        las_mins = np.array(fh.header.mins, float)
    # 1. 4 mm thinning in map coordinates; the code reads them from an ASCII PLY with 6 decimals
    utm = xyz + O
    keep, vox_of = _downsample(utm, las_mins, M["S1"].VOXEL)
    pts = np.round(utm[keep], 6)
    log(f"  instances (plant_instance_fadly): {len(xyz):,} plant points -> {len(pts):,} at {M['S1'].VOXEL * 1000:.0f} mm")
    # 2. stem / leaf with the bundled ExtraTrees model
    t1 = time.time()
    semantic = _predict(pts, log)
    log(f"    stem/leaf model: stem {np.mean(semantic == 0):.0%}, leaf {np.mean(semantic == 1):.0%}, "
        f"unknown {np.mean(semantic < 0):.0%} [{time.time() - t1:.0f} s]")
    # 3. stem hints; geometry in local metres like run_from_las.py (saved there as float32 xyz.npy,
    #    int8 semantic.npy and JSON hints, then read back by run_review.py)
    origin = pts.min(axis=0)
    hints = json.loads(json.dumps(RL.stem_location_hints(pts - origin, semantic)))
    local = (pts - origin).astype(np.float32)
    semantic = semantic.astype(np.int8)
    # 4a. review pass, as run_review.py: segment() without axis groups -> candidate axes
    settings = cfg.settings(G)
    if settings != G.Settings():
        log(f"    small-plant settings: slice {settings.slice_height * 100:.2f} cm, minimum stem "
            f"{settings.minimum_track_span * 100:.1f} cm, gap {settings.track_max_gap * 100:.1f} cm, "
            f"graph voxel {settings.graph_voxel * 1000:.0f} mm")
    rev = G.segment(local, semantic, hints, settings, augment=True, bridges=True, weighted=True)
    axes, diag = rev["axes"], rev["source_diagnostics"]
    origin = origin - O                                # back to this pipeline's frame
    match, mdist = _match(axes, plant_xy - origin[:2], cfg)
    n_reg = len({j for j in match.values() if j is not None})
    log(f"    {len(hints)} stem hints, {len(axes)} candidate axes; {n_reg} register plants found, "
        f"{sum(v is None for v in match.values())} axes not in the register")
    # 4b. confirmed pass, as segment_confirmed.py, with the automatic review
    # matched axis -> plant_id = register index + 1; unmatched axes stay blank (= rejected, not seeded)
    groups = {aid: j + 1 for aid, j in match.items() if j is not None}
    t1 = time.time()
    res = G.segment(local, semantic, hints, settings, augment=True, bridges=True, weighted=True,
                    axis_groups=groups) if groups else None
    if res is not None:
        plant_ids = np.array(sorted(set(groups.values())))
        remap = np.zeros(len(plant_ids) + 1, np.int64)
        remap[1:] = plant_ids
        per4 = {k: remap[res[k][res["inverse"]]] for k in ("accepted", "candidate")}
        status = res["status"][res["inverse"]]
        margin = res["margin"][res["inverse"]]
    else:
        per4 = {k: np.zeros(len(pts), np.int64) for k in ("accepted", "candidate")}
        status, margin = np.zeros(len(pts), np.int8), np.zeros(len(pts), np.float32)
    lab4 = per4[cfg.labels] - 1                    # register index, -1 = unassigned
    labels = lab4[vox_of]
    log(f"    plant groups [{time.time() - t1:.0f} s]: {(labels >= 0).mean():.1%} of points assigned "
        f"({cfg.labels} labels)")

    # info per register plant (same keys as separate.separate)
    counts = np.bincount(labels[labels >= 0], minlength=P)
    ptree = cKDTree(xyz[:, :2])
    ground = xyz[:, 2] - hag
    by_plant: dict[int, list] = {}
    for a in axes:
        j = match[int(a["axis_id"])]
        if j is not None:
            by_plant.setdefault(j, []).append(a)
    info = []
    for p in range(P):
        ax = by_plant.get(p, [])
        track = (np.vstack([np.asarray(a["observed_track"]).reshape(-1, 3) for a in ax]) + origin
                 if ax else np.zeros((0, 3)))
        top = 0.0
        if len(track):
            near = ptree.query_ball_point(plant_xy[p], 0.03) or ptree.query_ball_point(plant_xy[p], 0.10)
            g0 = float(np.median(ground[near])) if near else float(np.median(ground))
            top = float(track[:, 2].max() - g0)
        present = bool(ax) and counts[p] > 0
        vis = "visible"
        if not present:
            near = ptree.query_ball_point(plant_xy[p], cfg.seed_search)
            hh = hag[near] if near else np.zeros(0)
            if len(hh) >= cfg.hidden_min_points and (hh < cfg.hidden_low).sum() < cfg.hidden_max_low:
                vis = "covered_by_canopy"
            elif len(hh) < cfg.hidden_min_points:
                vis = "nothing_there"
            else:
                vis = "too_small" if ax else "no_stem_axis_found"
        info.append({"seed_voxels": int(sum(a["seed_nodes"] for a in ax)), "seed_wide_search": False,
                     "stem_followed_to_m": top, "axis": track.tolist(), "n_points": int(counts[p]),
                     "present": present, "visibility": vis, "n_stem_axes": len(ax)})

    # outputs in the code's own formats (local coordinates; origin in summary.json)
    fields = {"plant_id": per4["accepted"].astype(np.int32), "candidate_plant_id": per4["candidate"].astype(np.int32),
              "semantic": semantic.astype(np.int32), "assignment_status": status.astype(np.int32),
              "ownership_margin": margin.astype(np.float32)}
    Cio.write_ply(out_dir / "plants_conservative.ply", local, Cio.instance_colors(per4["accepted"]), fields)
    Cio.write_ply(out_dir / "plants_candidate_including_gaps.ply", local, Cio.instance_colors(per4["candidate"]), fields)
    sem_rgb = np.array([[0, 0, 0], [150, 75, 40], [60, 150, 45]], np.uint8)[semantic.astype(int) + 1]
    Cio.write_ply(out_dir / "semantic_stem_leaf.ply", local, sem_rgb, {"semantic": semantic.astype(np.int32)})
    with open(out_dir / "stem_axes_auto_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["axis_id", "source", "base_x_m", "base_y_m", "vertical_slices", "support_nodes",
                    "plant_id", "register_index", "match_dist_m", "review_notes"])
        for a in axes:
            aid = int(a["axis_id"])
            j = match[aid]
            w.writerow([aid, a["source"], f"{a['base_xy'][0]:.4f}", f"{a['base_xy'][1]:.4f}",
                        len(np.asarray(a["observed_track"]).reshape(-1, 3)), a["seed_nodes"],
                        groups.get(aid, ""), "" if j is None else j, f"{mdist[aid]:.4f}" if aid in mdist else "",
                        "matched to register plant" if j is not None else "no register plant within 9 cm: left blank (rejected)"])
    summary = {"code": str(FADLY_DIR), "labels_used": cfg.labels, "config": cfg.to_dict(),
               "settings": asdict(settings), "model_note": M["version_note"],
               "local_coordinate_origin_xyz_m": (origin + O).tolist(), "las": str(las_path),
               "plant_points": int(len(xyz)), "work_points": int(len(pts)), "stem_hints": len(hints),
               "candidate_axes": len(axes), "register_plants": P, "register_plants_with_axis": n_reg,
               "axes_not_in_register": int(sum(v is None for v in match.values())),
               "plants_present": int(sum(i["present"] for i in info)),
               "assigned_point_share": float((labels >= 0).mean()),
               "status_counts_4mm": {str(i): int((status == i).sum()) for i in range(4)},
               "diagnostics": (res or {}).get("diagnostics", {}),
               "rejected": {k: v for k, v in diag.items() if k != "row_fit_x_from_y"},
               "seconds": round(time.time() - t0, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(f"  separation: {summary['plants_present']} of {P} plants present; {(labels >= 0).mean():.1%} of points "
        f"assigned [{summary['seconds']:.0f} s]")
    return labels, info
