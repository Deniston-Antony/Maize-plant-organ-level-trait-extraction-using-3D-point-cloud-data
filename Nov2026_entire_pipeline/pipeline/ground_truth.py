"""Ground-truth checker: load hand-labelled plants, find the predicted plant
they correspond to, and compare segmentation (IoU) and traits.

Ground truth can be ONE plant or many. Each GT plant is matched to the
predictions of its scan date by position (UTM coordinates):
  1. position_overlap : most GT points lie on one predicted plant
  2. base_position    : no overlap, but a predicted stem base is within 15 cm
  3. shape_icp        : coordinates were shifted (e.g. exported from
                        CloudCompare without the global shift) -> the GT
                        shape is registered (ICP) to nearby / all predicted
                        plants and the best fit is taken
If the GT date is unknown, every loaded date is tried and the best is used.

IoU is computed on a common voxel grid (5 mm) so a full-resolution GT and a
downsampled prediction are compared fairly.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .config import (
    OUTPUT_ROOT, PROJECT_ORIGIN, GroundTruthConfig, RunPaths, STAGE_DIRS, date_from_name,
)
from .io_utils import POINT_SUFFIXES, read_cloud, voxel_keys

Log = Callable[[str], None]

# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------


@dataclass
class GTPlant:
    name: str
    source: Path
    xyz: np.ndarray                 # (N, 3) float64
    labels: np.ndarray              # 0 stem, 1..n leaves, -2 unlabelled
    date: str | None
    parts: list[dict] = field(default_factory=list)      # one row per input file
    fragments: dict[int, int] = field(default_factory=dict)  # label -> number of pieces (>1 only)
    warnings: list[str] = field(default_factory=list)

    @property
    def labelled(self) -> bool:
        return bool(np.any(self.labels >= 0))

    @property
    def has_stem(self) -> bool:
        return int(np.count_nonzero(self.labels == 0)) > 0

    @property
    def leaf_ids(self) -> list[int]:
        return [int(v) for v in np.unique(self.labels) if v > 0]

    @property
    def utm_like(self) -> bool:
        c = self.xyz.mean(axis=0)
        return abs(c[0]) > 1e5 and abs(c[1]) > 1e6

    def describe(self) -> str:
        lines = [f"GT plant '{self.name}'  date={self.date or 'unknown'}  points={len(self.xyz):,}"]
        if self.labelled:
            lines.append(f"  stem points: {int(np.count_nonzero(self.labels == 0)):,}"
                         + ("" if self.has_stem else "  (NO STEM - stem metrics will be N/A)"))
            for lid in self.leaf_ids:
                pieces = self.fragments.get(lid, 1)
                lines.append(f"  leaf {lid:>2}: {int(np.count_nonzero(self.labels == lid)):>8,} pts"
                             + (f"  ({pieces} pieces merged)" if pieces > 1 else ""))
        else:
            lines.append("  no stem/leaf labels -> only plant matching + plant-level IoU/height")
        lines.append(f"  coordinates: {'UTM (position matching)' if self.utm_like else 'SHIFTED (shape matching fallback)'}")
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class PredPlant:
    plant_id: str
    xyz: np.ndarray
    labels: np.ndarray              # 0 stem, 1..n leaves, -1 unassigned
    base: np.ndarray
    traits: dict | None = None
    leaves: list[dict] | None = None
    merges: list[tuple[int, int]] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Loading ground truth
# ----------------------------------------------------------------------------

_CC_TIMESTAMP = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}h\d{2}_\d{2}_\d{3}$")
_STEM_RE = re.compile(r"(^|[^a-z])stem([^a-z]|$)")
_LEAF_RE = re.compile(r"(?:^|[^a-z])leaf[_\- ]?0*(\d+)")
_PART_RE = re.compile(r"[_\- ](part|frag|piece)[_\- ]?\d+$")


def _label_from_filename(stem: str) -> int | None:
    s = stem.lower()
    if _STEM_RE.search(s):
        return 0
    m = _LEAF_RE.search(s)
    return int(m.group(1)) if m else None


def _clean_name(name: str) -> str:
    name = _CC_TIMESTAMP.sub("", name)
    name = re.sub(r"_asc$", "", name)
    return name


def _export_bin(bin_path: Path, log: Log) -> Path:
    """CloudCompare .bin -> folder of ASC files (cached in outputs/_gt_cache)."""
    from . import reference_structure as ref

    cache = OUTPUT_ROOT / "_gt_cache" / f"{bin_path.stem}_asc"
    if cache.is_dir() and len(list(cache.glob("*.asc"))) >= 1:
        return cache
    log(f"  exporting CloudCompare .bin -> {cache}")
    ref.export_bin_clouds_to_asc(bin_path, cache)
    return cache


def load_ground_truth(path: str | Path, date: str | None = None, log: Log = print,
                      gcfg: GroundTruthConfig | None = None) -> GTPlant:
    gcfg = gcfg or GroundTruthConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    name = _clean_name(path.stem if path.is_file() else path.name)
    source = path
    if path.is_file() and path.suffix.lower() == ".bin":
        path = _export_bin(path, log)

    files = sorted(p for p in path.iterdir() if p.suffix.lower() in POINT_SUFFIXES) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No point cloud files in {path}")

    xyz_parts, lab_parts, parts, warnings = [], [], [], []
    label_files: dict[int, int] = {}
    n_files = len(files)
    for f in files:
        cloud = read_cloud(f)
        n = len(cloud.xyz)
        from_name = _label_from_filename(_clean_name(f.stem))
        if cloud.labels is not None:
            uniq = np.unique(cloud.labels)
            if len(uniq) == 1:
                lab = np.full(n, int(uniq[0]), np.int32)
                how = f"field '{cloud.label_field}' = {int(uniq[0])}"
            else:
                lab = cloud.labels.astype(np.int32)
                how = f"field '{cloud.label_field}' (per point, {len(uniq)} values)"
            if from_name is not None and len(uniq) == 1 and from_name != int(uniq[0]):
                warnings.append(f"{f.name}: name says {from_name} but field says {int(uniq[0])}; field used")
        elif from_name is not None:
            lab = np.full(n, from_name, np.int32)
            how = f"file name -> {'stem' if from_name == 0 else 'leaf ' + str(from_name)}"
        elif n_files == 1:
            lab = np.full(n, gcfg.unlabelled_value, np.int32)
            how = "no labels (plant only)"
        else:
            raise ValueError(
                f"Cannot tell which organ '{f.name}' is. Rename it to ..._stem / ..._leafNN "
                "(..._leafNN_part2 for extra pieces) or add a Classification/label field. "
                "Click the (?) next to the ground-truth loader for the naming rules."
            )
        for v in np.unique(lab):
            if v >= 0 and len(np.unique(lab)) == 1:
                label_files[int(v)] = label_files.get(int(v), 0) + 1
        xyz_parts.append(cloud.xyz)
        lab_parts.append(lab)
        parts.append({"file": f.name, "points": n, "label_source": how})

    xyz = np.vstack(xyz_parts)
    labels = np.concatenate(lab_parts)
    fragments = {k: v for k, v in label_files.items() if v > 1}
    if 0 in fragments:
        warnings.append(f"stem given in {fragments.pop(0)} pieces (merged into one stem)")
    if labels.size and np.any(labels >= 0) and not np.any(labels == 0):
        warnings.append("no stem points in ground truth (stem IoU / stem traits will be N/A)")
    gt_date = date or date_from_name(name) or next((date_from_name(f.name) for f in files if date_from_name(f.name)), None)
    gt = GTPlant(name=name, source=source, xyz=xyz, labels=labels, date=gt_date,
                 parts=parts, fragments=fragments, warnings=warnings)
    return gt


# ----------------------------------------------------------------------------
# Predictions on disk (written by the pipeline, read here)
# ----------------------------------------------------------------------------

INDEX_FIELDS = ["plant_id", "group_id", "base_x", "base_y", "base_z", "n_points", "file"]


def save_predicted_plant(run: RunPaths, date: str, plant: PredPlant) -> Path:
    """Store one predicted plant so the GT checker (and later stages) can find it."""
    pdir = run.plant_dir(date, plant.plant_id)
    f = pdir / f"{plant.plant_id}_labelled.npz"
    np.savez_compressed(f, xyz=plant.xyz, labels=plant.labels, base=plant.base,
                        plant_id=plant.plant_id, origin=np.asarray(PROJECT_ORIGIN))
    if plant.traits is not None:
        with open(pdir / f"{plant.plant_id}_traits.json", "w") as fh:
            json.dump({"traits": plant.traits, "leaves": plant.leaves or [],
                       "merges": plant.merges}, fh, indent=2, default=float)
    idx = run.plants_index(date)
    rows = []
    if idx.exists():
        with open(idx) as fh:
            rows = [r for r in csv.DictReader(fh) if r["plant_id"] != plant.plant_id]
    rows.append({"plant_id": plant.plant_id, "group_id": plant.plant_id.split("_")[0],
                 "base_x": f"{plant.base[0]:.4f}", "base_y": f"{plant.base[1]:.4f}",
                 "base_z": f"{plant.base[2]:.4f}", "n_points": len(plant.xyz),
                 "file": str(f.relative_to(idx.parent))})
    rows.sort(key=lambda r: r["plant_id"])
    with open(idx, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return f


def available_dates(run_root: Path) -> list[str]:
    run_root = Path(run_root)
    return sorted(p.name for p in run_root.iterdir()
                  if p.is_dir() and (p / STAGE_DIRS["plants"] / "plants_index.csv").exists())


def load_predictions(run_root: Path, date: str) -> list[PredPlant]:
    idx = Path(run_root) / date / STAGE_DIRS["plants"] / "plants_index.csv"
    plants = []
    with open(idx) as fh:
        for r in csv.DictReader(fh):
            f = idx.parent / r["file"]
            d = np.load(f, allow_pickle=False)
            p = PredPlant(plant_id=r["plant_id"], xyz=d["xyz"], labels=d["labels"].astype(np.int32),
                          base=d["base"])
            tj = f.parent / f"{r['plant_id']}_traits.json"
            if tj.exists():
                t = json.loads(tj.read_text())
                p.traits, p.leaves = t.get("traits"), t.get("leaves")
                p.merges = [tuple(m) for m in t.get("merges", [])]
            plants.append(p)
    return plants


# ----------------------------------------------------------------------------
# Matching GT plant -> predicted plant
# ----------------------------------------------------------------------------


@dataclass
class MatchResult:
    plant_id: str | None
    method: str
    share: float                     # fraction of GT points on the matched plant
    coverage: float                  # fraction of GT points on ANY predicted plant
    candidates: list[tuple[str, float]]
    transform: np.ndarray | None = None   # 4x4 GT -> prediction frame (ICP only)
    fitness: float = float("nan")
    date: str | None = None


def _gt_base(gt: GTPlant) -> np.ndarray:
    from .structure import find_base_point
    return find_base_point(gt.xyz, gt.labels if gt.labelled else None)


def _sample(xyz: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(xyz) <= n:
        return xyz
    return xyz[np.random.default_rng(seed).choice(len(xyz), n, replace=False)]


def _icp_match(gt: GTPlant, preds: list[PredPlant], gcfg: GroundTruthConfig) -> tuple[int, float, np.ndarray] | None:
    import open3d as o3d

    gb = _gt_base(gt)
    src = _sample(gt.xyz, 6000) - gb
    if gt.utm_like:
        cand = [i for i, p in enumerate(preds) if np.linalg.norm(p.base[:2] - gb[:2]) <= gcfg.icp_search_radius]
    else:
        cand = list(range(len(preds)))
    best = None
    s_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    for i in cand:
        tgt = _sample(preds[i].xyz, 15000) - preds[i].base
        t_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
        for yaw in np.radians(np.arange(0, 360, 30)):
            c, s = math.cos(yaw), math.sin(yaw)
            init = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
            reg = o3d.pipelines.registration.registration_icp(
                s_pcd, t_pcd, gcfg.icp_max_corr * 3, init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            reg = o3d.pipelines.registration.registration_icp(
                s_pcd, t_pcd, gcfg.icp_max_corr, reg.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            if best is None or reg.fitness > best[1]:
                T = np.eye(4)
                T[:3, 3] = -gb
                T = reg.transformation @ T
                T2 = np.eye(4)
                T2[:3, 3] = preds[i].base
                best = (i, float(reg.fitness), T2 @ T)
    return best


def match_ground_truth(gt: GTPlant, preds: list[PredPlant], gcfg: GroundTruthConfig | None = None,
                       allow_icp: bool = True) -> MatchResult:
    gcfg = gcfg or GroundTruthConfig()
    if not preds:
        return MatchResult(None, "no_predictions", 0.0, 0.0, [])
    owner = np.concatenate([np.full(len(p.xyz), i) for i, p in enumerate(preds)])
    tree = cKDTree(np.vstack([p.xyz for p in preds]))
    q = _sample(gt.xyz, 50000)
    d, idx = tree.query(q, k=1, distance_upper_bound=gcfg.match_radius)
    hit = np.isfinite(d)
    votes = np.bincount(owner[idx[hit]], minlength=len(preds))
    shares = votes / max(len(q), 1)
    order = np.argsort(shares)[::-1]
    cands = [(preds[i].plant_id, float(shares[i])) for i in order[:3] if shares[i] > 0]
    coverage = float(hit.mean())
    if shares[order[0]] >= gcfg.min_match_share:
        return MatchResult(preds[order[0]].plant_id, "position_overlap", float(shares[order[0]]), coverage, cands)

    if gt.utm_like:
        gb = _gt_base(gt)
        dist = np.array([np.linalg.norm(p.base[:2] - gb[:2]) for p in preds])
        j = int(np.argmin(dist))
        if dist[j] <= gcfg.base_fallback_radius:
            return MatchResult(preds[j].plant_id, "base_position", float(shares[j]), coverage, cands)

    if allow_icp:
        best = _icp_match(gt, preds, gcfg)
        if best is not None and best[1] >= 0.3:
            i, fit, T = best
            return MatchResult(preds[i].plant_id, "shape_icp", fit, coverage, cands, transform=T, fitness=fit)
    return MatchResult(None, "not_found", float(shares[order[0]]), coverage, cands)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


def _vox(xyz: np.ndarray, v: float, ref) -> np.ndarray:
    """One representative point per voxel (removes density differences)."""
    if not len(xyz):
        return np.empty((0, 3))
    _, idx = np.unique(voxel_keys(xyz, v, ref), return_index=True)
    return xyz[idx]


class _Set:
    """Voxel-thinned point set with a KD-tree for tolerance queries."""

    def __init__(self, pts: np.ndarray):
        self.pts = pts
        self.tree = cKDTree(pts) if len(pts) else None

    def __len__(self):
        return len(self.pts)

    def covered_by(self, other: "_Set", tol: float) -> np.ndarray:
        """Bool per own point: is there a point of `other` within tol?"""
        if not len(self) or other.tree is None:
            return np.zeros(len(self), bool)
        d, _ = other.tree.query(self.pts, k=1, distance_upper_bound=tol)
        return np.isfinite(d)


def _iou(a: _Set, b: _Set, tol: float) -> tuple[float, int, int]:
    """Tolerance IoU: TP = a-points with a b-point within tol, FP = b-points with no a-point within tol."""
    if not len(a) and not len(b):
        return float("nan"), 0, 0
    tp = int(a.covered_by(b, tol).sum())
    fp = int((~b.covered_by(a, tol)).sum())
    union = len(a) + fp
    return (tp / union if union else float("nan")), tp, union


def _share_inside(p: _Set, g: _Set, tol: float) -> float:
    return float(p.covered_by(g, tol).mean()) if len(p) else 0.0


def evaluate_segmentation(gt_xyz: np.ndarray, gt_labels: np.ndarray, pred: PredPlant,
                          gcfg: GroundTruthConfig) -> dict:
    v = gcfg.voxel_size
    ref_pt = np.floor(np.minimum(gt_xyz.min(axis=0), pred.xyz.min(axis=0)))
    tol = gcfg.iou_tolerance
    G, P = _Set(_vox(gt_xyz, v, ref_pt)), _Set(_vox(pred.xyz, v, ref_pt))
    plant_iou, tp, _ = _iou(G, P, tol)
    res = {
        "plant_iou": plant_iou,
        "plant_recall": tp / len(G) if len(G) else float("nan"),
        "plant_precision": float(P.covered_by(G, tol).mean()) if len(P) else float("nan"),
        "rows": [], "fragments": {}, "merged_pred_leaves": [], "pred_to_gt": {},
    }
    labelled = np.any(gt_labels >= 0)
    if not labelled:
        return res

    gl = {int(k): _Set(_vox(gt_xyz[gt_labels == k], v, ref_pt)) for k in np.unique(gt_labels) if k >= 0}
    pl = {int(k): _Set(_vox(pred.xyz[pred.labels == k], v, ref_pt)) for k in np.unique(pred.labels) if k >= 0}
    empty = _Set(np.empty((0, 3)))

    # stem
    if 0 in gl:
        s_iou, _, _ = _iou(gl[0], pl.get(0, empty), tol)
        res["rows"].append({"gt_id": 0, "pred_id": 0 if 0 in pl else "", "name": "stem",
                            "iou": s_iou, "iou_with_fragments": s_iou,
                            "gt_voxels": len(gl[0]), "pred_voxels": len(pl.get(0, empty)), "n_fragments": 0})
        res["pred_to_gt"][0] = 0
    res["stem_iou"] = res["rows"][0]["iou"] if 0 in gl else float("nan")

    g_ids = sorted(k for k in gl if k > 0)
    p_ids = sorted(k for k in pl if k > 0)
    M = np.zeros((len(g_ids), len(p_ids)))
    for a, gi in enumerate(g_ids):
        for b, pi in enumerate(p_ids):
            M[a, b] = _iou(gl[gi], pl[pi], tol)[0]
    pairs = {}
    if M.size:
        r, c = linear_sum_assignment(-M)
        pairs = {g_ids[a]: p_ids[b] for a, b in zip(r, c) if M[a, b] >= gcfg.min_leaf_iou}

    # fragment-aware: unmatched predicted pieces mostly inside one GT leaf belong to it
    used = set(pairs.values())
    frags: dict[int, list[int]] = {}
    for b, pi in enumerate(p_ids):
        if pi in used or not len(pl[pi]):
            continue
        share = [_share_inside(pl[pi], gl[gi], tol) for gi in g_ids]
        if share and max(share) >= gcfg.fragment_min_share:
            frags.setdefault(g_ids[int(np.argmax(share))], []).append(pi)
    # under-segmentation: one predicted leaf covering several GT leaves
    for pi in p_ids:
        covered = [gi for gi in g_ids if _share_inside(gl[gi], pl[pi], tol) >= 0.25]
        if len(covered) > 1:
            res["merged_pred_leaves"].append({"pred_id": pi, "gt_ids": covered})

    leaf_ious, leaf_ious_frag = [], []
    for gi in g_ids:
        pi = pairs.get(gi)
        base_iou = M[g_ids.index(gi), p_ids.index(pi)] if pi is not None else 0.0
        members = ([pi] if pi is not None else []) + frags.get(gi, [])
        if members:
            union_set = _Set(np.vstack([pl[m].pts for m in members]))
            f_iou = _iou(gl[gi], union_set, tol)[0]
        else:
            f_iou = 0.0
        for m in members:
            res["pred_to_gt"][m] = gi
        leaf_ious.append(base_iou)
        leaf_ious_frag.append(f_iou)
        res["rows"].append({"gt_id": gi, "pred_id": "" if pi is None else pi, "name": f"leaf_{gi}",
                            "iou": base_iou, "iou_with_fragments": f_iou,
                            "gt_voxels": len(gl[gi]), "pred_voxels": len(pl[pi]) if pi is not None else 0,
                            "n_fragments": len(frags.get(gi, []))})
    for pi in p_ids:
        if pi not in res["pred_to_gt"]:
            res["rows"].append({"gt_id": "", "pred_id": pi, "name": f"pred_leaf_{pi}_unmatched",
                                "iou": 0.0, "iou_with_fragments": 0.0, "gt_voxels": 0,
                                "pred_voxels": len(pl[pi]), "n_fragments": 0})
    res["fragments"] = frags
    res["leaf_pairs"] = pairs
    res["mean_leaf_iou"] = float(np.mean(leaf_ious)) if leaf_ious else float("nan")
    res["mean_leaf_iou_with_fragments"] = float(np.mean(leaf_ious_frag)) if leaf_ious_frag else float("nan")
    organ = ([res["stem_iou"]] if 0 in gl else []) + leaf_ious_frag
    res["mean_iou"] = float(np.mean(organ)) if organ else float("nan")
    res["n_gt_leaves"], res["n_pred_leaves"] = len(g_ids), len(p_ids)
    return res


PLANT_TRAITS = [
    ("plant_height_m", "m"), ("stem_length_m", "m"), ("stem_length_to_top_leaf_m", "m"),
    ("stem_radius_m", "m"), ("n_leaves", "count"), ("total_leaf_area_m2", "m2"),
    ("mean_leaf_length_m", "m"), ("mean_leaf_angle_to_stem_deg", "deg"),
]
LEAF_TRAITS = [
    ("leaf_length_m", "m"), ("leaf_max_width_m", "m"), ("leaf_area_m2", "m2"),
    ("leaf_angle_to_stem_deg", "deg"), ("leaf_inclination_from_horizontal_deg", "deg"),
    ("insertion_height_m", "m"),
]


def _err_row(trait, unit, g, p) -> dict:
    ok = g is not None and p is not None and np.isfinite(g) and np.isfinite(p)
    return {"trait": trait, "unit": unit, "ground_truth": g, "predicted": p,
            "abs_error": abs(p - g) if ok else float("nan"),
            "rel_error_pct": abs(p - g) / abs(g) * 100 if ok and abs(g) > 1e-12 else float("nan")}


def compare_traits(gt_struct, pred: PredPlant, seg: dict) -> tuple[list[dict], list[dict]]:
    from .structure import structure_from_labels

    if pred.traits is None:
        ps = structure_from_labels(pred.xyz, pred.labels, base_point=pred.base)
        pred.traits, pred.leaves, pred.merges = ps.traits, ps.leaves, ps.merges
    plant_rows = [_err_row(t, u, gt_struct.traits.get(t), pred.traits.get(t)) for t, u in PLANT_TRAITS]
    # predicted leaves merged by the structure step are found under their host label
    host = {m: k for k, m in pred.merges}
    p_by_label = {lf["leaf_label"]: lf for lf in (pred.leaves or [])}
    leaf_rows = []
    for gl in gt_struct.leaves:
        gid = gl["leaf_label"]
        pid = seg.get("leaf_pairs", {}).get(gid)
        if pid is None:
            continue
        pl = p_by_label.get(host.get(pid, pid))
        if pl is None:
            continue
        for t, u in LEAF_TRAITS:
            r = _err_row(t, u, gl.get(t), pl.get(t))
            r.update(gt_leaf=gid, pred_leaf=pid)
            leaf_rows.append(r)
    return plant_rows, leaf_rows


# ----------------------------------------------------------------------------
# Figures and files
# ----------------------------------------------------------------------------


def _label_colors(labels: np.ndarray) -> np.ndarray:
    from . import reference_structure as ref

    lut = ref._COLOR_LUT
    lab = np.asarray(labels)
    out = np.full((len(lab), 3), 0.6)
    ok = (lab >= 0) & (lab < len(lut))
    out[ok] = lut[lab[ok]]
    return out


def save_comparison_figure(path: Path, gt_xyz, gt_labels, pred: PredPlant, seg: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    remap = np.full(len(pred.labels), -1)
    for p_id, g_id in seg.get("pred_to_gt", {}).items():
        remap[pred.labels == p_id] = g_id
    c0 = gt_xyz.mean(axis=0)
    rng = np.random.default_rng(0)
    gi = rng.choice(len(gt_xyz), min(len(gt_xyz), 25000), replace=False)
    pi = rng.choice(len(pred.xyz), min(len(pred.xyz), 25000), replace=False)
    fig = plt.figure(figsize=(15, 6))
    for k, (pts, cols, ttl) in enumerate([
        (gt_xyz[gi] - c0, _label_colors(gt_labels[gi]), "Ground truth"),
        (pred.xyz[pi] - c0, _label_colors(remap[pi]), f"Prediction ({pred.plant_id})\ncolours = matched GT ids, grey = unmatched"),
    ]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.4)
        ax.set_title(ttl, fontsize=9)
        ax.set_box_aspect(np.ptp(np.vstack([gt_xyz[gi] - c0]), axis=0) + 1e-3)
    # error map: GT points with / without a predicted point of the same plant nearby
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    d, _ = cKDTree(pred.xyz).query(gt_xyz[gi], k=1)
    d2, _ = cKDTree(gt_xyz).query(pred.xyz[pi], k=1)
    p = gt_xyz[gi] - c0
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=np.where(d[:, None] < 0.01, [[0.2, 0.7, 0.2]], [[0.85, 0.1, 0.1]]), s=0.4)
    q = pred.xyz[pi][d2 >= 0.01] - c0
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=[[0.1, 0.3, 0.9]], s=0.4)
    ax.set_title("green = found, red = missed (GT only)\nblue = extra (prediction only)", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------


def check_one(gt: GTPlant, run: RunPaths, preds_by_date: dict[str, list[PredPlant]],
              gcfg: GroundTruthConfig, log: Log = print) -> dict:
    from .structure import structure_from_labels

    dates = [gt.date] if gt.date in preds_by_date else list(preds_by_date)
    if gt.date and gt.date not in preds_by_date:
        log(f"  ! GT date {gt.date} has no predictions in this run; trying all dates {dates}")
    best: MatchResult | None = None
    for d in dates:
        m = match_ground_truth(gt, preds_by_date[d], gcfg)
        m.date = d
        if best is None or (m.plant_id and (best.plant_id is None or m.share > best.share)):
            best = m
    summary = {"gt_name": gt.name, "gt_date": gt.date or "", "matched_date": best.date if best else "",
               "matched_plant": best.plant_id if best else "", "match_method": best.method if best else "none",
               "match_share": best.share if best else 0.0, "gt_coverage_by_predictions": best.coverage if best else 0.0,
               "warnings": "; ".join(gt.warnings)}
    out = run.gt_dir(gt.name)
    (out / "gt_description.txt").write_text(gt.describe())
    if best is None or best.plant_id is None:
        log(f"  x no predicted plant found for '{gt.name}' (best share {summary['match_share']:.2f})")
        _write_csv(out / "match.csv", [summary])
        return summary

    pred = next(p for p in preds_by_date[best.date] if p.plant_id == best.plant_id)
    gt_xyz = gt.xyz
    if best.transform is not None:
        gt_xyz = (best.transform[:3, :3] @ gt.xyz.T).T + best.transform[:3, 3]
        log(f"  ! shifted coordinates: GT registered to {pred.plant_id} by ICP (fitness {best.fitness:.2f})")
    log(f"  -> matched to {best.date}/{pred.plant_id} by {best.method} "
        f"(share {best.share:.2f}; candidates {[(c, round(s, 2)) for c, s in best.candidates]})")
    if len(best.candidates) > 1 and best.candidates[1][1] > 0.15:
        log(f"  ! GT also overlaps {best.candidates[1][0]} ({best.candidates[1][1]:.0%}) "
            "-> plant separation between these two is not clean")

    seg = evaluate_segmentation(gt_xyz, gt.labels, pred, gcfg)
    summary.update({k: seg.get(k, float("nan")) for k in
                    ("plant_iou", "plant_recall", "plant_precision", "stem_iou", "mean_leaf_iou",
                     "mean_leaf_iou_with_fragments", "mean_iou", "n_gt_leaves", "n_pred_leaves")})
    summary["n_pred_fragments_merged"] = sum(len(v) for v in seg.get("fragments", {}).values())
    summary["n_pred_leaves_covering_several_gt_leaves"] = len(seg.get("merged_pred_leaves", []))

    plant_rows, leaf_rows = [], []
    if gt.labelled:
        gts = structure_from_labels(gt_xyz, gt.labels, merge_fragments=False)
        plant_rows, leaf_rows = compare_traits(gts, pred, seg)
        for r in plant_rows:
            summary[f"err_{r['trait']}"] = r["abs_error"]
        if leaf_rows:
            for t, _ in LEAF_TRAITS:
                e = [r["abs_error"] for r in leaf_rows if r["trait"] == t and np.isfinite(r["abs_error"])]
                summary[f"mae_{t}"] = float(np.mean(e)) if e else float("nan")
        summary["gt_stem_status"] = gts.traits["stem_status"]
    else:
        g_h = float(gt_xyz[:, 2].max() - _gt_base(gt)[2])
        p_h = float(pred.xyz[:, 2].max() - pred.base[2])
        plant_rows = [_err_row("plant_height_m", "m", g_h, p_h)]
        summary["err_plant_height_m"] = plant_rows[0]["abs_error"]

    remap = np.full(len(pred.labels), -1, np.int32)
    for p_id, g_id in seg.get("pred_to_gt", {}).items():
        remap[pred.labels == p_id] = g_id
    np.savez_compressed(out / "viewer_data.npz", gt_xyz=gt_xyz, gt_labels=gt.labels,
                        pred_xyz=pred.xyz, pred_labels=remap, pred_id=pred.plant_id)
    _write_csv(out / "match.csv", [summary])
    _write_csv(out / "iou_per_class.csv", seg["rows"])
    _write_csv(out / "trait_comparison_plant.csv", plant_rows)
    _write_csv(out / "trait_comparison_leaf.csv", leaf_rows)
    try:
        save_comparison_figure(out / "comparison.png", gt_xyz, gt.labels, pred, seg,
                               f"{gt.name}  vs  {best.date}/{pred.plant_id}  |  plant IoU {seg['plant_iou']:.2f}"
                               + (f"  mean organ IoU {seg.get('mean_iou', float('nan')):.2f}" if gt.labelled else ""))
    except Exception as e:  # figure is optional
        log(f"  ! figure not saved: {e}")
    log(f"  plant IoU {seg['plant_iou']:.3f} | stem IoU {seg.get('stem_iou', float('nan')):.3f} | "
        f"mean leaf IoU {seg.get('mean_leaf_iou_with_fragments', float('nan')):.3f}")
    return summary


def run_ground_truth_check(run_root: str | Path, gt_items: list[tuple[str | Path, str | None]],
                           gcfg: GroundTruthConfig | None = None, log: Log = print,
                           progress=None) -> list[dict]:
    """gt_items: list of (path to GT folder/file, date or None).
    progress: optional callback(ProgressInfo) for the UI timer."""
    from .progress import Progress
    gcfg = gcfg or GroundTruthConfig()
    prog = Progress({"load predictions": 1.0, "ground truth": 4.0 * max(1, len(gt_items))}, progress)
    run = RunPaths(Path(run_root))
    dates = available_dates(run.root)
    if not dates:
        raise FileNotFoundError(f"No predicted plants in {run.root} (expected <date>/03_plants/plants_index.csv)")
    preds_by_date = {d: load_predictions(run.root, d) for d in dates}
    log(f"Predictions found for dates: {', '.join(f'{d} ({len(p)} plants)' for d, p in preds_by_date.items())}")
    prog.finish_stage("load predictions")
    summaries = []
    for k, (path, date) in enumerate(gt_items):
        prog.update("ground truth", k, len(gt_items), f"ground truth {k + 1}/{len(gt_items)}")
        log(f"\nGround truth: {path}")
        try:
            gt = load_ground_truth(path, date, log, gcfg)
            log(gt.describe())
            summaries.append(check_one(gt, run, preds_by_date, gcfg, log))
        except Exception as e:
            log(f"  x failed: {e}")
            summaries.append({"gt_name": Path(path).name, "match_method": "error", "warnings": str(e)})
    gt_root = run.run_dir("ground_truth")
    _write_csv(gt_root / "gt_summary.csv", summaries)
    (gt_root / "gt_config.json").write_text(json.dumps(gcfg.to_dict(), indent=2))
    log(f"\nGround-truth summary -> {gt_root / 'gt_summary.csv'}  ({prog.finish():.1f} s)")
    return summaries
