"""The user's own scripts, used unchanged by the pipeline.

  soil      : maize_soil_segmentation_main/soil_segmentation/classify_plants.py
  structure : Research_project/copy_of_New_optimized_plant_structure.py
              (falls back to the identical copy pipeline/reference_structure.py)

Both files are imported from where they are, so a change of a parameter in
those scripts is used by the next run. This module only converts their
results to the pipeline's format (labels per point, traits dict, leaf list),
and adds one check of the leaves after the structure script (check_leaves):
leaf pieces that continue another leaf are joined to it, and pieces that do
not reach the stem (parts of neighbouring plants, blobs on the soil) are not
counted as leaves of this plant. The measurements are then repeated with the
script's own functions (steps 6-10). structure_output/ keeps the script's
original files; *_after_leaf_check.csv holds the checked measurements.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
SOIL_SCRIPT = _HERE.parent / "maize_soil_segmentation_main" / "soil_segmentation" / "classify_plants.py"
STRUCTURE_SCRIPT = _HERE.parent.parent / "Research_project" / "copy_of_New_optimized_plant_structure.py"
STRUCTURE_FALLBACK = _HERE / "reference_structure.py"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # needed for @dataclass in the loaded file
    spec.loader.exec_module(mod)
    return mod


def soil_module():
    """classify_plants.py (functions + its tuned constants)."""
    return _load("user_classify_plants", SOIL_SCRIPT)


def structure_module():
    """copy_of_New_optimized_plant_structure.py."""
    path = STRUCTURE_SCRIPT if STRUCTURE_SCRIPT.exists() else STRUCTURE_FALLBACK
    return _load("user_plant_structure", path)


def structure_script_path() -> Path:
    return STRUCTURE_SCRIPT if STRUCTURE_SCRIPT.exists() else STRUCTURE_FALLBACK


# ----------------------------------------------------------------------------
# plant structure with the user's code
# ----------------------------------------------------------------------------

def check_leaves(ref, res, cfg, scfg) -> tuple[list[np.ndarray], list[tuple[int, int]], list[int], list[np.ndarray]]:
    """Leaf check on the script's leaf clusters (work cloud, centred).

    1. A cluster that comes within leaf_touch_stem_points of the stem points
       is a leaf of this plant.
    2. A cluster that does not reach the stem is joined to the leaf it
       continues (merge_leaf_fragments: small gap, inner end touching the
       outer part of the leaf, same direction), but only to a leaf from 1.
    3. The rest is not counted (neighbouring plants, blobs on the soil).

    Returns the kept clusters, merges (kept id, merged id), the original id
    (1..n) of each kept cluster, and the point arrays of the pieces not counted.
    """
    from .structure import merge_leaf_fragments
    clusters = list(res.leaf_clusters.clusters)
    ids = list(range(1, len(clusters) + 1))
    axis = np.asarray(res.smoothed_stem)
    stem_pts = res.segmentation.stem_points
    if len(stem_pts) >= scfg.leaf_check_min_stem_points:
        tree, limit = cKDTree(stem_pts), scfg.leaf_touch_stem_points
    elif len(axis):
        tree, limit = cKDTree(axis), scfg.leaf_attach_dist
    else:
        return clusters, [], ids, []
    touch = [float(tree.query(c, k=1)[0].min()) <= limit for c in clusters]
    kept = [c for c, t in zip(clusters, touch) if t]
    kept_ids = [i for i, t in zip(ids, touch) if t]
    loose = [(i, c) for i, c, t in zip(ids, clusters, touch) if not t]
    merges, dropped = [], []
    for pid_, piece in sorted(loose, key=lambda t: float(tree.query(t[1], k=1)[0].min())):  # nearest first
        if kept and len(axis):
            new, new_ids, mg = merge_leaf_fragments(kept + [piece], kept_ids + [pid_], axis, scfg)
            if len(mg) == 1 and mg[0][1] == pid_:
                kept, kept_ids = new, new_ids
                merges.append(mg[0])
                continue
        dropped.append(piece)
    return kept, merges, kept_ids, dropped


def _work_labels(ref, res, cfg, clusters=None, dropped_pts=None) -> np.ndarray:
    """Labels on the user's work cloud: 0 = stem, 1..n = leaf cluster, -1 = not counted.
    Same rule as classify_plant_for_iou() in the user's script."""
    prep, seg = res.preprocess, res.segmentation
    clusters = res.leaf_clusters.clusters if clusters is None else clusters
    n = len(prep.xyz_work)
    pred = np.full(n, -1, np.int32)
    pred[seg.stem_mask] = 0
    leaf_idx = np.where(seg.leaf_mask)[0]
    if leaf_idx.size and clusters:
        pts = prep.xyz_work[leaf_idx]
        labs = np.full(len(leaf_idx), -1, np.int32)
        done = np.zeros(len(leaf_idx), bool)
        eps = max(cfg.leaf_cluster_eps, cfg.voxel_size * 2.5)
        if dropped_pts is not None and len(dropped_pts):     # pieces not counted stay -1
            d, _ = cKDTree(dropped_pts).query(pts, k=1, workers=-1)
            done |= d < 1e-9
        dist = np.full((len(clusters), len(leaf_idx)), np.inf)
        for cid, c in enumerate(clusters):
            d, _ = cKDTree(c).query(pts, k=1, workers=-1)
            dist[cid] = d
            take = (d <= eps) & ~done
            labs[take] = cid + 1
            done[take] = True
        rem = ~done
        if rem.any():
            labs[rem] = np.argmin(dist[:, rem], axis=0) + 1
        pred[leaf_idx] = labs
    # script rule: every unlabelled point takes the nearest label; with the leaf check, dropped pieces stay -1
    un = (pred < 0) & ~seg.leaf_mask if dropped_pts is not None else pred < 0
    if un.any() and (pred >= 0).any():
        _, nn = cKDTree(prep.xyz_work[pred >= 0]).query(prep.xyz_work[un], k=1, workers=-1)
        pred[un] = pred[pred >= 0][nn]
    elif un.any():
        pred[un] = 0
    return pred


def _remeasure(ref, res, clusters, cfg):
    """Steps 6-10 of the user's script on the checked leaf clusters."""
    stem_sk = res.stem_skeleton
    junctions, j_idx = ref.find_leaf_junctions(stem_sk, clusters)
    leaf_sk = ref.extract_all_leaf_skeletons(clusters, junctions, j_idx, cfg)
    sm_stem, sm_leaves = ref.smooth_skeletons(stem_sk.points, leaf_sk.paths, cfg)
    skel = ref.build_plant_skeleton_graph(sm_stem, leaf_sk)
    m = ref.compute_measurements(sm_stem, leaf_sk, skel, sm_leaves, plant_xyz=res.preprocess.xyz_clean,
                                 leaf_clusters=clusters, cfg=cfg)
    return leaf_sk, sm_stem, sm_leaves, m


def scene_arrays(res) -> dict:
    """Everything the script's _populate_structure_scene() needs, as plain arrays
    (centred frame of the script). Saved per plant so the app can draw the
    script's own viewer without running the structure again."""
    from dataclasses import asdict

    def cat(parts):
        parts = [np.asarray(q, np.float32).reshape(-1, 3) for q in parts]
        return (np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)), np.array([len(q) for q in parts], int)
    lc, lc_n = cat(res.leaf_clusters.clusters)
    lp, lp_n = cat(res.smoothed_leaf_paths)
    m = {k: np.asarray(v, float) for k, v in asdict(res.measurements).items()}
    return {"xyz_clean": res.preprocess.xyz_clean.astype(np.float32),
            "stem_points": np.asarray(res.segmentation.stem_points, np.float32),
            "stem_xy": np.asarray(res.segmentation.stem_xy, float),
            "leaf_clusters": lc, "leaf_cluster_sizes": lc_n,
            "smoothed_stem": np.asarray(res.smoothed_stem, float),
            "leaf_paths": lp, "leaf_path_sizes": lp_n,
            "junctions": np.asarray(res.leaf_skeleton.junctions, float).reshape(-1, 3),
            "tips": np.asarray(res.leaf_skeleton.tips, float).reshape(-1, 3),
            **{f"m_{k}": v for k, v in m.items()}}


def scene_from_arrays(d):
    """(result, prep) objects with the fields _populate_structure_scene() reads."""
    from types import SimpleNamespace as NS

    def split(a, n):
        return [np.asarray(q, float) for q in np.split(np.asarray(a), np.cumsum(n)[:-1])] if len(n) else []
    meas = NS(**{k[2:]: (d[k].tolist() if d[k].ndim else float(d[k])) for k in d if k.startswith("m_")})
    res = NS(segmentation=NS(stem_points=np.asarray(d["stem_points"], float), stem_xy=d["stem_xy"]),
             leaf_clusters=NS(clusters=split(d["leaf_clusters"], d["leaf_cluster_sizes"])),
             smoothed_stem=d["smoothed_stem"], smoothed_leaf_paths=split(d["leaf_paths"], d["leaf_path_sizes"]),
             leaf_skeleton=NS(junctions=d["junctions"], tips=d["tips"]), measurements=meas)
    return res, NS(xyz_clean=np.asarray(d["xyz_clean"], float))


def run_user_structure(xyz: np.ndarray, out_dir: Path | None = None, leaf_check: bool = True) -> dict:
    """Run the user's structure pipeline (run_structure_from_xyz) on one plant.

    xyz: points of the plant (local frame, float64). Returns labels for every
    input point, the plant traits, one dict per leaf, and the skeleton
    (stem centreline + leaf midribs) in the same frame as xyz.
    """
    from .structure import stem_radius
    from .config import StructureConfig

    ref = structure_module()
    cfg = ref.PipelineConfig()
    if out_dir is not None:
        cfg.output_dir = Path(out_dir)
    xyz = np.asarray(xyz, np.float64)
    centre = xyz.mean(axis=0)          # the user's function centres the cloud the same way
    with contextlib.redirect_stdout(io.StringIO()) as log:
        res = ref.run_structure_from_xyz(xyz, cfg, show_o3d_steps=False, show_pyvista=False)
    scfg = StructureConfig()
    if leaf_check:
        clusters, merges, kept_ids, dropped = check_leaves(ref, res, cfg, scfg)
        dropped_pts = np.vstack(dropped) if dropped else None
        leaf_sk, sm_stem, sm_leaves, m = _remeasure(ref, res, clusters, cfg)
        if out_dir is not None:
            ref.export_leaf_measurements_csv(m, Path(out_dir) / "leaf_measurements_after_leaf_check.csv")
            ref.export_measurements_csv(m, Path(out_dir) / "measurements_after_leaf_check.csv")
    else:
        clusters, merges, kept_ids, dropped, dropped_pts = res.leaf_clusters.clusters, [], None, [], None
        leaf_sk, sm_stem, sm_leaves, m = res.leaf_skeleton, res.smoothed_stem, res.smoothed_leaf_paths, res.measurements

    work = res.preprocess.xyz_work + centre
    labels = ref.propagate_labels(work, _work_labels(ref, res, cfg, clusters, dropped_pts), xyz)

    sm_stem = np.asarray(sm_stem) + centre
    leaves = []
    for i, ci in enumerate(leaf_sk.cluster_indices):
        leaves.append({
            "leaf_label": int(ci) + 1,
            "leaf_length_m": float(m.leaf_lengths_m[i]),
            "leaf_max_width_m": float(m.leaf_widths_m[i]) if i < len(m.leaf_widths_m) else float("nan"),
            "leaf_mean_width_m": float(m.leaf_mean_widths_m[i]) if i < len(m.leaf_mean_widths_m) else float("nan"),
            "leaf_area_m2": float(m.leaf_areas_m2[i]) if i < len(m.leaf_areas_m2) else float("nan"),
            "leaf_angle_to_stem_deg": float(m.leaf_angles_deg[i]),
            "insertion_height_m": float(m.insertion_heights_m[i]),
            "n_points": int(len(clusters[ci])),
        })
    if len(leaf_sk.junction_stem_indices):
        top = int(np.max(leaf_sk.junction_stem_indices))
        stem_to_top_leaf = ref._path_length(sm_stem[: top + 1])
    else:
        stem_to_top_leaf = float("nan")

    # stem radius: the user's script does not measure it; median of circle fits
    # per height slice on the stem points (agreed: median along the stem)
    rad = stem_radius(xyz[labels == 0], scfg)
    traits = {
        "method": "user structure code (" + structure_script_path().name + ")" + (" + leaf check" if leaf_check else ""),
        "plant_height_m": float(m.plant_height_m),
        "plant_bottom_z_local": float(m.plant_bottom_z_m + centre[2]),
        "stem_length_m": float(m.stem_length_m),
        "stem_length_to_top_leaf_m": float(stem_to_top_leaf),
        "stem_status": "stem_points",
        "n_stem_points": int((labels == 0).sum()),
        **rad,
        "stem_diameter_mm": rad["stem_radius_m"] * 2000.0 if np.isfinite(rad["stem_radius_m"]) else float("nan"),
        "n_leaves": int(m.n_leaves),
        "total_leaf_area_m2": float(sum(m.leaf_areas_m2)),
        "mean_leaf_length_m": float(np.mean(m.leaf_lengths_m)) if m.leaf_lengths_m else float("nan"),
        "mean_leaf_angle_to_stem_deg": float(np.mean(m.leaf_angles_deg)) if m.leaf_angles_deg else float("nan"),
        "mean_leaf_width_m": float(np.mean(m.leaf_widths_m)) if m.leaf_widths_m else float("nan"),
        "max_leaf_width_m": float(np.max(m.leaf_widths_m)) if m.leaf_widths_m else float("nan"),
        "total_skeleton_length_m": float(m.total_skeleton_length_m),
    }
    if leaf_check:
        traits.update(n_leaf_pieces_from_script=int(len(res.leaf_clusters.clusters)),
                      n_leaf_pieces_joined=int(len(merges)), n_pieces_not_reaching_stem=int(len(dropped)),
                      n_points_not_counted=int((labels < 0).sum()))
    return {
        "labels": labels.astype(np.int32), "traits": traits, "leaves": leaves,
        "centerline": sm_stem,
        "leaf_paths": [np.asarray(p) + centre for p in sm_leaves],
        "merges": merges,
        "leaf_labels": [int(ci) + 1 for ci in leaf_sk.cluster_indices],
        "log": log.getvalue(),
        "scene": scene_arrays(res) if not leaf_check else None,
    }
