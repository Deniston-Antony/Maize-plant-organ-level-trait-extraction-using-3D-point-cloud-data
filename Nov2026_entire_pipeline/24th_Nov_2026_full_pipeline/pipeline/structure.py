"""Plant structure and traits from an already labelled plant cloud.

Input: points of ONE plant (UTM, float64) and a label per point
(0 = stem, 1..n = leaf id, negative = unlabelled). Used both for ground
truth (labels drawn by hand) and for predicted plants (labels from the
automatic segmentation), so both are measured with exactly the same code.

Built on the functions of the reference script (pipeline/reference_structure.py,
a copy of Research_project/copy_of_New_optimized_plant_structure.py), plus:
  * stem axis rebuilt from leaf bases when a plant has no / very few stem points
  * a stem that is only partly visible is extended up through the leaf bases
  * leaf pieces that belong to one leaf are merged (gap + direction test)
  * stem radius from circle fits per height slice (robust to partial arcs)
  * leaf inclination from horizontal in addition to the angle to the stem
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from . import reference_structure as ref
from .config import StructureConfig
from .io_utils import voxel_keys


@dataclass
class PlantStructure:
    traits: dict                                   # plant-level traits
    leaves: list[dict]                             # one dict per measured leaf
    centerline: np.ndarray                         # (M, 3) smoothed stem axis, UTM
    leaf_paths: list[np.ndarray]                   # midrib per measured leaf, UTM
    leaf_labels: list[int]                         # label of each measured leaf
    stem_status: str                               # 'stem_points', 'stem_points+extended', 'rebuilt_from_leaf_bases', 'no_stem'
    merges: list[tuple[int, int]] = field(default_factory=list)  # (kept_label, merged_label)
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def voxel_downsample_labelled(xyz: np.ndarray, labels: np.ndarray, voxel: float):
    keys = voxel_keys(xyz, voxel)
    _, idx = np.unique(keys, return_index=True)
    idx.sort()
    return xyz[idx], labels[idx]


def _leaf_convergence_xy(xyz: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """Stem position (XY) without stem points: leaves radiate from the stem, so
    their inner ends converge there. Iterate: centre -> inner end of every leaf -> median."""
    leaf_ids = [v for v in np.unique(labels) if v > 0]
    if len(leaf_ids) < 2:
        return None
    leaves = [xyz[labels == v][:, :2] for v in leaf_ids]
    centre = np.median(np.vstack(leaves), axis=0)
    for _ in range(4):
        ends = []
        for c in leaves:
            d = np.linalg.norm(c - centre, axis=1)
            k = max(3, int(0.05 * len(c)))
            ends.append(c[np.argsort(d)[:k]].mean(axis=0))
        centre = np.median(np.asarray(ends), axis=0)
    return centre


def find_base_point(xyz: np.ndarray, labels: np.ndarray | None, given=None) -> np.ndarray:
    """Plant base = given point (from stem-base detection), else the lowest stem points,
    else (no stem) below the point where the leaves converge, else the lowest point."""
    if given is not None:
        return np.asarray(given, dtype=np.float64)
    if labels is not None and np.count_nonzero(labels == 0) >= 5:
        pts = xyz[labels == 0]
        z0 = pts[:, 2].min()
        low = pts[pts[:, 2] <= z0 + 0.01]
        return np.array([low[:, 0].mean(), low[:, 1].mean(), z0])
    if labels is not None:
        cxy = _leaf_convergence_xy(xyz, labels)
        if cxy is not None:
            near = np.linalg.norm(xyz[:, :2] - cxy, axis=1) <= 0.04
            z0 = float(xyz[near, 2].min()) if near.any() else float(xyz[:, 2].min())
            return np.array([cxy[0], cxy[1], z0])
    z0 = xyz[:, 2].min()
    low = xyz[xyz[:, 2] <= z0 + 0.01]
    return np.array([low[:, 0].mean(), low[:, 1].mean(), z0])


def _leaf_roots(clusters: list[np.ndarray], axis_xy: np.ndarray, max_dist: float) -> np.ndarray:
    """For each leaf, the centre of its points closest (horizontally) to the stem axis."""
    roots = []
    for c in clusters:
        d = np.linalg.norm(c[:, :2] - axis_xy, axis=1)
        k = max(3, int(0.05 * len(c)))
        near = np.argsort(d)[:k]
        if d[near].mean() <= max_dist:
            roots.append(c[near].mean(axis=0))
    return np.asarray(roots).reshape(-1, 3)


def _polyline_resample(points: np.ndarray, step: float) -> np.ndarray:
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return points[:1]
    s_new = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(s_new, s, points[:, d]) for d in range(3)])


def build_stem_axis(
    stem_pts: np.ndarray, clusters: list[np.ndarray], base: np.ndarray,
    cfg: "ref.PipelineConfig", scfg: StructureConfig,
) -> tuple[np.ndarray, str]:
    """Stem centerline from bottom to top, with fallbacks for missing stem points."""
    status = "stem_points"
    if len(stem_pts) >= scfg.min_stem_points:
        try:
            line = ref.extract_stem_centerline(SimpleNamespace(stem_points=stem_pts), cfg).points
        except RuntimeError:
            line = np.empty((0, 3))
    else:
        line = np.empty((0, 3))

    if len(line) < 2:
        # No usable stem: chain base -> leaf roots (sorted by height).
        roots = _leaf_roots(clusters, base[:2], max_dist=0.08)
        roots = roots[roots[:, 2] > base[2] + 0.005] if len(roots) else roots
        if len(roots) == 0:
            return np.vstack([base, base + [0, 0, 0.005]]), "no_stem"
        roots = roots[np.argsort(roots[:, 2])]
        line = np.vstack([base, roots])
        status = "rebuilt_from_leaf_bases"
    else:
        # Start the axis at the plant base.
        if line[0, 2] - base[2] > 0.01:
            line = np.vstack([base, line])
        # Extend a partly visible stem up through leaf roots above its top.
        top = line[-1]
        roots = _leaf_roots(clusters, top[:2], max_dist=0.06)
        above = roots[roots[:, 2] > top[2] + 0.03] if len(roots) else roots
        if len(above):
            line = np.vstack([line, above[np.argsort(above[:, 2])]])
            status = "stem_points+extended"

    # z must increase monotonically along the axis
    for i in range(1, len(line)):
        if line[i, 2] < line[i - 1, 2]:
            line[i, 2] = line[i - 1, 2] + 1e-6
    return _polyline_resample(line, cfg.stem_z_step), status


def _fit_circle(xy: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit with one outlier-rejection pass. Returns cx, cy, r, rms."""
    def kasa(p):
        m = p.mean(axis=0)
        q = p - m
        A = np.column_stack([q[:, 0], q[:, 1], np.ones(len(q))])
        b = -(q[:, 0] ** 2 + q[:, 1] ** 2)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -sol[0] / 2, -sol[1] / 2
        r2 = cx ** 2 + cy ** 2 - sol[2]
        return cx + m[0], cy + m[1], math.sqrt(r2) if r2 > 0 else float("nan")

    cx, cy, r = kasa(xy)
    if not np.isfinite(r):
        return cx, cy, r, float("nan")
    res = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r)
    mad = np.median(res) + 1e-9
    keep = res <= 3.0 * mad + 0.001
    if keep.sum() >= 6:
        cx, cy, r = kasa(xy[keep])
        res = np.abs(np.linalg.norm(xy[keep] - [cx, cy], axis=1) - r)
    return cx, cy, r, float(np.sqrt(np.mean(res ** 2)))


def stem_radius(stem_pts: np.ndarray, scfg: StructureConfig) -> dict:
    """Median stem radius over height slices with a trustworthy circle fit."""
    out = {"stem_radius_m": float("nan"), "stem_radius_slices": 0, "stem_radius_arc_coverage_deg": float("nan")}
    if len(stem_pts) < scfg.stem_radius_min_points:
        return out
    z0, z1 = stem_pts[:, 2].min(), stem_pts[:, 2].max()
    radii, cover = [], []
    for lo in np.arange(z0, z1, scfg.stem_radius_slice):
        sl = stem_pts[(stem_pts[:, 2] >= lo) & (stem_pts[:, 2] < lo + scfg.stem_radius_slice)]
        if len(sl) < scfg.stem_radius_min_points:
            continue
        cx, cy, r, rms = _fit_circle(sl[:, :2])
        if not np.isfinite(r) or r < 0.002 or r > scfg.stem_radius_max or rms > 0.35 * r:
            continue
        ang = np.sort(np.arctan2(sl[:, 1] - cy, sl[:, 0] - cx))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
        span = math.degrees(2 * np.pi - gaps.max())
        if span < 90.0:          # less than a quarter of the stem visible -> radius unreliable
            continue
        radii.append(r)
        cover.append(span)
    if radii:
        out.update(stem_radius_m=float(np.median(radii)), stem_radius_slices=len(radii),
                   stem_radius_arc_coverage_deg=float(np.median(cover)))
    return out


def _main_direction(pts: np.ndarray) -> np.ndarray:
    q = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    return vt[0]


def merge_leaf_fragments(
    clusters: list[np.ndarray], labels: list[int], axis: np.ndarray, scfg: StructureConfig,
) -> tuple[list[np.ndarray], list[int], list[tuple[int, int]]]:
    """Join leaf pieces that do not touch the stem to the leaf they continue.

    A piece is merged into another leaf when the gap between them is small and
    the piece points the same way as the host leaf at the contact point. This
    handles leaves split by occlusion or by a gap in the scan.
    """
    clusters = [c for c in clusters]
    labels = list(labels)
    merges: list[tuple[int, int]] = []
    if len(clusters) < 2 or len(axis) == 0:
        return clusters, labels, merges
    axis_tree = cKDTree(axis)
    changed = True
    while changed:
        changed = False
        dist_to_axis = [float(axis_tree.query(c, k=1)[0].min()) for c in clusters]
        order = np.argsort(dist_to_axis)[::-1]     # most detached pieces first
        for i in order:
            if dist_to_axis[i] <= scfg.leaf_attach_dist or len(clusters) < 2:
                continue
            piece = clusters[i]
            d_ax, _ = axis_tree.query(piece, k=1)
            inner = piece[np.argsort(d_ax)[: max(3, len(piece) // 20)]].mean(axis=0)
            # a piece that starts right beside the stem axis (seen from above) is a real
            # leaf, even if it is low or the stem points are missing -> never merge it
            if float(np.min(np.linalg.norm(axis[:, :2] - inner[:2], axis=1))) <= 1.2 * scfg.leaf_attach_dist:
                continue
            best, best_gap = None, scfg.fragment_merge_gap
            for j, host in enumerate(clusters):
                if j == i:
                    continue
                d, idx = cKDTree(host).query(piece, k=1)
                gap = float(d.min())
                if gap >= best_gap:
                    continue
                p_contact = piece[int(np.argmin(d))]
                h_contact = host[idx[int(np.argmin(d))]]
                # 1. the piece touches the host with its INNER end (not its side or tip)
                if np.linalg.norm(p_contact - inner) > 0.04:
                    continue
                # 2. the contact is on the OUTER part of the host, i.e. the piece continues
                #    the host beyond it (a separate leaf would touch near the host's base)
                host_ax, _ = axis_tree.query(host, k=1)
                if float(axis_tree.query(h_contact, k=1)[0]) < 0.6 * float(host_ax.max()):
                    continue
                # 3. same compass direction seen from above: a maize leaf keeps its azimuth
                #    along its length even when it arches down, so compare XY directions
                #    (stem -> host contact) and (contact -> nearby piece points)
                _, ai = axis_tree.query(h_contact, k=1)
                v_h = h_contact[:2] - axis[int(ai), :2]
                near = piece[np.linalg.norm(piece - p_contact, axis=1) < 0.10]
                v_p = near[:, :2].mean(axis=0) - p_contact[:2]
                if np.linalg.norm(v_p) >= 0.01 and np.linalg.norm(v_h) >= 0.01:
                    cosang = float(np.dot(v_h, v_p) / (np.linalg.norm(v_h) * np.linalg.norm(v_p)))
                    if math.degrees(math.acos(np.clip(cosang, -1.0, 1.0))) > scfg.fragment_merge_angle_deg:
                        continue
                best, best_gap = j, gap
            if best is not None:
                clusters[best] = np.vstack([clusters[best], piece])
                merges.append((labels[best], labels[i]))
                del clusters[i]
                del labels[i]
                changed = True
                break
    return clusters, labels, merges


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------

def structure_from_labels(
    xyz: np.ndarray, labels: np.ndarray, *,
    cfg: "ref.PipelineConfig | None" = None, scfg: StructureConfig | None = None,
    base_point=None, merge_fragments: bool = True,
) -> PlantStructure:
    cfg = cfg or ref.PipelineConfig()
    scfg = scfg or StructureConfig()
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    warnings: list[str] = []

    base = find_base_point(xyz, labels, base_point)
    # Work in a local frame (base at origin) for numerical comfort; shift back at the end.
    shift = base.copy()
    loc = xyz - shift
    base_loc = np.zeros(3)

    stem_full = loc[labels == 0]
    work, wlab = voxel_downsample_labelled(loc, labels, cfg.voxel_size)
    stem_pts = work[wlab == 0]
    leaf_ids = [int(v) for v in np.unique(wlab) if v > 0]
    clusters = [work[wlab == v] for v in leaf_ids]
    # the reference midrib graph needs more points than its k neighbours
    min_pts = max(cfg.leaf_graph_k + 2, 20)
    tiny = [leaf_ids[i] for i, c in enumerate(clusters) if len(c) < min_pts]
    if tiny:
        warnings.append(f"leaf labels {tiny} have < {min_pts} points after downsampling and were skipped")
    keep = [i for i, c in enumerate(clusters) if len(c) >= min_pts]
    clusters = [clusters[i] for i in keep]
    leaf_ids = [leaf_ids[i] for i in keep]

    if len(stem_full) < scfg.min_stem_points:
        warnings.append(f"only {len(stem_full)} stem points -> stem axis rebuilt from leaf bases")

    axis, status = build_stem_axis(stem_pts, clusters, base_loc, cfg, scfg)

    merges: list[tuple[int, int]] = []
    if merge_fragments and len(clusters) > 1:
        clusters, leaf_ids, merges = merge_leaf_fragments(clusters, leaf_ids, axis, scfg)
        if merges:
            warnings.append(f"merged leaf pieces (kept, merged): {merges}")

    stem_sk = ref.StemSkeletonResult(points=axis, ordered_indices=list(range(len(axis))))
    junctions, j_idx = ref.find_leaf_junctions(stem_sk, clusters)
    leaf_sk = ref.extract_all_leaf_skeletons(clusters, junctions, j_idx, cfg)
    sm_axis, sm_paths = ref.smooth_skeletons(axis, leaf_sk.paths, cfg)
    sm_axis[0] = axis[0]   # keep the base exactly where it was found

    # ---- plant level ----
    z_top = float(loc[:, 2].max())
    stem_len_full = ref._path_length(sm_axis)
    if len(leaf_sk.junctions):
        top_idx = int(np.max(leaf_sk.junction_stem_indices))
        stem_len_top_leaf = ref._path_length(sm_axis[: top_idx + 1])
    else:
        stem_len_top_leaf = float("nan")
    rad = stem_radius(stem_full, scfg)

    # ---- leaf level ----
    leaves: list[dict] = []
    for n, (path, ci, junc, si) in enumerate(zip(sm_paths, leaf_sk.cluster_indices, leaf_sk.junctions, leaf_sk.junction_stem_indices)):
        w_max, w_mean, area = ref.leaf_width_and_area(clusters[ci], path, cfg)
        angle = ref._stem_leaf_angle_deg(sm_axis, int(si), path, junc)
        d = ref._leaf_direction_at_junction(path, junc)
        incl = math.degrees(math.asin(np.clip(d[2], -1, 1))) if np.linalg.norm(d) > 0 else float("nan")
        leaves.append({
            "leaf_label": leaf_ids[ci],
            "leaf_length_m": ref._path_length(path),
            "leaf_max_width_m": w_max,
            "leaf_mean_width_m": w_mean,
            "leaf_area_m2": area,
            "leaf_angle_to_stem_deg": angle,
            "leaf_inclination_from_horizontal_deg": incl,
            "insertion_height_m": float(junc[2] - base_loc[2]),
            "n_points": int(len(clusters[ci])),
        })
    measured = {lf["leaf_label"] for lf in leaves}
    for lid, c in zip(leaf_ids, clusters):
        if lid not in measured:
            warnings.append(f"leaf {lid}: no midrib path found ({len(c)} pts)")

    traits = {
        "base_x": float(base[0]), "base_y": float(base[1]), "base_z": float(base[2]),
        "plant_height_m": z_top - base_loc[2],
        "stem_length_m": stem_len_full,
        "stem_length_to_top_leaf_m": stem_len_top_leaf,
        "stem_status": status,
        "n_stem_points": int(len(stem_full)),
        **rad,
        "stem_diameter_mm": rad["stem_radius_m"] * 2000.0 if np.isfinite(rad["stem_radius_m"]) else float("nan"),
        "n_leaves": len(leaves),
        "total_leaf_area_m2": float(sum(lf["leaf_area_m2"] for lf in leaves)),
        "mean_leaf_length_m": float(np.mean([lf["leaf_length_m"] for lf in leaves])) if leaves else float("nan"),
        "mean_leaf_angle_to_stem_deg": float(np.mean([lf["leaf_angle_to_stem_deg"] for lf in leaves])) if leaves else float("nan"),
    }
    return PlantStructure(
        traits=traits, leaves=leaves,
        centerline=sm_axis + shift, leaf_paths=[p + shift for p in sm_paths],
        leaf_labels=[lf["leaf_label"] for lf in leaves], stem_status=status,
        merges=merges, warnings=warnings,
    )
