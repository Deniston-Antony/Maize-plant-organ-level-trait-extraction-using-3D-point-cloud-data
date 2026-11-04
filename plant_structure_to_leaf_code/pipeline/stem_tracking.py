"""Stem/leaf segmentation candidates that follow a leaning stem.

The reference method (reference_structure.segment_stem_and_leaf) measures
every point's distance to ONE vertical axis and allows a stem radius of at
least 2.9 cm. Measured on the ground truth of 230621, stems lean 4-13 cm
from bottom to top and have a median radius of only 5-10 mm, so leaf bases
near the axis become stem and the upper stem of a leaning plant becomes leaf.

Methods here (all return xyz_work, labels: 0 stem, 1..n leaves):

  track_stem      : follow the stem upward slice by slice with circle fits;
                    stem = points within the fitted radius of the tracked axis.
  segment_tracked : stem from track_stem, leaves split by DBSCAN + the
                    reference angular split (using the local axis position).
  segment_hybrid  : reference PCA stem score, but with the distance measured
                    to the tracked (leaning) axis and the tracked radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import reference_structure as ref


@dataclass
class TrackConfig:
    slice_h: float = 0.03           # slice thickness (m)
    step: float = 0.01              # step between slices (m)
    base_band: float = 0.03         # bottom band used to find the stem start (m)
    search: float = 0.03            # search half-width around the current axis (m)
    cell: float = 0.004             # density grid cell (m)
    smooth_cells: float = 1.5       # Gaussian smoothing of the density grid (cells)
    pred_sigma: float = 0.008       # weight around the predicted axis (m); 0.012 lost RL_plant01
    max_move: float = 0.005         # max axis move per step (m)
    min_slice_pts: int = 10
    max_gap_slices: int = 4         # stop after this many empty slices in a row
    r_min: float = 0.004
    r_max: float = 0.020
    r_default: float = 0.008
    accept_factor: float = 2.0      # stem = within accept_factor x local radius of the axis
    accept_tol: float = 0.003       # ... plus this (m)
    extra: dict = field(default_factory=dict)


@dataclass
class StemTrack:
    centers: np.ndarray             # (M, 3) axis points, bottom to top
    radii: np.ndarray               # (M,)
    stem_mask: np.ndarray           # per work point
    dist_to_axis: np.ndarray        # per work point, horizontal distance to the axis at its height
    radius_at: np.ndarray           # per work point, axis radius at its height
    axis_xy_at: np.ndarray          # per work point, (x, y) of the axis at its height
    status: str = "ok"


def _arc_deg(xy: np.ndarray, c: np.ndarray) -> float:
    a = np.sort(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, a[:1] + 2 * np.pi]))
    return float(np.degrees(2 * np.pi - gaps.max()))


def _start_xy(xyz: np.ndarray, tc: TrackConfig) -> tuple[np.ndarray, float]:
    """Stem column = the 1 cm XY cell with the most points in the lower 55 % of
    the plant (as in the reference); the lowest points can be a drooping leaf
    tip. Start height = lowest point within 2 cm of that column."""
    z0, z1 = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    low = xyz[xyz[:, 2] <= z0 + 0.55 * (z1 - z0)]
    if len(low) < 10:
        low = xyz
    keys = np.floor(low[:, :2] / 0.01).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    c = low[inv == int(np.argmax(cnt)), :2].mean(axis=0)
    col = xyz[np.linalg.norm(xyz[:, :2] - c, axis=1) < 0.02]
    zs = float(np.percentile(col[:, 2], 1)) if len(col) else z0
    bottom = col[col[:, 2] <= zs + tc.base_band]
    return (bottom[:, :2].mean(axis=0) if len(bottom) >= 5 else c), zs


def track_stem(xyz: np.ndarray, tc: TrackConfig | None = None, label_xyz: np.ndarray | None = None) -> StemTrack:
    """Follow the stem upward. In each slice the axis moves to the densest
    small column near the predicted position (the stem is seen from several
    sides, so it is the densest column; leaves spread out). A Gaussian weight
    around the prediction and a per-step move limit keep it from following a
    leaf. Measured on the 230621 GT: axis within 2-12 mm (median) of the stem.

    Track on `xyz` (the full-resolution cleaned cloud); the stem mask and the
    per-point axis values are returned for `label_xyz` (default: xyz)."""
    from scipy.ndimage import gaussian_filter
    tc = tc or TrackConfig()
    c, z0 = _start_xy(xyz, tc)
    z_top = float(xyz[:, 2].max())
    order = np.argsort(xyz[:, 2])
    zs = xyz[order, 2]
    S, cell = tc.search, tc.cell
    nb = int(round(2 * S / cell))
    g = -S + (np.arange(nb) + 0.5) * cell

    centers, radii = [], []
    gap = 0
    z = z0
    while z < z_top:
        lo, hi = np.searchsorted(zs, [z, z + tc.slice_h])
        pts = xyz[order[lo:hi], :2]
        pred = c + (np.mean(np.diff(np.asarray(centers)[-5:, :2], axis=0), axis=0) if len(centers) >= 5 else 0)
        p = pts[np.linalg.norm(pts - c, axis=1) < S]
        if len(p) < tc.min_slice_pts:
            gap += 1
            if gap > tc.max_gap_slices and centers:
                break
            z += tc.step
            continue
        gap = 0
        H, _, _ = np.histogram2d(p[:, 0] - c[0], p[:, 1] - c[1], bins=nb, range=[[-S, S], [-S, S]])
        H = gaussian_filter(H, tc.smooth_cells)
        pc = pred - c
        H *= np.exp(-((g[:, None] - pc[0]) ** 2 + (g[None, :] - pc[1]) ** 2) / (2 * tc.pred_sigma ** 2))
        i, j = np.unravel_index(int(np.argmax(H)), H.shape)
        v = np.array([g[i], g[j]])
        d = float(np.linalg.norm(v))
        c = c + (v / d * min(d, tc.max_move) if d > 0 else 0)
        # local radius: median distance of the points close to the axis
        dd = np.linalg.norm(p - c, axis=1)
        core = dd[dd <= tc.r_max]
        r = float(np.clip(np.median(core) if len(core) >= 5 else tc.r_default, tc.r_min, tc.r_max))
        centers.append([c[0], c[1], z + tc.slice_h / 2])
        radii.append(r)
        z += tc.step

    if centers:
        # the start column can have a gap near the soil: extend the axis down to
        # the lowest points within 2 cm of the first centre
        c0 = np.asarray(centers[0])
        col = xyz[(np.linalg.norm(xyz[:, :2] - c0[:2], axis=1) < 0.02) & (xyz[:, 2] < c0[2])]
        if len(col) >= 5:
            zl = float(np.percentile(col[:, 2], 1))
            if zl < c0[2] - tc.step / 2:
                centers.insert(0, [c0[0], c0[1], zl])
                radii.insert(0, radii[0])
    if label_xyz is not None:
        xyz = label_xyz
    if not centers:
        n = len(xyz)
        return StemTrack(np.empty((0, 3)), np.empty(0), np.zeros(n, bool), np.full(n, np.inf),
                         np.zeros(n), np.tile(c, (n, 1)), status="no_stem")
    C, Rr = np.asarray(centers), np.asarray(radii)
    # smooth radius (median of 5) so single bad fits do not matter
    if len(Rr) >= 5:
        Rr = np.array([np.median(Rr[max(0, i - 2):i + 3]) for i in range(len(Rr))])
    axis_xy = np.column_stack([np.interp(xyz[:, 2], C[:, 2], C[:, 0]), np.interp(xyz[:, 2], C[:, 2], C[:, 1])])
    rad_at = np.interp(xyz[:, 2], C[:, 2], Rr)
    dist = np.linalg.norm(xyz[:, :2] - axis_xy, axis=1)
    inside_z = (xyz[:, 2] >= C[0, 2] - tc.slice_h) & (xyz[:, 2] <= C[-1, 2] + tc.slice_h / 2)
    stem = inside_z & (dist <= tc.accept_factor * rad_at + tc.accept_tol)
    return StemTrack(C, Rr, stem, np.where(inside_z, dist, np.inf), rad_at, axis_xy)


# ---------------------------------------------------------------- leaves
MIN_LEAF_SHARE = 0.0   # set by the bench; clusters below this share of all leaf points are merged
def _leaf_labels(xyz: np.ndarray, stem: np.ndarray, axis_xy: np.ndarray, cfg: ref.PipelineConfig) -> np.ndarray:
    """0 = stem, 1..n leaves; leaf clusters by the reference DBSCAN + angular split,
    with the angular split measured around the local (leaning) axis."""
    labels = np.full(len(xyz), -1, np.int32)
    labels[stem] = 0
    leaf_idx = np.where(~stem)[0]
    if len(leaf_idx) >= cfg.leaf_cluster_min_points:
        leaf = xyz[leaf_idx]
        eps, ms = ref._adaptive_dbscan_params(leaf, cfg)
        from sklearn.cluster import DBSCAN
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(leaf)
        clusters = []
        for k in sorted(set(lab.tolist()) - {-1}):
            sel = np.where(lab == k)[0]
            if len(sel) < cfg.leaf_cluster_min_points:
                continue
            # local axis = axis position at the cluster's lowest (innermost) height
            low = sel[np.argsort(leaf[sel, 2])[: max(5, len(sel) // 20)]]
            ax = axis_xy[leaf_idx[low]].mean(axis=0)
            parts = ref._split_cluster_by_stem_angle(leaf[sel], ax, cfg) if cfg.split_nearby_leaves else [leaf[sel]]
            clusters.extend(p for p in parts if len(p) >= cfg.leaf_cluster_min_points)
        clusters.sort(key=len, reverse=True)
        # tiny clusters (pieces of a leaf or whorl) are not leaves of their own:
        # their points go to the nearest real leaf
        min_share = float(cfg.__dict__.get("min_leaf_share", MIN_LEAF_SHARE))
        if clusters and min_share > 0:
            big = [c for c in clusters if len(c) >= min_share * len(leaf)]
            clusters = big or clusters[:1]
        if clusters:
            allp = np.vstack(clusters)
            cid = np.concatenate([np.full(len(p), i + 1) for i, p in enumerate(clusters)])
            _, nn = cKDTree(allp).query(leaf, k=1, workers=-1)
            labels[leaf_idx] = cid[nn]
    un = labels < 0
    if un.any() and (~un).any():
        _, nn = cKDTree(xyz[~un]).query(xyz[un], k=1, workers=-1)
        labels[un] = labels[~un][nn]
    elif un.any():
        labels[un] = 0
    return labels


def _prep(xyz: np.ndarray, cfg: ref.PipelineConfig):
    center = xyz.mean(axis=0)
    prep = ref.preprocess_point_cloud(xyz - center, cfg, rgb=None)
    return prep, center


def segment_tracked(xyz: np.ndarray, cfg: ref.PipelineConfig | None = None,
                    tc: TrackConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or ref.PipelineConfig()
    prep, center = _prep(xyz, cfg)
    w = prep.xyz_work
    tr = track_stem(prep.xyz_clean, tc, label_xyz=w)
    return w + center, _leaf_labels(w, tr.stem_mask, tr.axis_xy_at, cfg)


def segment_hybrid(xyz: np.ndarray, cfg: ref.PipelineConfig | None = None,
                   tc: TrackConfig | None = None, radius_factor: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """Reference PCA stem score with the distance to the tracked axis."""
    cfg = cfg or ref.PipelineConfig()
    prep, center = _prep(xyz, cfg)
    w, normals = prep.xyz_work, prep.normals
    tr = track_stem(prep.xyz_clean, tc, label_xyz=w)
    radial = np.where(np.isfinite(tr.dist_to_axis), tr.dist_to_axis, 1.0)
    srad = np.maximum(tr.radius_at * radius_factor, 0.006)       # per-point stem radius
    pca, _ = ref._compute_multiscale_pca_features(w, cfg.feature_scales or (cfg.feature_k,))
    vdeg = ref._compute_verticality_deg(normals)
    rs = radial / srad                                            # normalised distance, 1 = stem edge
    score = ref._compute_stem_scores(rs, pca.linearity, pca.curvature, vdeg, 1.0, cfg)
    stem = score >= cfg.stem_score_threshold
    stem = ref._region_grow_stem(w, normals, rs, pca.linearity, pca.curvature, pca.planarity, stem, 1.0, cfg)
    stem &= rs <= 1.0
    stem |= rs <= 0.6                                             # core of the tracked stem is always stem
    return w + center, _leaf_labels(w, stem, tr.axis_xy_at, cfg)


def segment_reference(xyz: np.ndarray, cfg: ref.PipelineConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    return ref.classify_plant_for_iou(xyz, cfg or ref.PipelineConfig())


# ---------------------------------------------------------------- measuring cloud
def light_clean(xyz: np.ndarray, voxel: float = 0.004, sor_neighbors: int = 20, sor_std: float = 3.0) -> np.ndarray:
    """Cloud for measuring traits: only a mild statistical outlier filter, then
    thinning. The reference radius filter (8 neighbours in 12 mm) removes thin
    leaf edges and tips: with perfect GT labels it alone made leaf area -19 %,
    leaf length -8 % and stem radius -14 % (230621 GT); without it 2-4 %."""
    import open3d as o3d
    c = xyz.mean(axis=0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz - c))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std)
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points) + c


def transfer_labels(seg_xyz: np.ndarray, seg_labels: np.ndarray, target_xyz: np.ndarray,
                    max_dist: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Copy labels from the segmentation cloud to the measuring cloud
    (nearest neighbour); points farther than max_dist from any labelled point are dropped."""
    d, nn = cKDTree(seg_xyz).query(target_xyz, k=1, workers=-1)
    keep = d <= max_dist
    return target_xyz[keep], seg_labels[nn[keep]].astype(np.int32)


def with_measuring_cloud(method):
    """Wrap a segmentation method so its labels end up on the lightly cleaned cloud."""
    def run(xyz, *a, **kw):
        sx, sl = method(xyz, *a, **kw)
        return transfer_labels(sx, sl, light_clean(xyz))
    run.__name__ = method.__name__ + "_measured"
    return run


# ---------------------------------------------------------------- default method
def segment_v2(xyz: np.ndarray, cfg=None) -> tuple[np.ndarray, np.ndarray]:
    """Research_project/new_structure_output/claude_plant_structure.py (copied
    unchanged to pipeline/structure_v2.py): reference PCA scoring plus a fitted
    leaning stem axis, a second stem pass, main-stem component filter and
    detached-fragment merge. Geometry only (no training, no colour)."""
    from . import structure_v2 as v2
    return v2.classify_plant_for_iou(xyz, cfg or v2.PipelineConfig())


# Default for the pipeline (best on the 230621 GT, tests/bench_structure.py):
# v2 segmentation, traits measured on the lightly cleaned cloud.
segment_default = with_measuring_cloud(segment_v2)
