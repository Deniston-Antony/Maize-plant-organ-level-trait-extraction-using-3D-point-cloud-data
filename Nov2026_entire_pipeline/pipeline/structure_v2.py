#!/usr/bin/env python3
"""
Combined plant classification + plant structure pipeline.

Based on New_optimized_plant_structurepy, with these changes:

 Stem
 1. Curved stem axis: after a first pass around a vertical line, the stem
    centreline is re-fitted and the stem is segmented again inside a 2 cm
    corridor around that curve (repeated until stable). A leaning stem is
    followed to its top and leaf bases are no longer absorbed.
 2. Stem islands not connected to the main stem body are dropped.
 3. Leaf clusters that hug the stem axis are given back to the stem.
 4. Stem centreline smoothing uses edge padding (zero padding bent both ends
    towards the origin).

 Leaves
 5. Clusters keep their point indices, so every point is labelled with the
    cluster it is actually in (no distance-based re-matching).
 6. Leaf fragments that do not reach the stem are merged into the nearest
    attached cluster; small leftovers are not reported as leaves.

 Measurements
 7. Junctions are projected onto the stem centreline, so insertion height is
    read on the stem. Leaf paths start at that junction.
 8. Leaf angle: line fitted over the first 4 cm of the leaf, against the stem
    direction taken over +-3 centreline points.
 9. Leaf length is measured on the smoothed path; smoothing keeps end points.
10. Plant bottom/top use z percentiles instead of the single extreme point.

 Evaluation
11. A strict mean IoU is reported as well; it counts predicted leaves that
    match no GT leaf as 0.
12. Skeletons that match no classified leaf keep their own number instead of
    being collapsed into one.

Install
-------
    pip install open3d numpy scipy scikit-learn networkx pyvista laspy

Example
-------
    python claude_plant_structure.py
    python claude_plant_structure.py --bin ../../P147/manual_segmented_leaf_file/plant.bin
    python claude_plant_structure.py --skeleton_only --input plant.las
    python claude_plant_structure.py --classification_only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import laspy
import networkx as nx
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

# This file lives in <project>/fadly_anna_code/new_structure_output/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BIN = (
    _PROJECT_ROOT
    / "P147"
    / "manual_segmented_leaf_file"
    / "LtR_12th_Plant_230621pc_kinematic_calibration.bin"
)



def resolve_bin_path(path: Path) -> Path:
    """Resolve .bin path; try project root if relative path is missing."""
    path = Path(path)
    if path.is_file():
        return path.resolve()

    # Common mistake: path relative to fadly_anna_code instead of project root
    if path.is_absolute():
        name = path.name
        alt = _PROJECT_ROOT / "P147" / "manual_segmented_leaf_file" / name
        if alt.is_file():
            return alt.resolve()
        return path

    for base in (_PROJECT_ROOT, Path(__file__).resolve().parent):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return path.resolve()


def find_cloudcompare_exe() -> Path | None:
    """Locate CloudCompare CLI on Windows, macOS, or Linux."""
    env = os.environ.get("CLOUDCOMPARE_EXE")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    candidates: list[Path] = [
        Path(r"C:\Program Files\CloudCompare\CloudCompare.exe"),
        Path(r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe"),
        Path("/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"),
        Path("/Applications/ccViewer.app/Contents/MacOS/ccViewer"),
        Path("/usr/bin/CloudCompare"),
        Path("/usr/local/bin/CloudCompare"),
        Path("/snap/bin/cloudcompare.CloudCompare"),
    ]
    for name in ("CloudCompare", "cloudcompare", "cloudcompare.CloudCompare"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    for cand in candidates:
        if cand.is_file():
            return cand
    return None


CLOUDCOMPARE_EXE = find_cloudcompare_exe()

def _make_leaf_palette(n: int = 50) -> dict[int, tuple[float, float, float]]:
    """
    Build ``n`` visually distinct leaf RGB colors (class ids 1..n).
    Stem (id 0) is added separately. Uses spaced hues with varying S/V.
    """
    colors: dict[int, tuple[float, float, float]] = {
        0: (0.15, 0.72, 0.22),  # stem - green color
    }
    # Golden-ratio hue walk + alternating sat/val for separation
    phi = 0.618033988749895
    h = 0.12  # start offset (avoid stem-green clash)
    for i in range(1, n + 1):
        h = (h + phi) % 1.0
        s = 0.55 + 0.40 * ((i % 3) / 2.0)   # 0.55 .. 0.95
        v = 0.75 + 0.20 * ((i % 4) / 3.0)   # 0.75 .. 0.95
        # HSV → RGB
        c = v * s
        x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
        m = v - c
        hi = int(h * 6.0) % 6
        if hi == 0:
            r, g, b = c, x, 0.0
        elif hi == 1:
            r, g, b = x, c, 0.0
        elif hi == 2:
            r, g, b = 0.0, c, x
        elif hi == 3:
            r, g, b = 0.0, x, c
        elif hi == 4:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
        colors[i] = (r + m, g + m, b + m)
    return colors


CLASS_COLORS = _make_leaf_palette(50)
UNMATCHED_COLOR = (0.55, 0.55, 0.55)
IOU_TABLE_POSITION = (0.02, 0.58)

# Pre-build a colour lookup array for fast vectorised label→RGB
_MAX_CLASS_COLOR = max(CLASS_COLORS) + 1
_COLOR_LUT = np.full((_MAX_CLASS_COLOR + 1, 3), UNMATCHED_COLOR, dtype=np.float64)
for _cid, _rgb in CLASS_COLORS.items():
    _COLOR_LUT[_cid] = _rgb

# Flat leaf-only palette (50 colors) for cluster / skeleton viewers
_LEAF_PALETTE = np.array(
    [CLASS_COLORS[i] for i in range(1, 51)], dtype=np.float64
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """All tunable pipeline parameters."""

    sor_neighbors: int = 20  # statistical outlier removal neighbors = For each point, look at its 20 nearest neighbors and compute mean distance to them.
    sor_std_ratio: float = 2.0 # Keep a point if its mean distance is within mean + 2σ of the global distribution.
    ror_radius: float = 0.012 # Radius outlier removal = Search ball around each point.
    ror_min_neighbors: int = 8 # If fewer than 8 points fall in that ball, the point is noise.
    voxel_size: float = 0.004 # Grid cell size. All points in a 4 mm cube are replaced by one (usually the centroid).
    normal_radius: float = 0.012 # Fixed neighborhood radius for the PCA (used when adaptive mode is off).
    normal_radius_factor: float = 3.0 # Adaptive radius = 3 × voxel_size = 12 mm. Same scale as normal_radius with the current defaults.
    use_adaptive_normal_radius: bool = True # Use voxel_size * normal_radius_factor instead of the hard-coded normal_radius. If you change voxel size, the normal neighborhood scales with it.
    normal_max_nn: int = 40 # Cap: use at most 40 neighbors even if more fall in the radius. Prevents dense regions from using a huge neighborhood (which blurs normals).

# These knobs belong to step 2: stem vs leaf. After cleaning, the pipeline estimates where the stem axis is, describes each point with local PCA, then scores “how stem-like” it is.
    feature_k: int = 48
    feature_scales: tuple[int, ...] = (20, 40, 80)
    stem_radial_max: float = 0.0345 # 3.45cm Nominal max stem radius. Points farther than this are unlikely stem.
    stem_radial_cap_factor: float = 1.5
    stem_radial_floor_factor: float = 0.85
    use_adaptive_stem_radius: bool = True # Estimate radius from this plant instead of always using 3.45 cm.
    adaptive_stem_radius_sigma: float = 1.0
    # PCA eigenvalues : https://mathformachines.com/posts/eigenvalues-and-singular-values/ 
    stem_verticality_min: float = 0.55 # Surface should be more vertical than horizontal. Filters flat leaf blades.
    stem_linearity_min: float = 0.35 # Neighborhood should be at least somewhat line-like. Soft gate; seeds later use a stricter 0.60.
    stem_curvature_max: float = 0.25 # Stem neighborhoods should not be too “3D.” Higher curvature → junction or leaf.
    leaf_planarity_min: float = 0.18 # Enough planar structure to call a point leaf-like.
    root_xy_cell: float = 0.01 # XY bin size. Count points per 1 cm × 1 cm cell.
    root_density_z_frac: float = 0.55 # Only use the bottom 55% of the height range.

    stem_score_weight_radial: float = 0.40
    stem_score_weight_linearity: float = 0.30
    stem_score_weight_curvature: float = 0.20
    stem_score_weight_verticality: float = 0.10
    stem_score_threshold: float = 0.58

    # Curved stem axis: after a first pass around a vertical line, the stem
    # centreline is re-fitted and distances are measured to that curve, so a
    # leaning or bending stem stays inside a much narrower corridor.
    use_curved_stem_axis: bool = True
    stem_axis_iterations: int = 8  # max refits; stops early once the mask is stable
    stem_axis_radius: float = 0.020  # corridor radius around the fitted axis (2 cm)
    stem_axis_z_step: float = 0.015  # height of the slices used to fit the axis
    stem_axis_min_slice_points: int = 3
    stem_axis_max_slope: float = 1.0  # cap on d(xy)/dz when extrapolating past the ends
    stem_component_max_edge_factor: float = 3.0  # x voxel_size; link distance for stem connectivity
    stem_component_min_frac: float = 0.10  # drop stem islands smaller than this share of the largest

    use_region_growing: bool = True
    stem_seed_linearity_min: float = 0.60
    stem_seed_curvature_max: float = 0.12
    stem_seed_radial_frac: float = 0.55
    region_grow_k: int = 16
    region_grow_normal_sim_min: float = 0.75
    region_grow_curvature_sim: float = 0.08
    region_grow_linearity_sim: float = 0.15
    region_grow_radial_sim: float = 0.012

    min_leaf_component_size: int = 50
    leaf_component_graph_k: int = 12

    graph_k: int = 12
    graph_max_edge: float = 0.07

    laplacian_k: int = 16
    laplacian_iterations: int = 35
    laplacian_contract: float = 0.55
    laplacian_volume: float = 0.12
    stem_z_step: float = 0.008
    stem_smooth_window: int = 5

    leaf_cluster_eps: float = 0.028
    leaf_cluster_min_points: int = 25
    leaf_max_clusters: int = 50
    use_adaptive_dbscan: bool = True
    dbscan_eps_nn_factor: float = 2.5
    dbscan_eps_voxel_factor: float = 4.0
    dbscan_min_samples_density_factor: float = 0.002
    # Leaf fragments: a cluster that does not touch the stem is merged into the
    # nearest cluster that does; small leftovers are not reported as leaves.
    merge_detached_leaf_fragments: bool = True
    leaf_attach_max_dist: float = 0.025  # cluster counts as attached within 2.5 cm of stem points
    leaf_fragment_merge_gap: float = 0.040  # max gap to the cluster it is merged into
    leaf_detached_keep_min_points: int = 300  # detached clusters at least this big stay as leaves
    # A leaf has to extend away from the stem. A cluster that stays within
    # stem_hugging_radius_factor x stem radius of the axis is a piece of stem
    # that failed the stem test, and is given back to the stem.
    reassign_stem_hugging_clusters: bool = True
    stem_hugging_radius_factor: float = 2.0
    stem_hugging_min_frac: float = 0.90
    split_nearby_leaves: bool = True
    leaf_split_min_cluster_size: int = 3000
    leaf_split_min_angle_deg: float = 40.0
    leaf_split_max_k: int = 3

    use_geodesic_primary: bool = False
    use_dbscan_leaf_fallback: bool = True
    geodesic_min_junctions: int = 2
    geodesic_max_junctions: int = 8
    junction_z_bin_m: float = 0.032
    junction_z_step_m: float = 0.018
    junction_seed_min_points: int = 20
    junction_nms_z_m: float = 0.03
    junction_nms_ang_deg: float = 12.0
    junction_min_support: int = 20
    junction_min_support_frac: float = 0.05
    plant_graph_k: int = 12
    plant_graph_max_edge: float = 0.050
    crown_height_frac: float = 0.30
    crown_graph_height_frac: float = 0.30
    crown_stem_radius_scale: float = 1.55
    crown_stem_z_bins: int = 8
    junction_radial_max: float = 0.045
    junction_density_k: int = 16
    junction_density_radius: float = 0.012
    junction_min_direction_modes: int = 2
    junction_angle_bin_deg: float = 30.0
    crown_graph_k: int = 10
    crown_graph_max_edge: float = 0.035
    crown_bridge_dist_scale: float = 1.15
    crown_bridge_angle_deg: float = 22.0
    crown_bridge_radial_diff: float = 0.012
    crown_angular_jump_deg: float = 28.0
    crown_assign_alpha: float = 1.0
    crown_assign_beta: float = 0.035
    crown_assign_gamma: float = 0.8
    crown_assign_delta: float = 1.2
    crown_geodesic_k: int = 12
    crown_geodesic_max_edge: float = 0.040
    crown_min_seed_points: int = 40
    crown_seed_radial_band: float = 0.055
    use_crown_geodesic_growing: bool = True
    crown_fragment_min_points: int = 150
    crown_fragment_min_frac: float = 0.03
    fragment_merge_w_euc: float = 0.45
    fragment_merge_w_ang: float = 0.35
    fragment_merge_w_rad: float = 0.20

    leaf_graph_k: int = 18
    leaf_graph_max_edge: float = 0.075

    smooth_window: int = 7
    smooth_polyorder: int = 2

    leaf_angle_arc_length: float = 0.04  # leaf direction is fitted over the first 4 cm of the leaf
    stem_tangent_half_window: int = 3  # stem direction from +-3 centreline points around the junction
    height_trim_percentile: float = 0.05  # plant bottom/top = this percentile of z from each end

    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "skeleton_output"
    )


@dataclass
class PreprocessResult:
    xyz_raw: np.ndarray
    xyz_clean: np.ndarray
    xyz_work: np.ndarray
    normals: np.ndarray
    pcd_clean: o3d.geometry.PointCloud
    pcd_work: o3d.geometry.PointCloud
    rgb_raw: np.ndarray | None = None
    rgb_work: np.ndarray | None = None


@dataclass
class PCAFeatures:
    linearity: np.ndarray
    planarity: np.ndarray
    scattering: np.ndarray
    curvature: np.ndarray
    omnivariance: np.ndarray
    anisotropy: np.ndarray
    eigenentropy: np.ndarray
    surface_variation: np.ndarray


@dataclass
class SegmentationResult:
    stem_mask: np.ndarray
    leaf_mask: np.ndarray
    stem_points: np.ndarray
    leaf_points: np.ndarray
    verticality: np.ndarray
    planarity: np.ndarray
    linearity: np.ndarray
    curvature: np.ndarray
    scattering: np.ndarray
    stem_xy: np.ndarray
    stem_score: np.ndarray | None = None
    radial: np.ndarray | None = None
    verticality_deg: np.ndarray | None = None
    stem_axis: "StemAxis | None" = None
    stem_radius: float = 0.0


@dataclass
class StemSkeletonResult:
    points: np.ndarray
    ordered_indices: list[int]


@dataclass
class LeafClusterResult:
    clusters: list[np.ndarray]
    labels: np.ndarray
    # Cluster id (0-based, same order as ``clusters``) for every leaf point of
    # the segmentation; -1 where the point belongs to no reported cluster.
    point_labels: np.ndarray | None = None


@dataclass
class LeafSkeletonResult:
    # ``junctions`` lie on the stem centreline. Each path starts at its junction
    # (path[0]), continues at the leaf point nearest the stem (path[1]) and runs
    # to the tip.
    junctions: np.ndarray
    junction_stem_indices: np.ndarray
    paths: list[np.ndarray]
    tips: np.ndarray
    cluster_indices: list[int]


@dataclass
class PlantSkeletonGraph:
    graph: nx.Graph
    node_positions: np.ndarray
    node_roles: list[str]
    stem_edge_indices: list[tuple[int, int]]
    leaf_edge_indices: list[tuple[int, int]]


@dataclass
class Measurements:
    stem_length_m: float
    leaf_lengths_m: list[float]
    n_leaves: int
    insertion_heights_m: list[float]
    leaf_angles_deg: list[float]
    total_skeleton_length_m: float
    plant_bottom_z_m: float = 0.0
    plant_height_m: float = 0.0


@dataclass
class PipelineResult:
    preprocess: PreprocessResult
    segmentation: SegmentationResult
    plant_graph: nx.Graph
    stem_skeleton: StemSkeletonResult
    leaf_clusters: LeafClusterResult
    leaf_skeleton: LeafSkeletonResult
    smoothed_stem: np.ndarray
    smoothed_leaf_paths: list[np.ndarray]
    skeleton_graph: PlantSkeletonGraph
    measurements: Measurements


# =============================================================================
# Step 1 – Load & preprocess
# =============================================================================


def _resolve_readable_cloud_path(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".bin":
        return path
    if not path.is_file():
        return path
    head = path.read_bytes()[:4]
    if head != b"CCB2":
        return path

    candidates = [
        path.with_suffix(".las"),
        path.parent / f"{path.stem}.las",
        path.parent.parent / "old segmented_files" / f"{path.stem}.las",
        Path(__file__).resolve().parent / "P147" / "230621pc_kinematic_calibration"
        / "old segmented_files" / f"{path.stem}.las",
    ]
    for cand in candidates:
        if cand.is_file():
            print(
                f"    Note: CloudCompare .bin is not readable in Python; "
                f"using LAS instead:\n      {cand}"
            )
            return cand
    raise ValueError(
        f"CloudCompare .bin file cannot be loaded directly: {path}\n"
        "Export the cloud to .las/.ply in CloudCompare, or place a matching "
        f"{path.stem}.las next to the .bin / in old segmented_files."
    )


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    path = _resolve_readable_cloud_path(Path(path))
    suffix = path.suffix.lower()
    rgb: np.ndarray | None = None
    if suffix in {".las", ".laz"}:
        las = laspy.read(str(path))
        xyz = np.column_stack(
            (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))
        ).astype(np.float64)
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            rgb = np.column_stack(
                (np.asarray(las.red, dtype=np.float64),
                 np.asarray(las.green, dtype=np.float64),
                 np.asarray(las.blue, dtype=np.float64))
            )
            if rgb.max() > 1.0:
                rgb /= 255.0
    elif suffix in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise ValueError(f"No points in {path}")
        xyz = np.asarray(pcd.points, dtype=np.float64)
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported format: {suffix} ({path})")
    xyz -= xyz.mean(axis=0)
    return xyz, rgb


def _effective_normal_radius(cfg: PipelineConfig) -> float:
    if cfg.use_adaptive_normal_radius:
        return cfg.normal_radius_factor * cfg.voxel_size
    return cfg.normal_radius


def preprocess_point_cloud(
    xyz: np.ndarray,
    cfg: PipelineConfig,
    rgb: np.ndarray | None = None,
) -> PreprocessResult:
    """SOR -> ROR -> voxel downsample; estimate normals on the work cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None and len(rgb) == len(xyz):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=cfg.sor_neighbors, std_ratio=cfg.sor_std_ratio,
    )
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=cfg.ror_min_neighbors, radius=cfg.ror_radius,
    )

    pcd_work = pcd.voxel_down_sample(cfg.voxel_size)
    normal_radius = _effective_normal_radius(cfg)
    pcd_work.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=cfg.normal_max_nn,
        )
    )
    try:
        pcd_work.orient_normals_consistent_tangent_plane(k=min(20, len(pcd_work.points) - 1))
    except RuntimeError:
        pcd_work.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 1.0]))

    xyz_clean = np.asarray(pcd.points, dtype=np.float64)
    xyz_work = np.asarray(pcd_work.points, dtype=np.float64)
    normals = np.asarray(pcd_work.normals, dtype=np.float64)

    rgb_work: np.ndarray | None = None
    if pcd_work.has_colors():
        rgb_work = np.asarray(pcd_work.colors, dtype=np.float64)

    return PreprocessResult(
        xyz_raw=xyz, xyz_clean=xyz_clean, xyz_work=xyz_work,
        normals=normals, pcd_clean=pcd, pcd_work=pcd_work,
        rgb_raw=rgb, rgb_work=rgb_work,
    )


def visualize_preprocessed(prep: PreprocessResult, title: str = "Cleaned point cloud") -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(prep.xyz_clean)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=800)


# =============================================================================
# Step 2 – Stem / leaf segmentation
# =============================================================================

STEM_DIRECTION = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _stem_xy_from_density(xyz: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    z0, z1 = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    z_cut = z0 + cfg.root_density_z_frac * (z1 - z0)
    low = xyz[xyz[:, 2] <= z_cut]
    if len(low) < 10:
        low = xyz
    cell = cfg.root_xy_cell
    keys = np.floor(low[:, :2] / cell).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    best = int(np.argmax(counts))
    mask = inv == best
    return low[mask, :2].mean(axis=0)


def _compute_multiscale_pca_features(
    xyz: np.ndarray,
    scales: tuple[int, ...],
) -> tuple[PCAFeatures, np.ndarray]:
    """
    OPTIMISED: Vectorised batch SVD on stacked neighbourhood matrices.

    Instead of calling _pca_descriptors_from_neighbors once per point (Python
    loop over ~7000 points × 3 scales), we gather all neighbourhoods into a
    single (N, k, 3) array, subtract means in bulk, run np.linalg.svd on the
    entire batch, and compute all descriptors via vectorised NumPy.
    """
    n = len(xyz)
    keys = (
        "linearity", "planarity", "scattering", "curvature",
        "omnivariance", "anisotropy", "eigenentropy", "surface_variation",
    )
    accum = {k: np.zeros(n, dtype=np.float64) for k in keys}
    counts = np.zeros(n, dtype=np.int32)
    nn_idx = np.zeros((n, 1), dtype=np.int64)

    for k_scale in scales:
        k_eff = min(k_scale, n)
        if k_eff < 4:
            continue
        nn = NearestNeighbors(n_neighbors=k_eff).fit(xyz)
        _, idx = nn.kneighbors(xyz)
        if k_scale == max(scales):
            nn_idx = idx

        # Batch: gather all neighbourhoods → (n, k_eff, 3)
        nb = xyz[idx]  # (n, k_eff, 3)
        nb_centered = nb - nb.mean(axis=1, keepdims=True)  # (n, k_eff, 3)

        # Batch SVD → singular values shape (n, min(k_eff, 3))
        try:
            s = np.linalg.svd(nb_centered, compute_uv=False)
        except np.linalg.LinAlgError:
            continue

        evals = np.sort(s ** 2, axis=1)[:, ::-1]  # (n, min(k_eff, 3))
        if evals.shape[1] < 3:
            continue

        l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]
        l1_safe = l1 + 1e-12
        total = l1 + l2 + l3 + 1e-12
        probs = evals[:, :3] / total[:, None]
        probs = np.clip(probs, 1e-12, 1.0)

        valid = l1 > 1e-15
        linearity = np.where(valid, (l1 - l2) / l1_safe, 0.0)
        planarity = np.where(valid, (l2 - l3) / l1_safe, 0.0)
        scattering = np.where(valid, l3 / l1_safe, 0.0)
        curvature_arr = np.where(valid, l3 / total, 0.0)
        omni = np.where(valid, (l1 * l2 * l3) ** (1.0 / 3.0), 0.0)
        aniso = np.where(valid, (l1 - l3) / l1_safe, 0.0)
        entropy = np.where(valid, -np.sum(probs * np.log(probs), axis=1), 0.0)
        surf_var = curvature_arr

        accum["linearity"] += linearity
        accum["planarity"] += planarity
        accum["scattering"] += scattering
        accum["curvature"] += curvature_arr
        accum["omnivariance"] += omni
        accum["anisotropy"] += aniso
        accum["eigenentropy"] += entropy
        accum["surface_variation"] += surf_var
        counts += valid.astype(np.int32)

    for key in keys:
        valid = counts > 0
        accum[key][valid] /= counts[valid]

    return (
        PCAFeatures(
            linearity=accum["linearity"],
            planarity=accum["planarity"],
            scattering=accum["scattering"],
            curvature=accum["curvature"],
            omnivariance=accum["omnivariance"],
            anisotropy=accum["anisotropy"],
            eigenentropy=accum["eigenentropy"],
            surface_variation=accum["surface_variation"],
        ),
        nn_idx,
    )


def _compute_verticality_deg(normals: np.ndarray) -> np.ndarray:
    nz = np.clip(np.abs(normals @ STEM_DIRECTION), 0.0, 1.0)
    return np.degrees(np.arccos(nz))


def _estimate_adaptive_stem_radius(
    xyz: np.ndarray, radial: np.ndarray, cfg: PipelineConfig,
) -> float:
    if not cfg.use_adaptive_stem_radius:
        return cfg.stem_radial_max
    z0, z1 = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    base_mask = xyz[:, 2] <= z0 + cfg.root_density_z_frac * (z1 - z0)
    base_radial = radial[base_mask]
    if len(base_radial) < 10:
        base_radial = radial
    core_thresh = float(np.percentile(base_radial, 30))
    core = base_radial[base_radial <= core_thresh]
    if len(core) < 5:
        core = base_radial
    est = float(np.median(core) + cfg.adaptive_stem_radius_sigma * np.std(core))
    floor = cfg.stem_radial_max * cfg.stem_radial_floor_factor
    cap = cfg.stem_radial_max * cfg.stem_radial_cap_factor
    return float(np.clip(est, max(floor, cfg.voxel_size * 2.0), cap))


def _normalize_01(values: np.ndarray, invert: bool = False) -> np.ndarray:
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-12:
        out = np.ones_like(values, dtype=np.float64)
    else:
        out = (values - vmin) / (vmax - vmin)
    return 1.0 - out if invert else out


def _compute_stem_scores(
    radial: np.ndarray, linearity: np.ndarray, curvature: np.ndarray,
    verticality_deg: np.ndarray, stem_radius: float, cfg: PipelineConfig,
) -> np.ndarray:
    radial_score = np.clip(1.0 - radial / max(stem_radius, 1e-9), 0.0, 1.0)
    linearity_score = _normalize_01(linearity)
    curvature_score = _normalize_01(curvature, invert=True)
    verticality_score = _normalize_01(np.abs(verticality_deg - 90.0), invert=True)

    w_r = cfg.stem_score_weight_radial
    w_l = cfg.stem_score_weight_linearity
    w_c = cfg.stem_score_weight_curvature
    w_v = cfg.stem_score_weight_verticality
    w_sum = w_r + w_l + w_c + w_v
    if w_sum < 1e-12:
        w_r, w_l, w_c, w_v = 0.4, 0.3, 0.2, 0.1
        w_sum = 1.0

    return (
        w_r * radial_score + w_l * linearity_score
        + w_c * curvature_score + w_v * verticality_score
    ) / w_sum


def _region_grow_stem(
    xyz: np.ndarray, normals: np.ndarray, radial: np.ndarray,
    linearity: np.ndarray, curvature: np.ndarray, planarity: np.ndarray,
    initial_stem: np.ndarray, stem_radius: float, cfg: PipelineConfig,
) -> np.ndarray:
    """
    OPTIMISED region growing: builds k-NN once, uses NumPy masks for
    candidate pre-filtering, then only tests border points each round.
    """
    n = len(xyz)
    stem_mask = initial_stem.copy()

    seed_mask = (
        initial_stem
        & (linearity >= cfg.stem_seed_linearity_min)
        & (curvature <= cfg.stem_seed_curvature_max)
        & (radial <= stem_radius * cfg.stem_seed_radial_frac)
    )
    if not np.any(seed_mask):
        seed_mask = initial_stem & (radial <= stem_radius * 0.7)
    if not np.any(seed_mask):
        return stem_mask

    k = min(cfg.region_grow_k + 1, n)
    nn_obj = NearestNeighbors(n_neighbors=k).fit(xyz)
    _, nn_idx = nn_obj.kneighbors(xyz)

    # Pre-compute rejection mask: points that can never be stem
    never_stem = (radial > stem_radius) | (
        (planarity >= cfg.leaf_planarity_min)
        & (curvature > cfg.stem_curvature_max * 0.7)
    )

    queue = list(np.where(seed_mask)[0])
    visited = np.zeros(n, dtype=bool)
    visited[queue] = True
    stem_mask[queue] = True

    while queue:
        i = queue.pop()
        ri, ci, li, ni = radial[i], curvature[i], linearity[i], normals[i]
        neighbours = nn_idx[i]
        for j in neighbours:
            j = int(j)
            if visited[j] or never_stem[j]:
                continue
            normal_sim = float(np.dot(ni, normals[j]))
            if normal_sim < cfg.region_grow_normal_sim_min:
                continue
            if abs(curvature[j] - ci) > cfg.region_grow_curvature_sim:
                continue
            if abs(linearity[j] - li) > cfg.region_grow_linearity_sim:
                continue
            if abs(radial[j] - ri) > cfg.region_grow_radial_sim:
                continue
            visited[j] = True
            stem_mask[j] = True
            queue.append(j)

    return stem_mask


def _refine_stem_mask(
    stem_mask: np.ndarray, radial: np.ndarray, planarity: np.ndarray,
    curvature: np.ndarray, stem_radius: float, cfg: PipelineConfig,
) -> np.ndarray:
    refined = stem_mask.copy()
    refined &= radial <= stem_radius
    leaf_like = (planarity >= cfg.leaf_planarity_min) & (curvature >= cfg.stem_curvature_max * 0.75)
    refined &= ~leaf_like
    return refined


def _remove_small_leaf_components(
    xyz: np.ndarray, leaf_mask: np.ndarray, cfg: PipelineConfig,
) -> np.ndarray:
    """OPTIMISED: scipy connected_components instead of manual union-find."""
    leaf_idx = np.where(leaf_mask)[0]
    if len(leaf_idx) < cfg.min_leaf_component_size:
        return leaf_mask

    leaf_pts = xyz[leaf_idx]
    k = min(cfg.leaf_component_graph_k + 1, len(leaf_pts))
    adj = kneighbors_graph(leaf_pts, n_neighbors=k, mode="distance", include_self=False)
    adj = adj.maximum(adj.T)

    # Filter edges by max distance
    cx = adj.tocoo()
    mask = cx.data <= cfg.graph_max_edge
    filtered = csr_matrix(
        (cx.data[mask], (cx.row[mask], cx.col[mask])), shape=adj.shape
    )

    n_comp, comp_labels = connected_components(filtered, directed=False)
    comp_sizes = np.bincount(comp_labels, minlength=n_comp)

    cleaned = leaf_mask.copy()
    small_comps = np.where(comp_sizes < cfg.min_leaf_component_size)[0]
    for c in small_comps:
        local_idx = np.where(comp_labels == c)[0]
        cleaned[leaf_idx[local_idx]] = False

    return cleaned


class StemAxis:
    """Stem centreline as XY(z): linear between knots, straight beyond the ends."""

    def __init__(self, z: np.ndarray, xy: np.ndarray, max_slope: float = 1.0) -> None:
        self.z = np.asarray(z, dtype=np.float64)
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        k = min(5, len(self.z))
        self.slope_low = self._end_slope(self.z[:k], self.xy[:k], max_slope)
        self.slope_high = self._end_slope(self.z[-k:], self.xy[-k:], max_slope)

    @staticmethod
    def _end_slope(z: np.ndarray, xy: np.ndarray, max_slope: float) -> np.ndarray:
        if len(z) < 2 or float(np.ptp(z)) < 1e-9:
            return np.zeros(2, dtype=np.float64)
        design = np.column_stack([z - z.mean(), np.ones(len(z))])
        slope = np.linalg.lstsq(design, xy, rcond=None)[0][0]
        norm = float(np.linalg.norm(slope))
        return slope if norm <= max_slope else slope * (max_slope / norm)

    def center_at(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        c = np.column_stack(
            [np.interp(z, self.z, self.xy[:, 0]), np.interp(z, self.z, self.xy[:, 1])]
        )
        low, high = z < self.z[0], z > self.z[-1]
        c[low] = self.xy[0] + (z[low] - self.z[0])[:, None] * self.slope_low
        c[high] = self.xy[-1] + (z[high] - self.z[-1])[:, None] * self.slope_high
        return c

    def distance(self, xyz: np.ndarray) -> np.ndarray:
        """Horizontal distance of each point to the axis at the point's height."""
        return np.linalg.norm(xyz[:, :2] - self.center_at(xyz[:, 2]), axis=1)


def _fit_stem_axis(stem_pts: np.ndarray, cfg: PipelineConfig) -> StemAxis | None:
    """Median XY of the stem points per z-slice, lightly smoothed."""
    if len(stem_pts) < 10:
        return None
    z = stem_pts[:, 2]
    step = cfg.stem_axis_z_step
    edges = np.arange(float(z.min()), float(z.max()) + step, step)
    zs: list[float] = []
    centers: list[np.ndarray] = []
    for a, b in zip(edges[:-1], edges[1:]):
        band = stem_pts[(z >= a) & (z < b)]
        if len(band) < cfg.stem_axis_min_slice_points:
            continue
        zs.append(float(band[:, 2].mean()))
        centers.append(np.median(band[:, :2], axis=0))
    if len(zs) < 3:
        return None
    xy = np.asarray(centers, dtype=np.float64)
    padded = np.pad(xy, ((1, 1), (0, 0)), mode="edge")
    xy = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    return StemAxis(np.asarray(zs), xy, cfg.stem_axis_max_slope)


def _keep_main_stem_components(
    xyz: np.ndarray, stem_mask: np.ndarray, cfg: PipelineConfig,
) -> np.ndarray:
    """Drop stem islands that are not connected to the main stem body."""
    idx = np.where(stem_mask)[0]
    if len(idx) < 3:
        return stem_mask
    max_edge = cfg.stem_component_max_edge_factor * cfg.voxel_size
    pairs = cKDTree(xyz[idx]).query_pairs(max_edge, output_type="ndarray")
    if len(pairs) == 0:
        return stem_mask
    adj = csr_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx))
    )
    _, comp = connected_components(adj, directed=False)
    sizes = np.bincount(comp)
    keep = sizes >= max(cfg.stem_component_min_frac * sizes.max(), 1.0)
    out = np.zeros_like(stem_mask)
    out[idx[keep[comp]]] = True
    return out


def _stem_pass(
    xyz: np.ndarray, normals: np.ndarray, pca: PCAFeatures,
    verticality_deg: np.ndarray, radial: np.ndarray, stem_radius: float,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Score -> threshold -> region grow -> refine, for a given axis distance."""
    score = _compute_stem_scores(
        radial, pca.linearity, pca.curvature, verticality_deg, stem_radius, cfg,
    )
    stem_mask = score >= cfg.stem_score_threshold
    if cfg.use_region_growing:
        stem_mask = _region_grow_stem(
            xyz, normals, radial, pca.linearity, pca.curvature,
            pca.planarity, stem_mask, stem_radius, cfg,
        )
    stem_mask = _refine_stem_mask(
        stem_mask, radial, pca.planarity, pca.curvature, stem_radius, cfg
    )
    return stem_mask, score


def segment_stem_and_leaf(
    prep: PreprocessResult, cfg: PipelineConfig,
) -> SegmentationResult:
    xyz = prep.xyz_work
    normals = prep.normals
    stem_xy = _stem_xy_from_density(prep.xyz_clean, cfg)
    radial = np.linalg.norm(xyz[:, :2] - stem_xy, axis=1)

    scales = cfg.feature_scales if cfg.feature_scales else (cfg.feature_k,)
    pca, _ = _compute_multiscale_pca_features(xyz, scales)
    verticality_deg = _compute_verticality_deg(normals)
    verticality = 1.0 - np.abs(normals[:, 2])

    # Pass 0: wide corridor around a vertical line through the densest base cell.
    stem_radius = _estimate_adaptive_stem_radius(xyz, radial, cfg)
    stem_mask, stem_score_out = _stem_pass(
        xyz, normals, pca, verticality_deg, radial, stem_radius, cfg,
    )

    # Passes 1..n: re-fit the axis to the current stem and segment again inside
    # a narrow corridor around that curve. The axis is extrapolated past its
    # ends, so each pass can follow a leaning stem a little further.
    stem_axis: StemAxis | None = None
    if cfg.use_curved_stem_axis:
        for _ in range(cfg.stem_axis_iterations):
            axis = _fit_stem_axis(xyz[stem_mask], cfg)
            if axis is None:
                break
            axis_radial = axis.distance(xyz)
            new_mask, new_score = _stem_pass(
                xyz, normals, pca, verticality_deg, axis_radial,
                cfg.stem_axis_radius, cfg,
            )
            new_mask = _keep_main_stem_components(xyz, new_mask, cfg)
            if np.count_nonzero(new_mask) < 10:
                break
            changed = int(np.count_nonzero(new_mask ^ stem_mask))
            stem_axis, radial, stem_radius = axis, axis_radial, cfg.stem_axis_radius
            stem_mask, stem_score_out = new_mask, new_score
            if changed <= 0.01 * np.count_nonzero(stem_mask):
                break

    leaf_mask = (~stem_mask) & (
        (pca.planarity >= cfg.leaf_planarity_min)
        | (radial > stem_radius * 0.8)
        | (pca.curvature > cfg.stem_curvature_max)
    )
    unassigned = ~(stem_mask | leaf_mask)
    leaf_mask[unassigned & (radial > stem_radius * 0.5)] = True
    if stem_axis is None:
        stem_mask[unassigned & ~leaf_mask] = True
    else:
        # Points near the axis that failed the stem test (or sit on a dropped
        # island) are not stem; let leaf clustering decide where they belong.
        leaf_mask[unassigned & ~leaf_mask] = True
    leaf_mask = _remove_small_leaf_components(xyz, leaf_mask, cfg)

    return SegmentationResult(
        stem_mask=stem_mask, leaf_mask=leaf_mask,
        stem_points=xyz[stem_mask], leaf_points=xyz[leaf_mask],
        verticality=verticality, planarity=pca.planarity,
        linearity=pca.linearity, curvature=pca.curvature,
        scattering=pca.scattering, stem_xy=stem_xy,
        stem_score=stem_score_out, radial=radial,
        verticality_deg=verticality_deg,
        stem_axis=stem_axis, stem_radius=float(stem_radius),
    )


def visualize_segmentation(prep: PreprocessResult, seg: SegmentationResult) -> None:
    geoms: list[o3d.geometry.Geometry] = []
    if np.any(seg.stem_mask):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(prep.xyz_work[seg.stem_mask])
        p.paint_uniform_color([0.1, 0.85, 0.2])
        geoms.append(p)
    if np.any(seg.leaf_mask):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(prep.xyz_work[seg.leaf_mask])
        p.paint_uniform_color([0.9, 0.15, 0.15])
        geoms.append(p)
    o3d.visualization.draw_geometries(geoms, window_name="Segmentation (green=stem, red=leaf)", width=1280, height=800)


# =============================================================================
# Step 3 – k-NN graph  (OPTIMISED: bulk edge insertion)
# =============================================================================


def build_plant_graph(xyz: np.ndarray, cfg: PipelineConfig) -> nx.Graph:
    k = min(cfg.graph_k + 1, len(xyz))
    adj = kneighbors_graph(xyz, n_neighbors=k, mode="distance", include_self=False)
    cx = adj.tocoo()
    mask = cx.data <= cfg.graph_max_edge
    rows, cols, data = cx.row[mask], cx.col[mask], cx.data[mask]

    g = nx.Graph()
    g.add_nodes_from(range(len(xyz)))
    edges = [
        (int(r), int(c), {"weight": float(w)})
        for r, c, w in zip(rows, cols, data)
    ]
    g.add_edges_from(edges)
    return g


def visualize_graph_overlay(xyz: np.ndarray, graph: nx.Graph, sample_edges: int = 2000) -> None:
    pts = [o3d.geometry.PointCloud()]
    pts[0].points = o3d.utility.Vector3dVector(xyz)
    pts[0].paint_uniform_color([0.75, 0.75, 0.75])
    edges = list(graph.edges())
    if len(edges) > sample_edges:
        rng = np.random.default_rng(0)
        edges = [edges[i] for i in rng.choice(len(edges), sample_edges, replace=False)]
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(xyz)
    lines.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    lines.paint_uniform_color([0.2, 0.2, 0.9])
    o3d.visualization.draw_geometries(pts + [lines], window_name="k-NN graph overlay", width=1280, height=800)


# =============================================================================
# Step 4 – Stem centerline (Laplacian contraction)  OPTIMISED
# =============================================================================


def laplacian_contract(
    points: np.ndarray, k: int, iterations: int,
    contract_weight: float, volume_weight: float,
) -> np.ndarray:
    """OPTIMISED: vectorised Laplacian via mean of neighbours array."""
    pts = np.asarray(points, dtype=np.float64).copy()
    original = pts.copy()
    n = len(pts)
    if n < 4:
        return pts
    k = min(k + 1, n)
    for _ in range(iterations):
        tree = cKDTree(pts)
        _, nn = tree.query(pts, k=k)
        nn = nn[:, 1:] if k > 1 else nn
        # Vectorised: mean of neighbours minus self
        lap = pts[nn].mean(axis=1) - pts
        pts = pts + contract_weight * lap
        pts = (1.0 - volume_weight) * pts + volume_weight * original
    return pts


def extract_stem_centerline(
    seg: SegmentationResult, cfg: PipelineConfig,
) -> StemSkeletonResult:
    stem_pts = seg.stem_points
    if len(stem_pts) < 10:
        raise RuntimeError("Too few stem points for centerline extraction.")

    contracted = laplacian_contract(
        stem_pts, k=cfg.laplacian_k, iterations=cfg.laplacian_iterations,
        contract_weight=cfg.laplacian_contract, volume_weight=cfg.laplacian_volume,
    )

    z0, z1 = float(contracted[:, 2].min()), float(contracted[:, 2].max())
    edges = np.arange(z0, z1 + cfg.stem_z_step, cfg.stem_z_step)
    centers: list[np.ndarray] = []
    for i in range(len(edges) - 1):
        band = contracted[(contracted[:, 2] >= edges[i]) & (contracted[:, 2] < edges[i + 1])]
        if len(band) < 2:
            continue
        centers.append(band.mean(axis=0))

    if len(centers) < 2:
        order = np.argsort(contracted[:, 2])
        subs = contracted[order][:: max(1, len(contracted) // 40)]
        points = subs
    else:
        points = np.asarray(centers, dtype=np.float64)
        w = min(cfg.stem_smooth_window, len(points))
        if w >= 3 and w % 2 == 1:
            kernel = np.ones(w) / w
            half = w // 2
            for d in (0, 1):
                # Edge padding: zero padding would pull both stem ends towards x=y=0.
                padded = np.pad(points[:, d], half, mode="edge")
                points[:, d] = np.convolve(padded, kernel, mode="valid")

    for i in range(1, len(points)):
        if points[i, 2] < points[i - 1, 2]:
            points[i, 2] = points[i - 1, 2] + 1e-6

    return StemSkeletonResult(points=points, ordered_indices=list(range(len(points))))


def visualize_stem_skeleton(stem: StemSkeletonResult) -> None:
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(stem.points)
    pts.paint_uniform_color([0.1, 0.35, 0.95])
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(stem.points)
    if len(stem.points) >= 2:
        lines.lines = o3d.utility.Vector2iVector(
            np.asarray([[i, i + 1] for i in range(len(stem.points) - 1)], dtype=np.int32)
        )
    lines.paint_uniform_color([0.1, 0.35, 0.95])
    o3d.visualization.draw_geometries([pts, lines], window_name="Stem skeleton (blue)", width=1280, height=800)


# =============================================================================
# Step 5 – Leaf cluster detection
# =============================================================================


def _adaptive_dbscan_params(leaf: np.ndarray, cfg: PipelineConfig) -> tuple[float, int]:
    if not cfg.use_adaptive_dbscan or len(leaf) < 5:
        return cfg.leaf_cluster_eps, cfg.leaf_cluster_min_points
    k_nn = min(20, len(leaf))
    nn = NearestNeighbors(n_neighbors=k_nn).fit(leaf)
    dists, _ = nn.kneighbors(leaf)
    avg_nn = float(np.median(dists[:, 1])) if dists.shape[1] > 1 else cfg.voxel_size
    eps = max(
        cfg.dbscan_eps_nn_factor * avg_nn,
        cfg.dbscan_eps_voxel_factor * cfg.voxel_size,
        cfg.leaf_cluster_eps * 0.5,
    )
    local_count = np.sum(dists <= eps, axis=1)
    local_density = float(np.median(local_count))
    min_samples = int(np.clip(
        local_density * 0.4,
        cfg.leaf_cluster_min_points,
        max(len(leaf) // 4, cfg.leaf_cluster_min_points),
    ))
    min_samples = max(min_samples, 5)
    return eps, min_samples


def _angular_span_deg(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 0.0
    sorted_a = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_a, sorted_a[:1] + 2.0 * np.pi]))
    return float(np.degrees(2.0 * np.pi - gaps.max()))


def _min_pairwise_angle_sep_deg(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 360.0
    sorted_a = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_a, sorted_a[:1] + 2.0 * np.pi]))
    return float(np.degrees(gaps.max()))


def _split_cluster_by_stem_angle(
    idx: np.ndarray, leaf: np.ndarray, stem_xy: np.ndarray, cfg: PipelineConfig,
) -> list[np.ndarray]:
    """Split one cluster (indices into ``leaf``) by azimuth around the stem."""
    min_pts = cfg.leaf_cluster_min_points
    if len(idx) < max(cfg.leaf_split_min_cluster_size, min_pts * 2):
        return [idx]
    cluster = leaf[idx]
    angles = np.arctan2(cluster[:, 1] - stem_xy[1], cluster[:, 0] - stem_xy[0])
    if _angular_span_deg(angles) < cfg.leaf_split_min_angle_deg:
        return [idx]
    feat = np.column_stack([np.cos(angles), np.sin(angles)])
    best: list[np.ndarray] = [idx]
    max_k = min(cfg.leaf_split_max_k, len(idx) // min_pts)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sub_labels = km.fit_predict(feat)
        parts = [idx[sub_labels == i] for i in range(k)]
        if not all(len(p) >= min_pts for p in parts):
            continue
        cent_angles = np.array([
            np.arctan2(leaf[p, 1].mean() - stem_xy[1], leaf[p, 0].mean() - stem_xy[0])
            for p in parts
        ])
        if _min_pairwise_angle_sep_deg(cent_angles) >= cfg.leaf_split_min_angle_deg * 0.6:
            best = parts
            break
    return best


def _split_nearby_leaf_clusters(
    groups: list[np.ndarray], leaf: np.ndarray, stem_xy: np.ndarray,
    cfg: PipelineConfig,
) -> list[np.ndarray]:
    if not cfg.split_nearby_leaves:
        return groups
    split: list[np.ndarray] = []
    for block in groups:
        parts = _split_cluster_by_stem_angle(block, leaf, stem_xy, cfg)
        for part in parts:
            if len(part) >= cfg.leaf_split_min_cluster_size:
                sub = _split_cluster_by_stem_angle(part, leaf, stem_xy, cfg)
                split.extend(sub if len(sub) > 1 else [part])
            else:
                split.append(part)
    split = [g for g in split if len(g) >= cfg.leaf_cluster_min_points]
    split.sort(key=len, reverse=True)
    return split[: cfg.leaf_max_clusters]


def _merge_detached_leaf_fragments(
    groups: list[np.ndarray], leaf: np.ndarray, seg: SegmentationResult,
    cfg: PipelineConfig,
) -> list[np.ndarray]:
    """
    A leaf has to reach the stem. Clusters that do not are fragments of a leaf
    that DBSCAN cut at a gap: merge each into the nearest stem-attached cluster,
    and stop reporting small leftovers as leaves of their own.
    """
    stem_pts = seg.stem_points
    if not cfg.merge_detached_leaf_fragments or len(groups) < 2 or len(stem_pts) == 0:
        return groups
    stem_tree = cKDTree(stem_pts)
    attach = np.array([float(stem_tree.query(leaf[g], k=1)[0].min()) for g in groups])
    attached = [g for g, d in zip(groups, attach) if d <= cfg.leaf_attach_max_dist]
    detached = [g for g, d in zip(groups, attach) if d > cfg.leaf_attach_max_dist]
    if not attached:
        return groups

    merged = True
    while merged and detached:
        merged = False
        trees = [cKDTree(leaf[g]) for g in attached]
        still: list[np.ndarray] = []
        for frag in detached:
            gaps = np.array([float(t.query(leaf[frag], k=1)[0].min()) for t in trees])
            j = int(np.argmin(gaps))
            if gaps[j] <= cfg.leaf_fragment_merge_gap:
                attached[j] = np.concatenate([attached[j], frag])
                merged = True
            else:
                still.append(frag)
        detached = still

    kept = attached + [g for g in detached if len(g) >= cfg.leaf_detached_keep_min_points]
    kept.sort(key=len, reverse=True)
    return kept


def detect_leaf_clusters(seg: SegmentationResult, cfg: PipelineConfig) -> LeafClusterResult:
    leaf = seg.leaf_points
    point_labels = np.full(len(leaf), -1, dtype=np.int32)
    if len(leaf) < cfg.leaf_cluster_min_points:
        return LeafClusterResult(clusters=[], labels=np.array([], dtype=int), point_labels=point_labels)
    eps, min_samples = _adaptive_dbscan_params(leaf, cfg)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(leaf)
    groups: list[np.ndarray] = []
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) >= cfg.leaf_cluster_min_points:
            groups.append(idx)
    groups.sort(key=len, reverse=True)
    groups = groups[: cfg.leaf_max_clusters]
    groups = _split_nearby_leaf_clusters(groups, leaf, seg.stem_xy, cfg)
    groups = _merge_detached_leaf_fragments(groups, leaf, seg, cfg)
    for cid, idx in enumerate(groups):
        point_labels[idx] = cid
    return LeafClusterResult(
        clusters=[leaf[idx] for idx in groups], labels=labels, point_labels=point_labels,
    )


def segment_and_cluster(
    prep: PreprocessResult, cfg: PipelineConfig,
) -> tuple[SegmentationResult, LeafClusterResult]:
    """Stem/leaf segmentation followed by leaf clustering, with one feedback step."""
    seg = segment_stem_and_leaf(prep, cfg)
    leaf = detect_leaf_clusters(seg, cfg)
    if (
        not cfg.reassign_stem_hugging_clusters
        or seg.stem_axis is None
        or leaf.point_labels is None
        or not leaf.clusters
    ):
        return seg, leaf

    xyz = prep.xyz_work
    leaf_idx = np.where(seg.leaf_mask)[0]
    z_low, z_high = float(seg.stem_axis.z[0]), float(seg.stem_axis.z[-1])
    limit = cfg.stem_hugging_radius_factor * seg.stem_radius
    to_stem = np.zeros(len(xyz), dtype=bool)
    for cid, cluster in enumerate(leaf.clusters):
        near = seg.stem_axis.distance(cluster) <= limit
        # Above the stem top a cluster on the axis is a leaf growing upwards.
        if near.mean() >= cfg.stem_hugging_min_frac and float(cluster[:, 2].max()) <= z_high:
            to_stem[leaf_idx[leaf.point_labels == cid]] = True
    if not np.any(to_stem):
        return seg, leaf

    stem_mask = seg.stem_mask | to_stem
    leaf_mask = seg.leaf_mask & ~to_stem
    seg = replace(
        seg, stem_mask=stem_mask, leaf_mask=leaf_mask,
        stem_points=xyz[stem_mask], leaf_points=xyz[leaf_mask],
    )
    return seg, detect_leaf_clusters(seg, cfg)


def cluster_colors(n: int) -> np.ndarray:
    """Return ``n`` leaf colors from the 50-color palette (cycles if n > 50)."""
    if n <= 0:
        return np.empty((0, 3))
    return _LEAF_PALETTE[np.arange(n) % len(_LEAF_PALETTE)]


def visualize_leaf_clusters(clusters: list[np.ndarray]) -> None:
    geoms: list[o3d.geometry.Geometry] = []
    cols = cluster_colors(len(clusters))
    for i, c in enumerate(clusters):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(c)
        p.paint_uniform_color(cols[i].tolist())
        geoms.append(p)
    o3d.visualization.draw_geometries(geoms, window_name="Leaf clusters", width=1280, height=800)


# =============================================================================
# Step 6 – Stem–leaf junction detection
# =============================================================================


def _closest_on_polyline(
    points: np.ndarray, poly: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point: closest point on the polyline, distance, fractional vertex index."""
    if len(poly) == 1:
        proj = np.repeat(poly[:1], len(points), axis=0)
        return proj, np.linalg.norm(points - proj, axis=1), np.zeros(len(points))
    a, ab = poly[:-1], poly[1:] - poly[:-1]
    denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
    t = np.einsum("nsj,sj->ns", points[:, None, :] - a[None], ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    dist = np.linalg.norm(points[:, None, :] - proj, axis=2)
    seg = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    return proj[rows, seg], dist[rows, seg], seg + t[rows, seg]


def find_leaf_junctions(
    stem_points: np.ndarray, clusters: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Junction of each leaf = projection onto the stem centreline of the leaf
    point that comes closest to it. Returns (junctions on the stem, nearest
    stem vertex index, the leaf-side attachment points).
    """
    n = len(clusters)
    if n == 0 or len(stem_points) == 0:
        return np.empty((0, 3)), np.empty(0, dtype=int), np.empty((0, 3))
    junctions = np.zeros((n, 3), dtype=np.float64)
    attach = np.zeros((n, 3), dtype=np.float64)
    stem_indices = np.zeros(n, dtype=int)
    for i, cluster in enumerate(clusters):
        proj, dist, frac = _closest_on_polyline(cluster, stem_points)
        j = int(np.argmin(dist))
        attach[i] = cluster[j]
        junctions[i] = proj[j]
        stem_indices[i] = int(np.clip(round(float(frac[j])), 0, len(stem_points) - 1))
    return junctions, stem_indices, attach


# =============================================================================
# Step 7 – Leaf skeleton (geodesic midrib)
# =============================================================================


def _leaf_knn_graph(points: np.ndarray, k: int, max_edge: float) -> tuple[csr_matrix, np.ndarray]:
    k = min(k + 1, len(points))
    adj = kneighbors_graph(points, n_neighbors=k, mode="distance", include_self=False)
    adj = adj.maximum(adj.T)
    cx = adj.tocoo()
    mask = cx.data <= max_edge
    rows, cols, data = cx.row[mask], cx.col[mask], cx.data[mask]
    return csr_matrix((data, (rows, cols)), shape=adj.shape), points


def extract_leaf_midrib_path(
    cluster: np.ndarray, junction: np.ndarray, cfg: PipelineConfig,
) -> np.ndarray | None:
    if len(cluster) < 5:
        return None
    csr, pts = _leaf_knn_graph(cluster, cfg.leaf_graph_k, cfg.leaf_graph_max_edge)
    tree = cKDTree(pts)
    _, start = tree.query(junction.reshape(1, 3), k=1)
    start = int(start[0])

    dist, pred = shortest_path(csr, directed=False, indices=start, return_predecessors=True)
    finite = np.isfinite(dist)
    if not np.any(finite):
        return None
    tip = int(np.argmax(np.where(finite, dist, -1.0)))

    path_rev: list[int] = [tip]
    while path_rev[-1] != start:
        p = int(pred[path_rev[-1]])
        if p < 0:
            return None
        path_rev.append(p)
    return pts[path_rev[::-1]]


def extract_all_leaf_skeletons(
    clusters: list[np.ndarray], junctions: np.ndarray,
    junction_stem_indices: np.ndarray, attach_points: np.ndarray,
    cfg: PipelineConfig,
) -> LeafSkeletonResult:
    paths: list[np.ndarray] = []
    tips: list[np.ndarray] = []
    kept_j: list[np.ndarray] = []
    kept_si: list[int] = []
    kept_ci: list[int] = []

    for i, cluster in enumerate(clusters):
        if i >= len(junctions):
            break
        junc = junctions[i]
        path = extract_leaf_midrib_path(cluster, attach_points[i], cfg)
        if path is None or len(path) < 2:
            continue
        path = np.vstack([junc, path])
        paths.append(path)
        tips.append(path[-1])
        kept_j.append(junc)
        kept_si.append(int(junction_stem_indices[i]))
        kept_ci.append(i)

    return LeafSkeletonResult(
        junctions=np.asarray(kept_j) if kept_j else np.empty((0, 3)),
        junction_stem_indices=np.asarray(kept_si, dtype=int),
        paths=paths,
        tips=np.asarray(tips) if tips else np.empty((0, 3)),
        cluster_indices=kept_ci,
    )


# =============================================================================
# Step 8 – Skeleton smoothing
# =============================================================================


def smooth_path(path: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if len(path) < window:
        return path.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(path) if len(path) % 2 == 1 else len(path) - 1)
    if window < polyorder + 2:
        return path.copy()
    out = np.zeros_like(path)
    for d in range(3):
        out[:, d] = savgol_filter(path[:, d], window_length=window, polyorder=polyorder)
    out[0], out[-1] = path[0], path[-1]  # smoothing must not move the end points
    return out


def smooth_leaf_path(path: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Smooth the leaf part only; path[0] is the junction on the stem and stays put."""
    if len(path) < 3:
        return path.copy()
    return np.vstack([path[:1], smooth_path(path[1:], window, polyorder)])


def smooth_skeletons(
    stem: np.ndarray, leaf_paths: list[np.ndarray], cfg: PipelineConfig,
) -> tuple[np.ndarray, list[np.ndarray]]:
    sm_stem = smooth_path(stem, cfg.smooth_window, cfg.smooth_polyorder)
    sm_leaves = [smooth_leaf_path(p, cfg.smooth_window, cfg.smooth_polyorder) for p in leaf_paths]
    return sm_stem, sm_leaves


# =============================================================================
# Step 9 – Final plant skeleton graph (tree)
# =============================================================================


def build_plant_skeleton_graph(
    stem: np.ndarray, leaf_sk: LeafSkeletonResult,
    leaf_paths: list[np.ndarray] | None = None,
) -> PlantSkeletonGraph:
    g = nx.Graph()
    positions: list[np.ndarray] = []
    roles: list[str] = []
    stem_edges: list[tuple[int, int]] = []
    leaf_edges: list[tuple[int, int]] = []

    for i, p in enumerate(stem):
        nid = len(positions)
        positions.append(p)
        roles.append("stem_base" if i == 0 else ("stem_top" if i == len(stem) - 1 else "stem"))
        g.add_node(nid, role=roles[-1], z=float(p[2]))
        if i > 0:
            g.add_edge(nid - 1, nid, edge_type="stem", weight=float(np.linalg.norm(p - stem[i - 1])))
            stem_edges.append((nid - 1, nid))

    root = 0
    # path[0] is the junction on the stem; it is represented by the stem node.
    for li, path in enumerate(leaf_paths if leaf_paths is not None else leaf_sk.paths):
        if len(path) < 2:
            continue
        j_stem = int(leaf_sk.junction_stem_indices[li]) if li < len(leaf_sk.junction_stem_indices) else 0
        j_stem = min(j_stem, len(stem) - 1)
        parent = j_stem
        prev = parent
        for k_idx, pt in enumerate(path):
            if k_idx == 0:
                continue
            nid = len(positions)
            positions.append(pt)
            role = "leaf_tip" if k_idx == len(path) - 1 else "leaf_midrib"
            roles.append(role)
            g.add_node(nid, role=role, leaf_id=li)
            w = float(np.linalg.norm(pt - positions[prev]))
            g.add_edge(prev, nid, edge_type="leaf", weight=w, leaf_id=li)
            leaf_edges.append((prev, nid))
            prev = nid

    if not nx.is_tree(g) and len(g) > 0:
        components = list(nx.connected_components(g))
        if len(components) > 1:
            main = max(components, key=len)
            for comp in components:
                if comp == main:
                    continue
                g.add_edge(root, min(comp), edge_type="bridge", weight=0.0)

    return PlantSkeletonGraph(
        graph=g,
        node_positions=np.asarray(positions, dtype=np.float64),
        node_roles=roles,
        stem_edge_indices=stem_edges,
        leaf_edge_indices=leaf_edges,
    )


# =============================================================================
# Step 10 – Measurements
# =============================================================================


def _path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _stem_tangent_at(stem: np.ndarray, idx: int, half_window: int = 3) -> np.ndarray:
    """Stem direction at a vertex, from the centreline points +-half_window around it."""
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if len(stem) < 2:
        return up
    idx = int(np.clip(idx, 0, len(stem) - 1))
    lo, hi = max(idx - half_window, 0), min(idx + half_window, len(stem) - 1)
    tangent = stem[hi] - stem[lo]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-9:
        return up
    return tangent / norm


def _leaf_direction_at_junction(path: np.ndarray, arc_length: float = 0.04) -> np.ndarray:
    """
    Direction in which the leaf leaves the stem: a straight line fitted to the
    first ``arc_length`` metres of the leaf. path[0] is the junction on the stem
    axis and is skipped, because junction -> leaf is perpendicular to the stem
    by construction and would bias every angle towards 90 degrees.
    """
    leaf = path[1:] if len(path) >= 3 else path
    if len(leaf) < 2:
        return np.zeros(3, dtype=np.float64)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(leaf, axis=0), axis=1))])
    n_use = max(int(np.searchsorted(arc, arc_length, side="right")), 2)
    window = leaf[:n_use]
    chord = window[-1] - window[0]
    direction = chord
    if n_use >= 3:
        _, _, vt = np.linalg.svd(window - window.mean(axis=0), full_matrices=False)
        direction = vt[0] if float(np.dot(vt[0], chord)) >= 0.0 else -vt[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return direction / norm


def _stem_leaf_angle_deg(
    stem: np.ndarray, stem_idx: int, leaf_path: np.ndarray,
    arc_length: float = 0.04, half_window: int = 3,
) -> float:
    stem_dir = _stem_tangent_at(stem, stem_idx, half_window)
    leaf_dir = _leaf_direction_at_junction(leaf_path, arc_length)
    if float(np.linalg.norm(leaf_dir)) < 1e-9:
        return 0.0
    cos_a = float(np.dot(stem_dir, leaf_dir))
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def compute_measurements(
    stem: np.ndarray, leaf_sk: LeafSkeletonResult,
    skel_graph: PlantSkeletonGraph,
    leaf_paths: list[np.ndarray] | None = None,
    plant_xyz: np.ndarray | None = None,
    cfg: PipelineConfig | None = None,
) -> Measurements:
    cfg = cfg or PipelineConfig()
    stem_len = _path_length(stem)
    # Lengths and angles both come from the smoothed paths when available.
    paths_for_angle = leaf_paths if leaf_paths is not None else leaf_sk.paths
    leaf_lengths = [_path_length(p) for p in paths_for_angle]

    if plant_xyz is not None and len(plant_xyz) > 0:
        # Percentiles, so that a single stray point cannot shift every height.
        q = cfg.height_trim_percentile
        z_bottom = float(np.percentile(plant_xyz[:, 2], q))
        z_top = float(np.percentile(plant_xyz[:, 2], 100.0 - q))
    elif len(stem) > 0:
        z_bottom = float(np.min(stem[:, 2]))
        z_top = float(np.max(stem[:, 2]))
        if len(leaf_sk.junctions) > 0:
            z_bottom = min(z_bottom, float(np.min(leaf_sk.junctions[:, 2])))
            z_top = max(z_top, float(np.max(leaf_sk.junctions[:, 2])))
    else:
        z_bottom = 0.0
        z_top = 0.0

    heights = [float(j[2] - z_bottom) for j in leaf_sk.junctions]
    angles = [
        _stem_leaf_angle_deg(
            stem, int(si), path,
            cfg.leaf_angle_arc_length, cfg.stem_tangent_half_window,
        )
        for path, si in zip(paths_for_angle, leaf_sk.junction_stem_indices)
    ]
    total = stem_len + sum(leaf_lengths)
    return Measurements(
        stem_length_m=stem_len, leaf_lengths_m=leaf_lengths,
        n_leaves=len(leaf_sk.paths), insertion_heights_m=heights,
        leaf_angles_deg=angles, total_skeleton_length_m=total,
        plant_bottom_z_m=z_bottom, plant_height_m=float(z_top - z_bottom),
    )


def export_leaf_measurements_csv(measurements: Measurements, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["leaf_id", "insertion_height_from_bottom_cm", "leaf_angle_deg", "geodesic_length_m"])
        for i, (height, angle, length) in enumerate(
            zip(measurements.insertion_heights_m, measurements.leaf_angles_deg, measurements.leaf_lengths_m),
            start=1,
        ):
            w.writerow([i, f"{height*100:.2f}", f"{angle:.2f}", f"{length:.6f}"])


def export_measurements_csv(measurements: Measurements, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        w.writerow(["stem_length", f"{measurements.stem_length_m:.6f}", "m"])
        w.writerow(["n_leaves", measurements.n_leaves, "count"])
        w.writerow(["total_skeleton_length", f"{measurements.total_skeleton_length_m:.6f}", "m"])
        w.writerow(["plant_bottom_z", f"{measurements.plant_bottom_z_m:.6f}", "m"])
        w.writerow(["plant_height", f"{measurements.plant_height_m:.6f}", "m"])
        for i, (L, h, a) in enumerate(
            zip(measurements.leaf_lengths_m, measurements.insertion_heights_m, measurements.leaf_angles_deg),
            start=1,
        ):
            w.writerow([f"leaf_{i}_length", f"{L:.6f}", "m"])
            w.writerow([f"leaf_{i}_insertion_height_from_bottom", f"{h*100:.2f}", "cm"])
            w.writerow([f"leaf_{i}_angle_from_stem", f"{a:.2f}", "deg"])


def save_outputs(result: PipelineResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ply(name: str, pts: np.ndarray, color: list[float]) -> None:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts)
        p.paint_uniform_color(color)
        o3d.io.write_point_cloud(str(out_dir / name), p)

    _ply("stem_points.ply", result.segmentation.stem_points, [0.1, 0.85, 0.2])
    _ply("leaf_points.ply", result.segmentation.leaf_points, [0.9, 0.15, 0.15])
    _ply("stem_skeleton.ply", result.smoothed_stem, [0.1, 0.35, 0.95])

    with (out_dir / "skeleton_graph_edges.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_a", "node_b", "edge_type", "weight"])
        for u, v, data in result.skeleton_graph.graph.edges(data=True):
            w.writerow([u, v, data.get("edge_type", ""), data.get("weight", 0.0)])

    nx.write_gexf(result.skeleton_graph.graph, out_dir / "plant_skeleton_graph.gexf")
    export_leaf_measurements_csv(result.measurements, out_dir / "leaf_measurements.csv")
    export_measurements_csv(result.measurements, out_dir / "measurements.csv")
    print(f"Outputs saved to {out_dir}")


# =============================================================================
# Step 11 – PyVista interactive viewer
# =============================================================================


def _populate_structure_scene(
    plotter: pv.Plotter, result: PipelineResult, prep: PreprocessResult,
    leaf_display_ids: list[int] | None = None,
) -> dict[str, Any]:
    actors: dict[str, Any] = {}

    def _add_pts(key: str, pts: np.ndarray, color: str, size: float = 3.0) -> None:
        if len(pts) == 0:
            return
        cloud = pv.PolyData(pts)
        actors[key] = plotter.add_points(cloud, color=color, point_size=size, render_points_as_spheres=True)

    def _add_colored_leaf_clusters(key: str, clusters: list[np.ndarray], size: float = 3.0) -> None:
        if not clusters:
            return
        colors = cluster_colors(len(clusters))
        pts_list: list[np.ndarray] = []
        rgb_list: list[np.ndarray] = []
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            pts_list.append(cluster)
            rgb_list.append(np.tile(colors[i], (len(cluster), 1)))
        if not pts_list:
            return
        cloud = pv.PolyData(np.vstack(pts_list))
        cloud["rgb"] = (np.vstack(rgb_list) * 255).astype(np.uint8)
        actors[key] = plotter.add_points(cloud, scalars="rgb", rgb=True, point_size=size, render_points_as_spheres=True)

    def _add_colored_leaf_skeletons(key: str, paths: list[np.ndarray], line_width: float = 5.0) -> None:
        if not paths:
            return
        colors = cluster_colors(len(paths))
        line_actors: list[Any] = []
        for i, path in enumerate(paths):
            if len(path) < 2:
                continue
            segments = [pv.Line(path[j], path[j + 1]) for j in range(len(path) - 1)]
            merged = segments[0]
            for seg in segments[1:]:
                merged = merged.merge(seg)
            line_actors.append(plotter.add_mesh(merged, color=tuple(colors[i]), line_width=line_width))
        if line_actors:
            actors[key] = line_actors

    def _add_leaf_number_labels(key: str, paths: list[np.ndarray]) -> None:
        if not paths:
            return
        stem_xy = result.segmentation.stem_xy
        label_pts: list[np.ndarray] = []
        label_text: list[str] = []
        for i, path in enumerate(paths):
            if len(path) == 0:
                continue
            mid_idx = max(0, int(len(path) * 0.55))
            pt = path[mid_idx].copy()
            radial = pt[:2] - stem_xy
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > 1e-9:
                pt[:2] += (radial / radial_norm) * 0.012
            pt[2] += 0.010
            label_pts.append(pt)
            if leaf_display_ids is not None and i < len(leaf_display_ids):
                label_text.append(f"Leaf {leaf_display_ids[i]}")
            else:
                label_text.append(f"Leaf {i + 1}")
        if not label_pts:
            return
        actors[key] = plotter.add_point_labels(
            np.asarray(label_pts), label_text,
            font_size=14, point_size=0, show_points=False,
            shape="rounded_rect", shape_opacity=0.85, margin=4,
            bold=True, text_color="white", shape_color="black",
            always_visible=True,
        )

    _add_pts("cloud", prep.xyz_clean, "lightgray", size=2.0)
    _add_pts("stem_pts", result.segmentation.stem_points, "green", size=4.0)
    _add_colored_leaf_clusters("leaf_pts", result.leaf_clusters.clusters, size=3.0)

    stem = result.smoothed_stem
    if len(stem) >= 2:
        stem_lines = [pv.Line(stem[i], stem[i + 1]) for i in range(len(stem) - 1)]
        merged = stem_lines[0]
        for seg in stem_lines[1:]:
            merged = merged.merge(seg)
        actors["stem_skel"] = plotter.add_mesh(merged, color="blue", line_width=5)

    _add_colored_leaf_skeletons("leaf_skel", result.smoothed_leaf_paths)
    _add_pts("junctions", result.leaf_skeleton.junctions, "magenta", size=8.0)
    _add_pts("tips", result.leaf_skeleton.tips, "cyan", size=8.0)
    _add_leaf_number_labels("leaf_labels", result.smoothed_leaf_paths)

    # Bottom and top point markers
    all_z = prep.xyz_clean[:, 2]
    bottom_idx = int(np.argmin(all_z))
    top_idx = int(np.argmax(all_z))
    bottom_pt = prep.xyz_clean[bottom_idx]
    top_pt = prep.xyz_clean[top_idx]

    for tag, pt, color in [("bottom_pt", bottom_pt, "red"), ("top_pt", top_pt, "lime")]:
        sphere = pv.Sphere(radius=0.004, center=pt)
        actors[tag] = plotter.add_mesh(sphere, color=color)

    actors["bottom_label"] = plotter.add_point_labels(
        bottom_pt.reshape(1, 3), ["Bottom (0 cm)"],
        font_size=12, point_size=0, show_points=False,
        shape="rounded_rect", shape_opacity=0.85, margin=3,
        bold=True, text_color="white", shape_color="red", always_visible=True,
    )
    actors["top_label"] = plotter.add_point_labels(
        top_pt.reshape(1, 3),
        [f"Top ({result.measurements.plant_height_m * 100:.1f} cm)"],
        font_size=12, point_size=0, show_points=False,
        shape="rounded_rect", shape_opacity=0.85, margin=3,
        bold=True, text_color="white", shape_color="green", always_visible=True,
    )

    # Junction labels with measurements
    junctions = result.leaf_skeleton.junctions
    if len(junctions) > 0:
        m = result.measurements
        hover_texts: list[str] = []
        for i in range(len(junctions)):
            lid = leaf_display_ids[i] if (leaf_display_ids and i < len(leaf_display_ids)) else i + 1
            h_cm = m.insertion_heights_m[i] * 100 if i < len(m.insertion_heights_m) else 0.0
            a_deg = m.leaf_angles_deg[i] if i < len(m.leaf_angles_deg) else 0.0
            hover_texts.append(
                f"Leaf {lid}\n"
                f"Insertion height: {h_cm:.2f} cm\n"
                f"Stem-leaf angle: {a_deg:.1f} deg"
            )
        actors["junction_hover"] = plotter.add_point_labels(
            junctions, hover_texts,
            font_size=11, point_size=0, show_points=False,
            shape="rounded_rect", shape_opacity=0.9, margin=4,
            bold=False, text_color="white", shape_color="purple",
            always_visible=True,
        )

    return actors


def pyvista_viewer(result: PipelineResult, prep: PreprocessResult) -> None:
    plotter = pv.Plotter(window_size=(1400, 900), title="Plant skeleton")
    actors = _populate_structure_scene(plotter, result, prep)
    visible = {k: True for k in actors}

    def _set_visibility(actor: Any, show: bool) -> None:
        if isinstance(actor, list):
            for a in actor:
                a.SetVisibility(show)
        else:
            actor.SetVisibility(show)

    def _toggle(name: str) -> None:
        if name not in actors:
            return
        visible[name] = not visible[name]
        _set_visibility(actors[name], visible[name])
        plotter.render()

    plotter.add_key_event("1", lambda: _toggle("cloud"))
    plotter.add_key_event("2", lambda: _toggle("stem_pts"))
    plotter.add_key_event("3", lambda: _toggle("leaf_pts"))
    plotter.add_key_event("4", lambda: _toggle("stem_skel"))
    plotter.add_key_event("5", lambda: _toggle("leaf_skel"))
    plotter.add_key_event("6", lambda: _toggle("junctions"))
    plotter.add_key_event("7", lambda: _toggle("tips"))
    plotter.add_key_event("8", lambda: _toggle("leaf_labels"))

    plotter.add_text(
        "Keys: 1=cloud 2=stem 3=leaves 4=stem skel 5=leaf skels 6=junctions 7=tips 8=labels",
        font_size=10,
    )
    plotter.show()


# =============================================================================
# Full pipeline (structure)
# =============================================================================


def run_structure_from_xyz(
    xyz: np.ndarray, cfg: PipelineConfig, *,
    rgb: np.ndarray | None = None,
    show_o3d_steps: bool = False, show_pyvista: bool = True,
) -> PipelineResult:
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz - xyz.mean(axis=0)
    print(f"[Structure 1] Preprocessing ({len(xyz):,} points)")
    prep = preprocess_point_cloud(xyz, cfg, rgb=rgb)
    print(f"    Clean: {len(prep.xyz_clean):,} | Work: {len(prep.xyz_work):,}")
    if show_o3d_steps:
        visualize_preprocessed(prep)

    print("[Structure 2] Stem / leaf segmentation")
    seg, leaf_clusters = segment_and_cluster(prep, cfg)
    print(f"    Stem: {len(seg.stem_points):,} | Leaf: {len(seg.leaf_points):,}")
    if show_o3d_steps:
        visualize_segmentation(prep, seg)

    print("[Structure 3] k-NN graph")
    plant_graph = build_plant_graph(prep.xyz_work, cfg)
    print(f"    Nodes: {plant_graph.number_of_nodes():,} | Edges: {plant_graph.number_of_edges():,}")
    if show_o3d_steps:
        visualize_graph_overlay(prep.xyz_work, plant_graph)

    print("[Structure 4] Stem centerline (Laplacian contraction)")
    stem_sk = extract_stem_centerline(seg, cfg)
    print(f"    Stem skeleton points: {len(stem_sk.points)}")
    if show_o3d_steps:
        visualize_stem_skeleton(stem_sk)

    print("[Structure 5] Leaf cluster detection")
    print(f"    Clusters: {len(leaf_clusters.clusters)}")
    if show_o3d_steps and leaf_clusters.clusters:
        visualize_leaf_clusters(leaf_clusters.clusters)

    print("[Structure 6] Stem-leaf junctions (projected onto the smoothed stem)")
    sm_stem = smooth_path(stem_sk.points, cfg.smooth_window, cfg.smooth_polyorder)
    junctions, j_stem_idx, attach_pts = find_leaf_junctions(sm_stem, leaf_clusters.clusters)
    print(f"    Junctions: {len(junctions)}")

    print("[Structure 7] Leaf midrib skeletons (geodesic paths)")
    leaf_sk = extract_all_leaf_skeletons(
        leaf_clusters.clusters, junctions, j_stem_idx, attach_pts, cfg,
    )
    print(f"    Leaf skeletons: {len(leaf_sk.paths)}")

    print("[Structure 8] Smoothing leaf skeletons")
    sm_leaves = [
        smooth_leaf_path(p, cfg.smooth_window, cfg.smooth_polyorder) for p in leaf_sk.paths
    ]

    print("[Structure 9] Plant skeleton graph")
    skel_graph = build_plant_skeleton_graph(sm_stem, leaf_sk, sm_leaves)
    print(f"    Graph nodes: {skel_graph.graph.number_of_nodes()} | edges: {skel_graph.graph.number_of_edges()}")
    print(f"    Is tree: {nx.is_tree(skel_graph.graph)}")

    print("[Structure 10] Measurements")
    measurements = compute_measurements(
        sm_stem, leaf_sk, skel_graph, sm_leaves, plant_xyz=prep.xyz_clean, cfg=cfg,
    )
    print(f"    Stem length: {measurements.stem_length_m:.3f} m")
    print(f"    Leaves: {measurements.n_leaves} | Total skeleton: {measurements.total_skeleton_length_m:.3f} m")

    result = PipelineResult(
        preprocess=prep, segmentation=seg, plant_graph=plant_graph,
        stem_skeleton=stem_sk, leaf_clusters=leaf_clusters,
        leaf_skeleton=leaf_sk, smoothed_stem=sm_stem,
        smoothed_leaf_paths=sm_leaves, skeleton_graph=skel_graph,
        measurements=measurements,
    )

    save_outputs(result, cfg.output_dir)

    if show_pyvista:
        print("[Structure 11] Opening PyVista structure viewer (keys 1-8)")
        pyvista_viewer(result, prep)

    return result


def run_pipeline(
    input_path: Path, cfg: PipelineConfig,
    show_o3d_steps: bool = False, show_pyvista: bool = True,
) -> PipelineResult:
    print(f"[1] Loading & preprocessing: {input_path}")
    xyz, rgb = load_point_cloud(input_path)
    print(f"    Raw points: {len(xyz):,}")
    return run_structure_from_xyz(xyz, cfg, rgb=rgb, show_o3d_steps=show_o3d_steps, show_pyvista=show_pyvista)


# =============================================================================
# IoU evaluation vs CloudCompare .bin
# =============================================================================


@dataclass
class MatchRow:
    gt_id: int
    pred_id: int | None
    name: str
    iou: float
    intersection: int
    union: int
    gt_count: int
    pred_count: int


def load_las_xyz_uncentered(path: Path) -> np.ndarray:
    path = _resolve_readable_cloud_path(Path(path))
    las = laspy.read(str(path))
    return np.column_stack([np.asarray(las.x, float), np.asarray(las.y, float), np.asarray(las.z, float)])


def load_manual_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    las = laspy.read(str(path))
    xyz = np.column_stack([np.asarray(las.x, float), np.asarray(las.y, float), np.asarray(las.z, float)])
    gt = np.asarray(las.classification, dtype=np.int32)
    return xyz, gt


def _read_asc_xyz(path: Path) -> np.ndarray:
    """OPTIMISED: try np.loadtxt first, fallback to line-by-line."""
    try:
        data = np.loadtxt(str(path), comments=("#", "//"), delimiter=None, usecols=(0, 1, 2), dtype=np.float64)
        if data.ndim == 2 and data.shape[1] >= 3:
            return data[:, :3]
    except (ValueError, IndexError):
        pass
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No XYZ points in ASC: {path}")
    return np.asarray(rows, dtype=np.float64)


def _indexed_asc_files(files: list[Path]) -> dict[int, Path]:
    """Shared helper to index ASC files by their numeric cloud index."""
    by_idx: dict[int, Path] = {}
    for asc in files:
        parts = asc.stem.split("_")
        idx = None
        for i, part in enumerate(parts):
            if (
                part.isdigit()
                and int(part) <= 50
                and i + 1 < len(parts)
                and len(parts[i + 1]) >= 4
                and parts[i + 1][:4].isdigit()
            ):
                idx = int(part)
                break
        if idx is None:
            continue
        prev = by_idx.get(idx)
        if prev is None or asc.stat().st_mtime >= prev.stat().st_mtime:
            by_idx[idx] = asc
    return by_idx


def export_bin_clouds_to_asc(bin_path: Path, export_dir: Path) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)

    existing = _indexed_asc_files(list(export_dir.glob("*.asc")))
    if len(existing) >= 2:
        return [existing[i] for i in sorted(existing)]

    cc_exe = CLOUDCOMPARE_EXE or find_cloudcompare_exe()
    if cc_exe is None or not cc_exe.is_file():
        raise FileNotFoundError(
            "CloudCompare is required to read multi-cloud .bin GT labels, "
            "but it was not found on this machine.\n"
            "Install options:\n"
            "  macOS:   brew install --cask cloudcompare\n"
            "  Windows: install from https://www.danielgm.net/cc/\n"
            "Or set CLOUDCOMPARE_EXE to the CloudCompare binary path,\n"
            f"or manually export each .bin cloud to ASC into:\n  {export_dir}"
        )

    for old in export_dir.glob("*.asc"):
        old.unlink()

    print(f"    Exporting CloudCompare .bin via CLI ({cc_exe}) -> {export_dir}")
    proc = subprocess.run(
        [str(cc_exe), "-SILENT", "-O", str(bin_path.resolve()),
         "-C_EXPORT_FMT", "ASC", "-PREC", "6", "-ADD_HEADER",
         "-AUTO_SAVE", "OFF", "-SAVE_CLOUDS"],
        cwd=str(export_dir), capture_output=True, text=True, check=False,
    )
    stem = bin_path.stem
    produced = list(bin_path.parent.glob(f"{stem}_*.asc")) + list(export_dir.glob("*.asc"))
    unique = {p.name: p for p in produced}
    if not unique:
        raise RuntimeError(
            "CloudCompare export produced no ASC files.\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )

    for name, src in sorted(unique.items()):
        dst = export_dir / name
        if src.resolve() != dst.resolve():
            if dst.exists():
                dst.unlink()
            src.replace(dst)

    existing = _indexed_asc_files(list(export_dir.glob("*.asc")))
    if len(existing) < 2:
        raise RuntimeError(f"Expected multiple bin clouds, found {len(existing)} in {export_dir}")
    return [existing[i] for i in sorted(existing)]


def _read_asc_xyz_and_class(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    rows: list[list[float]] = []
    classes: list[float] = []
    has_class = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = ""
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("//") or s.startswith("#"):
                header = s.lower()
                has_class = "classification" in header
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_class and len(parts) >= 4:
                    classes.append(float(parts[-1]))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No XYZ points in ASC: {path}")
    xyz = np.asarray(rows, dtype=np.float64)
    cls_arr = (
        np.asarray(classes, dtype=np.float64)
        if classes and len(classes) == len(rows) else None
    )
    return xyz, cls_arr


def load_plant_from_bin(
    bin_path: Path, *, cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cache = cache_dir or (bin_path.parent / "iou_evaluation" / f"{bin_path.stem}_asc")
    asc_files = export_bin_clouds_to_asc(bin_path, cache)

    by_idx = _indexed_asc_files(asc_files)
    asc_files = [by_idx[i] for i in sorted(by_idx)]

    all_pts: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []
    print(f"    Bin clouds used: {len(asc_files)}")
    for cloud_idx, asc in enumerate(asc_files):
        pts, cls_col = _read_asc_xyz_and_class(asc)
        n = len(pts)
        if cls_col is not None and len(np.unique(cls_col)) == 1:
            cid = int(np.unique(cls_col)[0])
        elif cls_col is not None:
            all_pts.append(pts)
            all_cls.append(cls_col.astype(np.int32))
            print(f"    mixed classes in {asc.name}: n={n:,}")
            continue
        else:
            cid = cloud_idx
            print(f"    Warning: no Classification in {asc.name}; using index {cid}")
        all_pts.append(pts)
        all_cls.append(np.full(n, cid, dtype=np.int32))
        print(f"    class {cid}: {n:,} pts from {asc.name}")

    xyz = np.vstack(all_pts)
    labels = np.concatenate(all_cls).astype(np.int32)
    print(
        f"    Loaded {len(xyz):,} points from .bin; "
        f"classes={sorted(int(i) for i in np.unique(labels))}"
    )
    return xyz, labels


def labels_from_bin_clouds(
    bin_path: Path, reference_xyz: np.ndarray, *, cache_dir: Path | None = None,
) -> np.ndarray:
    bin_xyz, bin_labels = load_plant_from_bin(bin_path, cache_dir=cache_dir)
    _, nn = cKDTree(bin_xyz).query(reference_xyz, k=1, workers=-1)
    labels = bin_labels[nn].astype(np.int32)
    print(
        f"    Labelled {len(labels):,} reference points from {len(bin_xyz):,} bin pts; "
        f"classes={sorted(int(i) for i in np.unique(labels))}"
    )
    return labels


def classify_plant_for_iou(
    xyz: np.ndarray, cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    center = xyz.mean(axis=0)
    xyz_c = xyz - center
    prep = preprocess_point_cloud(xyz_c, cfg, rgb=None)
    seg, leaf = segment_and_cluster(prep, cfg)

    n = len(prep.xyz_work)
    pred_work = np.full(n, -1, dtype=np.int32)
    pred_work[seg.stem_mask] = 0
    # Every clustered leaf point keeps the id of the cluster it is actually in.
    # Leaf points outside any cluster (DBSCAN noise, dropped fragments) stay -1
    # here and take the label of the nearest labelled point below.
    leaf_idx = np.where(seg.leaf_mask)[0]
    if leaf_idx.size and leaf.clusters and leaf.point_labels is not None:
        in_cluster = leaf.point_labels >= 0
        pred_work[leaf_idx[in_cluster]] = leaf.point_labels[in_cluster] + 1

    unlabeled = pred_work < 0
    if np.any(unlabeled) and np.any(pred_work >= 0):
        known = prep.xyz_work[~unlabeled]
        known_l = pred_work[~unlabeled]
        _, nn = cKDTree(known).query(prep.xyz_work[unlabeled], k=1, workers=-1)
        pred_work[unlabeled] = known_l[nn]
    elif np.any(unlabeled):
        pred_work[unlabeled] = 0

    return prep.xyz_work + center, pred_work


def propagate_labels(
    xyz_work: np.ndarray, pred_work: np.ndarray, xyz_full: np.ndarray,
) -> np.ndarray:
    _, nn = cKDTree(xyz_work).query(xyz_full, k=1, workers=-1)
    return pred_work[nn].astype(np.int32)


def _pair_iou(
    pred: np.ndarray, gt: np.ndarray, pred_id: int, gt_id: int,
) -> tuple[float, int, int, int, int]:
    p = pred == pred_id
    g = gt == gt_id
    inter = int(np.count_nonzero(p & g))
    union = int(np.count_nonzero(p | g))
    iou = float(inter / union) if union else 0.0
    return iou, inter, union, int(np.count_nonzero(g)), int(np.count_nonzero(p))


def compute_matched_mapping(
    pred: np.ndarray, gt: np.ndarray, *, min_match_iou: float = 0.05,
) -> tuple[list[MatchRow], dict[str, float], float]:
    rows: list[MatchRow] = []
    iou, inter, union, gt_n, pred_n = _pair_iou(pred, gt, 0, 0)
    rows.append(MatchRow(0, 0, "stem", iou, inter, union, gt_n, pred_n))

    pred_leaf_ids = sorted(int(i) for i in np.unique(pred) if i > 0)
    gt_leaf_ids = sorted(int(i) for i in np.unique(gt) if i > 0)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    if pred_leaf_ids and gt_leaf_ids:
        cost = np.zeros((len(pred_leaf_ids), len(gt_leaf_ids)), dtype=np.float64)
        for i, pid in enumerate(pred_leaf_ids):
            for j, gid in enumerate(gt_leaf_ids):
                cost[i, j] = -_pair_iou(pred, gt, pid, gid)[0]
        for i, j in zip(*linear_sum_assignment(cost)):
            pid, gid = pred_leaf_ids[i], gt_leaf_ids[j]
            iou_ij, inter, union, gt_n, pred_n = _pair_iou(pred, gt, pid, gid)
            if iou_ij < min_match_iou:
                continue
            rows.append(MatchRow(gid, pid, f"leaf_{gid}", iou_ij, inter, union, gt_n, pred_n))
            matched_gt.add(gid)
            matched_pred.add(pid)

    for gid in gt_leaf_ids:
        if gid not in matched_gt:
            n = int(np.count_nonzero(gt == gid))
            rows.append(MatchRow(gid, None, f"leaf_{gid}", 0.0, 0, n, n, 0))
    for pid in pred_leaf_ids:
        if pid not in matched_pred:
            n = int(np.count_nonzero(pred == pid))
            rows.append(MatchRow(-1, pid, f"pred_leaf_{pid}_unmatched", 0.0, 0, n, 0, n))

    rows.sort(key=lambda r: (
        0 if r.gt_id == 0 else 1 if r.gt_id > 0 else 2,
        r.gt_id if r.gt_id >= 0 else 999,
        r.pred_id if r.pred_id is not None else 999,
    ))

    def _bin_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = int(np.count_nonzero(a & b))
        union = int(np.count_nonzero(a | b))
        return float(inter / union) if union else float("nan")

    binary = {
        "stem_iou": _bin_iou(pred == 0, gt == 0),
        "leaf_iou": _bin_iou(pred > 0, gt > 0),
    }
    scored = [r.iou for r in rows if r.gt_id >= 0]
    mean_iou = float(np.mean(scored)) if scored else float("nan")
    # Strict mean also counts predicted leaves that match no GT leaf (IoU 0),
    # so splitting one leaf into several pieces is penalised.
    binary["mean_iou_strict"] = float(np.mean([r.iou for r in rows])) if rows else float("nan")
    return rows, binary, mean_iou


def print_mapping_table(
    rows: list[MatchRow], binary: dict[str, float], mean_iou: float, *, title: str,
) -> str:
    lines = [
        title, "-" * 77,
        f"{'GT':>4} {'Pred':>6}  {'Name':<26} {'IoU':>8} {'GT pts':>9} {'Pred pts':>9} {'Inter':>9}",
        "-" * 77,
    ]
    for r in rows:
        pred_s = "-" if r.pred_id is None else str(r.pred_id)
        lines.append(
            f"{r.gt_id:>4} {pred_s:>6}  {r.name:<26} {r.iou:8.4f} "
            f"{r.gt_count:>9,} {r.pred_count:>9,} {r.intersection:>9,}"
        )
    lines.append("-" * 77)
    lines.append(
        f"Binary stem IoU={binary['stem_iou']:.4f}  |  "
        f"Binary leaf IoU={binary['leaf_iou']:.4f}  |  Mean IoU={mean_iou:.4f}"
    )
    lines.append(
        f"Strict mean IoU (unmatched predicted leaves count as 0)="
        f"{binary['mean_iou_strict']:.4f}"
    )
    text = "\n".join(lines)
    print(text)
    return text


def write_iou_report(
    path: Path, rows: list[MatchRow], binary: dict[str, float], mean_iou: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_id", "pred_id", "name", "iou", "intersection", "union", "gt_count", "pred_count"])
        for r in rows:
            w.writerow([
                r.gt_id, "" if r.pred_id is None else r.pred_id,
                r.name, f"{r.iou:.6f}", r.intersection, r.union, r.gt_count, r.pred_count,
            ])
        w.writerow([])
        w.writerow(["binary_stem_iou", f"{binary['stem_iou']:.6f}"])
        w.writerow(["binary_leaf_iou", f"{binary['leaf_iou']:.6f}"])
        w.writerow(["mean_iou", f"{mean_iou:.6f}"])
        w.writerow(["mean_iou_strict", f"{binary['mean_iou_strict']:.6f}"])


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """
    Vectorised label → RGB.

    - Class 0..50: stem / leaf palette
    - Class >= 100: unmatched Python predictions (encoded as 100 + pred_id)
      each get their own color from the 50-leaf palette (no shared grey)
    - Negative / other: unmatched grey
    """
    lab = labels.astype(int, copy=False)
    rgb = np.broadcast_to(
        np.asarray(UNMATCHED_COLOR, dtype=np.float64), (len(lab), 3)
    ).copy()

    known = (lab >= 0) & (lab <= max(CLASS_COLORS))
    if np.any(known):
        rgb[known] = _COLOR_LUT[lab[known]]

    # Unmatched predictions: 100 + pred_id → distinct leaf colors
    extra = lab >= 100
    if np.any(extra):
        rgb[extra] = _LEAF_PALETTE[(lab[extra] - 100) % len(_LEAF_PALETTE)]

    return rgb


def _subsample(xyz: np.ndarray, *arrays: np.ndarray, max_pts: int = 100_000):
    if len(xyz) <= max_pts:
        return (xyz, *arrays)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), max_pts, replace=False)
    return (xyz[idx], *(a[idx] for a in arrays))


def _class_label_anchors(
    pts: np.ndarray, labels: np.ndarray, *, name_of,
) -> tuple[np.ndarray, list[str]]:
    stem_mask = labels == 0
    stem_xy = pts[stem_mask, :2].mean(axis=0) if np.any(stem_mask) else pts[:, :2].mean(axis=0)
    anchors: list[np.ndarray] = []
    texts: list[str] = []
    for cid in sorted(int(i) for i in np.unique(labels) if i >= 0):
        mask = labels == cid
        if not np.any(mask):
            continue
        seg = pts[mask]
        radial = np.linalg.norm(seg[:, :2] - stem_xy, axis=1)
        tip_score = 0.65 * radial + 0.35 * (seg[:, 2] - seg[:, 2].min())
        tip = seg[int(np.argmax(tip_score))].copy()
        rvec = tip[:2] - stem_xy
        rn = float(np.linalg.norm(rvec))
        if rn > 1e-9:
            tip[:2] += (rvec / rn) * 0.018
        tip[2] += 0.012
        anchors.append(tip)
        texts.append(name_of(cid))
    if not anchors:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(anchors, dtype=np.float64), texts


def _remap_pred_to_gt_ids(pred: np.ndarray, match_rows: list[MatchRow]) -> np.ndarray:
    remapped = np.full_like(pred, -1)
    remapped[pred == 0] = 0
    for r in match_rows:
        if r.gt_id > 0 and r.pred_id is not None:
            remapped[pred == r.pred_id] = r.gt_id
    for pid in np.unique(pred):
        pid = int(pid)
        if pid <= 0:
            continue
        if not any(r.pred_id == pid and r.gt_id > 0 for r in match_rows):
            remapped[pred == pid] = 100 + pid
    return remapped


def _classification_clouds(
    xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray, match_rows: list[MatchRow],
) -> tuple[np.ndarray, pv.PolyData, pv.PolyData]:
    origin = xyz.mean(axis=0)
    remapped = _remap_pred_to_gt_ids(pred, match_rows)
    xyz_v, gt_v, rem_v = _subsample(xyz, gt, remapped, max_pts=100_000)
    pts = xyz_v - origin
    cloud_g = pv.PolyData(pts)
    cloud_g["rgb"] = (labels_to_rgb(gt_v) * 255).astype(np.uint8)
    cloud_p = pv.PolyData(pts)
    cloud_p["rgb"] = (labels_to_rgb(rem_v) * 255).astype(np.uint8)
    return pts, cloud_g, cloud_p


def _add_pyvista_class_labels(
    plotter: pv.Plotter, anchors: np.ndarray, texts: list[str], font_size: int = 13,
) -> None:
    """Shared helper for adding class labels to a PyVista subplot."""
    if not texts:
        return
    plotter.add_point_labels(
        anchors, texts, font_size=font_size, point_size=0, show_points=False,
        shape="rounded_rect", shape_opacity=0.85, margin=3,
        bold=True, text_color="white", shape_color="black", always_visible=True,
    )


def visualize_combined_viewer(
    xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray,
    match_rows: list[MatchRow], table_text: str,
    result: PipelineResult, prep: PreprocessResult,
    leaf_display_ids: list[int] | None = None,
) -> None:
    pts, cloud_g, cloud_p = _classification_clouds(xyz, pred, gt, match_rows)

    def _gt_name(cid: int) -> str:
        return "stem" if cid == 0 else f"leaf_{cid}"

    def _pred_name(cid: int) -> str:
        if cid == 0:
            return "stem"
        if cid >= 100:
            return f"pred_{cid - 100}"
        return f"leaf_{cid}"

    _, gt_v, rem_v = _subsample(xyz, gt, _remap_pred_to_gt_ids(pred, match_rows), max_pts=100_000)
    gt_anchors, gt_texts = _class_label_anchors(pts, gt_v, name_of=_gt_name)
    pred_anchors, pred_texts = _class_label_anchors(pts, rem_v, name_of=_pred_name)

    plotter = pv.Plotter(shape=(1, 3), window_size=(2200, 900), title="Plant classification + structure")
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("CloudCompare .bin GT", font_size=13, color="black")
    plotter.add_mesh(cloud_g, scalars="rgb", rgb=True, point_size=3.0)
    _add_pyvista_class_labels(plotter, gt_anchors, gt_texts)

    plotter.subplot(0, 1)
    plotter.add_text("Python classification", font_size=13, color="black")
    plotter.add_mesh(cloud_p, scalars="rgb", rgb=True, point_size=3.0)
    _add_pyvista_class_labels(plotter, pred_anchors, pred_texts)
    short = table_text.splitlines()[:16]
    plotter.add_text("\n".join(short), position=IOU_TABLE_POSITION, font_size=9, color="black")

    plotter.subplot(0, 2)
    plotter.add_text("Plant structure", font_size=13, color="black")
    _populate_structure_scene(plotter, result, prep, leaf_display_ids=leaf_display_ids)

    plotter.link_views()
    print("\nClose the PyVista window to finish.\n")
    plotter.show()


def visualize_gt_vs_prediction(
    xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray,
    match_rows: list[MatchRow], table_text: str,
) -> None:
    pts, cloud_g, cloud_p = _classification_clouds(xyz, pred, gt, match_rows)
    remapped = _remap_pred_to_gt_ids(pred, match_rows)

    def _gt_name(cid: int) -> str:
        return "stem" if cid == 0 else f"leaf_{cid}"

    def _pred_name(cid: int) -> str:
        if cid == 0:
            return "stem"
        if cid >= 100:
            return f"pred_{cid - 100}"
        return f"leaf_{cid}"

    _, gt_v, rem_v = _subsample(xyz, gt, remapped, max_pts=100_000)
    gt_anchors, gt_texts = _class_label_anchors(pts, gt_v, name_of=_gt_name)
    pred_anchors, pred_texts = _class_label_anchors(pts, rem_v, name_of=_pred_name)

    plotter = pv.Plotter(shape=(1, 2), window_size=(1700, 900), title="CloudCompare .bin GT | Python prediction")
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("CloudCompare .bin GT", font_size=14, color="black")
    plotter.add_mesh(cloud_g, scalars="rgb", rgb=True, point_size=3.0)
    _add_pyvista_class_labels(plotter, gt_anchors, gt_texts, font_size=14)

    plotter.subplot(0, 1)
    plotter.add_text("Python classification", font_size=14, color="black")
    plotter.add_mesh(cloud_p, scalars="rgb", rgb=True, point_size=3.0)
    _add_pyvista_class_labels(plotter, pred_anchors, pred_texts, font_size=14)

    plotter.subplot(0, 1)
    short = table_text.splitlines()[:16]
    plotter.add_text("\n".join(short), position=IOU_TABLE_POSITION, font_size=9, color="black")
    plotter.link_views()
    print("\nClose the PyVista window to finish.\n")
    plotter.show()


def run_iou_evaluation(
    bin_path: Path, cfg: PipelineConfig, output_dir: Path, *,
    visualize: bool = True, min_match_iou: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[MatchRow], str]:
    print(f"[1] Plant .bin (geometry + GT): {bin_path}")
    xyz, gt = load_plant_from_bin(bin_path)
    print(f"    Points: {len(xyz):,}")
    print(
        f"    GT class counts: "
        + ", ".join(
            f"{c}:{int(np.count_nonzero(gt == c)):,}"
            for c in sorted(int(i) for i in np.unique(gt) if i >= 0)
        )
    )

    print("[2] Geometric classification (stem + leaf clusters) -- no GT labels used")
    xyz_work, pred_work = classify_plant_for_iou(xyz, cfg)
    pred = propagate_labels(xyz_work, pred_work, xyz)
    print(
        f"    Work points: {len(xyz_work):,} | "
        f"Pred classes: {sorted(int(i) for i in np.unique(pred))}"
    )

    print("[3] IoU: Python prediction  vs  CloudCompare .bin GT")
    rows, binary, mean_iou = compute_matched_mapping(pred, gt, min_match_iou=min_match_iou)
    table = print_mapping_table(rows, binary, mean_iou, title="Python prediction  vs  CloudCompare .bin GT")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_iou_report(output_dir / "iou_python_vs_bin.csv", rows, binary, mean_iou)
    print(f"[4] Wrote IoU CSV -> {output_dir / 'iou_python_vs_bin.csv'}")

    if visualize:
        print("[5] Viewer: CloudCompare .bin GT | Python prediction")
        visualize_gt_vs_prediction(xyz, pred, gt, rows, table)

    return xyz, pred, gt, rows, table


def _match_structure_leaves_to_classification(
    result: PipelineResult, gt: np.ndarray, xyz_full: np.ndarray,
    match_rows: list[MatchRow], pred: np.ndarray | None = None,
    cfg: PipelineConfig | None = None,
) -> tuple[PipelineResult, list[int]]:
    n_leaves = len(result.leaf_skeleton.paths)
    if n_leaves == 0:
        return result, []

    pred_to_gt: dict[int, int] = {}
    for row in match_rows:
        if row.pred_id is not None and row.gt_id >= 0:
            pred_to_gt[row.pred_id] = row.gt_id

    origin = xyz_full.mean(axis=0)
    ref_pts = xyz_full - origin
    ref_tree = cKDTree(ref_pts)
    ref_labels = pred if pred is not None else gt

    struct_to_gt: list[int] = []
    for cluster in result.leaf_clusters.clusters:
        if len(cluster) == 0:
            struct_to_gt.append(-1)
            continue
        _, nn = ref_tree.query(cluster, k=1, workers=-1)
        labels_near = ref_labels[nn]
        leaf_labels = labels_near[labels_near > 0]
        if len(leaf_labels) == 0:
            struct_to_gt.append(-1)
            continue
        vals, counts = np.unique(leaf_labels, return_counts=True)
        dominant_pred = int(vals[np.argmax(counts)])
        gt_id = pred_to_gt.get(dominant_pred, dominant_pred)
        struct_to_gt.append(gt_id)

    skel_gt_ids: list[int] = []
    for ci in result.leaf_skeleton.cluster_indices:
        if ci < len(struct_to_gt):
            skel_gt_ids.append(struct_to_gt[ci])
        else:
            skel_gt_ids.append(-1)

    # Skeletons that match no classified leaf each get a fresh number; before,
    # they all shared id -1 and only the largest one survived the de-duplication.
    next_id = max([g for g in skel_gt_ids if g > 0] + [int(gt.max()), 0]) + 1
    for i, gid in enumerate(skel_gt_ids):
        if gid <= 0:
            skel_gt_ids[i] = next_id
            next_id += 1

    seen: dict[int, int] = {}
    for i, gid in enumerate(skel_gt_ids):
        ci = result.leaf_skeleton.cluster_indices[i]
        sz = len(result.leaf_clusters.clusters[ci]) if ci < len(result.leaf_clusters.clusters) else 0
        if gid not in seen or sz > len(result.leaf_clusters.clusters[
                result.leaf_skeleton.cluster_indices[seen[gid]]]):
            seen[gid] = i
    keep = sorted(seen.values())
    skel_gt_ids = [skel_gt_ids[i] for i in keep]

    order = sorted(range(len(skel_gt_ids)), key=lambda i: skel_gt_ids[i])
    final_indices = [keep[i] for i in order]
    gt_ids_sorted = [skel_gt_ids[i] for i in order]

    ls = result.leaf_skeleton
    new_paths = [ls.paths[i] for i in final_indices]
    new_junctions = ls.junctions[final_indices] if len(ls.junctions) > 0 else ls.junctions
    new_jsi = ls.junction_stem_indices[final_indices] if len(ls.junction_stem_indices) > 0 else ls.junction_stem_indices
    new_tips = ls.tips[final_indices] if len(ls.tips) > 0 else ls.tips
    new_ci = [ls.cluster_indices[i] for i in final_indices]

    new_leaf_sk = LeafSkeletonResult(
        junctions=new_junctions, junction_stem_indices=new_jsi,
        paths=new_paths, tips=new_tips, cluster_indices=new_ci,
    )

    new_sm_leaves = [result.smoothed_leaf_paths[i] for i in final_indices]

    new_clusters = [result.leaf_clusters.clusters[ci] for ci in new_ci
                    if ci < len(result.leaf_clusters.clusters)]

    new_measurements = compute_measurements(
        result.smoothed_stem, new_leaf_sk, result.skeleton_graph,
        new_sm_leaves, plant_xyz=result.preprocess.xyz_clean, cfg=cfg,
    )

    new_result = PipelineResult(
        preprocess=result.preprocess, segmentation=result.segmentation,
        plant_graph=result.plant_graph, stem_skeleton=result.stem_skeleton,
        leaf_clusters=LeafClusterResult(clusters=new_clusters, labels=result.leaf_clusters.labels),
        leaf_skeleton=new_leaf_sk, smoothed_stem=result.smoothed_stem,
        smoothed_leaf_paths=new_sm_leaves, skeleton_graph=result.skeleton_graph,
        measurements=new_measurements,
    )

    m = new_measurements
    print(f"\n    === Corrected per-leaf measurements (classification numbering) ===")
    print(f"    Plant bottom (reference): 0.00 cm")
    print(f"    Plant height:   {m.plant_height_m*100:.2f} cm")
    print(f"    Insertion height = junction Z - plant bottom Z")
    for i, (gid, h, a, L) in enumerate(zip(
        gt_ids_sorted, m.insertion_heights_m, m.leaf_angles_deg, m.leaf_lengths_m,
    )):
        print(
            f"    Leaf {gid}: insertion_height={h*100:.2f} cm, "
            f"stem_leaf_angle={a:.1f} deg, geodesic_length={L:.4f} m"
        )

    return new_result, gt_ids_sorted


def run_classification_and_structure(
    bin_path: Path, cfg: PipelineConfig, output_dir: Path, *,
    visualize: bool = True, min_match_iou: float = 0.05,
    show_o3d_steps: bool = False, run_structure: bool = True,
) -> PipelineResult | None:
    print("=" * 72)
    print("PART A -- Plant classification + IoU vs CloudCompare GT")
    print("=" * 72)
    xyz, pred, gt, rows, table = run_iou_evaluation(
        bin_path, cfg, output_dir, visualize=False, min_match_iou=min_match_iou,
    )

    if not run_structure:
        if visualize:
            visualize_gt_vs_prediction(xyz, pred, gt, rows, table)
        return None

    print()
    print("=" * 72)
    print("PART B -- Plant structure")
    print("=" * 72)
    structure_dir = output_dir / "skeleton_output"
    cfg.output_dir = structure_dir
    result = run_structure_from_xyz(xyz, cfg, show_o3d_steps=show_o3d_steps, show_pyvista=False)

    print("[Structure <-> Classification] Matching leaf numbering ...")
    result, gt_leaf_ids = _match_structure_leaves_to_classification(
        result, gt, xyz, rows, pred=pred, cfg=cfg,
    )

    if visualize:
        print("[Combined viewer] GT | classification | structure")
        visualize_combined_viewer(
            xyz, pred, gt, rows, table, result, result.preprocess,
            leaf_display_ids=gt_leaf_ids,
        )

    return result


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Combined plant classification (IoU vs CloudCompare .bin) "
            "followed by plant structure extraction."
        )
    )
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--voxel_size", type=float, default=0.004)
    p.add_argument("--sor_neighbors", type=int, default=20)
    p.add_argument("--sor_std_ratio", type=float, default=2.0)
    p.add_argument("--ror_radius", type=float, default=0.012)
    p.add_argument("--ror_min_neighbors", type=int, default=8)
    p.add_argument("--graph_k", type=int, default=12)
    p.add_argument("--leaf_cluster_eps", type=float, default=0.028)
    p.add_argument("--leaf_cluster_min_points", type=int, default=25)
    p.add_argument("--dbscan_eps_nn_factor", type=float, default=2.5)
    p.add_argument("--dbscan_eps_voxel_factor", type=float, default=4.0)
    p.add_argument("--leaf_graph_k", type=int, default=18)
    p.add_argument("--leaf_graph_max_edge", type=float, default=0.075)
    p.add_argument("--leaf_planarity_min", type=float, default=0.18)
    p.add_argument("--leaf_split_min_cluster_size", type=int, default=3000)
    p.add_argument("--leaf_split_min_angle_deg", type=float, default=40.0)
    p.add_argument("--no_split_nearby_leaves", action="store_true")
    p.add_argument("--laplacian_iterations", type=int, default=35)
    p.add_argument("--feature_k", type=int, default=48)
    p.add_argument("--stem_score_threshold", type=float, default=None)
    p.add_argument("--min_leaf_component_size", type=int, default=50)
    p.add_argument("--no_region_growing", action="store_true")
    p.add_argument("--no_adaptive_dbscan", action="store_true")
    p.add_argument("--show_o3d_steps", action="store_true")
    p.add_argument("--skeleton_only", action="store_true")
    p.add_argument("--classification_only", action="store_true")
    p.add_argument("--no_viewer", action="store_true")
    p.add_argument("--min_match_iou", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.bin = resolve_bin_path(args.bin)
    if not args.bin.is_file():
        raise FileNotFoundError(f"Plant .bin not found: {args.bin}")

    cfg_kwargs: dict[str, Any] = dict(
        sor_neighbors=args.sor_neighbors, sor_std_ratio=args.sor_std_ratio,
        ror_radius=args.ror_radius, ror_min_neighbors=args.ror_min_neighbors,
        voxel_size=args.voxel_size, graph_k=args.graph_k,
        leaf_cluster_eps=args.leaf_cluster_eps,
        leaf_cluster_min_points=args.leaf_cluster_min_points,
        dbscan_eps_voxel_factor=args.dbscan_eps_voxel_factor,
        dbscan_eps_nn_factor=args.dbscan_eps_nn_factor,
        leaf_graph_k=args.leaf_graph_k, leaf_graph_max_edge=args.leaf_graph_max_edge,
        leaf_planarity_min=args.leaf_planarity_min,
        leaf_split_min_angle_deg=args.leaf_split_min_angle_deg,
        leaf_split_min_cluster_size=args.leaf_split_min_cluster_size,
        split_nearby_leaves=not args.no_split_nearby_leaves,
        laplacian_iterations=args.laplacian_iterations,
        feature_k=args.feature_k,
        min_leaf_component_size=args.min_leaf_component_size,
        use_region_growing=not args.no_region_growing,
        use_adaptive_dbscan=not args.no_adaptive_dbscan,
    )
    if args.stem_score_threshold is not None:
        cfg_kwargs["stem_score_threshold"] = args.stem_score_threshold
    cfg = PipelineConfig(**cfg_kwargs)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    if args.skeleton_only:
        if args.input is not None and args.input.is_file():
            run_pipeline(args.input, cfg, show_o3d_steps=args.show_o3d_steps, show_pyvista=not args.no_viewer)
            return
        xyz, _ = load_plant_from_bin(args.bin)
        run_structure_from_xyz(xyz, cfg, show_o3d_steps=args.show_o3d_steps, show_pyvista=not args.no_viewer)
        return

    out_dir = args.output_dir or (args.bin.parent / "iou_evaluation" / args.bin.stem)
    run_classification_and_structure(
        args.bin, cfg, out_dir,
        visualize=not args.no_viewer, min_match_iou=args.min_match_iou,
        show_o3d_steps=args.show_o3d_steps,
        run_structure=not args.classification_only,
    )


if __name__ == "__main__":
    main()
