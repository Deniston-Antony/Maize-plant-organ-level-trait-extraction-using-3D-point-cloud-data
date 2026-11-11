"""Plant / soil classification (CSF + improved DTM + optional sideband).

Uses the GT-tuned parameters from DTM_sideband (high precision on small plants).

Edit DATASET / USE_SIDEBAND below, then run:
    .venv\\Scripts\\python.exe soil_segmentation/classify_plants.py

USE_SIDEBAND:
    True  -> small plants (force side bands to soil, clean middle corridor)
    False -> large plants (no sideband crop; full bed kept)
"""

from __future__ import annotations

from pathlib import Path

import CSF
import laspy
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

# ============================================================
# USER SETTINGS — only these need changing
# ============================================================

# Dataset folder name under data/  (all files share this stem)
#   data/<DATASET>/<DATASET>.las
# DATASET = "230629_pc_kinematic_calibration"
DATASET = "230601pc_kinematic_calibration"

# False for this experiment: compare current tuned parameters against the
# historical baseline without giving either method a sideband crop.
USE_SIDEBAND = False

# Where to write PLYs (None = data/<DATASET>/output/soil_segmentation/)
OUTPUT_DIR = None

SAVE_CLASSIFIED = True
SAVE_PLANTS_ONLY = True
SAVE_GROUND_ONLY = True

INPUT_LAS = rf"data/{DATASET}/{DATASET}.las"
DEFAULT_OUTPUT_DIR = rf"data/{DATASET}/output/soil_segmentation"

# ============================================================
# BEST PARAMETERS (from GT tune on small plants / DTM_sideband)
# ============================================================

CSF_MAX_POINTS = 9_000_000
CHUNK_SIZE = 500_000

CSF_CLOTH_RES = 0.02
CSF_RIGIDNESS = 1
CSF_TIME_STEP = 0.50
CSF_ITERATIONS = 1000
CSF_THRESHOLD = 0.012
CSF_SLOPE_SMOOTH = False

DTM_CELL = 0.03
DTM_FILL_MAX_DIST = 0.12
DTM_FILL_K_NEIGHBORS = 12

HAG_SOIL_MAX = 0.020
HAG_VEG_MIN = 0.020

PLANT_MIN_NEIGHBORHOOD = 8

# Sideband (only used when USE_SIDEBAND=True)
X_TRIM_FRAC = 0.38
HARD_SIDE_CROP = True

COLOR_SOIL = (140, 140, 140)
COLOR_MIXED = (255, 140, 0)
COLOR_PLANT = (13, 217, 26)
COLOR_INVALID = (242, 217, 26)

# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# Final_code/ (parent of soil_segmentation/) — where data/ and .venv live
PROJECT_ROOT = SCRIPT_DIR.parent

_PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        # Prefer project root (data/), then this folder
        for base in (PROJECT_ROOT, SCRIPT_DIR):
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return (PROJECT_ROOT / p).resolve()
    return p.resolve()


def load_las_subsample(las_path: Path, max_points: int):
    with laspy.open(str(las_path)) as reader:
        total = int(reader.header.point_count)
        stride = max(1, (total + max_points - 1) // max_points)
        print(f"Total LAS points     : {total:,}")
        print(f"CSF subsample stride : {stride}  (-> ~{total // stride:,} pts)")
        xs, ys, zs, gi = [], [], [], 0
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            n = len(x)
            keep = (np.arange(gi, gi + n) % stride) == 0
            xs.append(x[keep]); ys.append(y[keep]); zs.append(z[keep])
            gi += n
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), total


def csf_classify(x, y, z):
    print("\nRunning CSF on the subsample ...")
    csf = CSF.CSF()
    csf.params.bSloopSmooth = CSF_SLOPE_SMOOTH
    csf.params.cloth_resolution = CSF_CLOTH_RES
    csf.params.rigidness = CSF_RIGIDNESS
    csf.params.time_step = CSF_TIME_STEP
    csf.params.class_threshold = CSF_THRESHOLD
    if hasattr(csf.params, "iterations"):
        csf.params.iterations = CSF_ITERATIONS
    else:
        csf.params.interations = CSF_ITERATIONS
    csf.setPointCloud(np.column_stack((x, y, z)).astype(np.float64))
    ground_idx, non_ground_idx = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)
    ground_mask = np.zeros(len(x), dtype=bool)
    ground_mask[np.asarray(ground_idx, dtype=np.int64)] = True
    print(f"  CSF ground     : {int(ground_mask.sum()):,}")
    print(f"  CSF non-ground : {int((~ground_mask).sum()):,}")
    return ground_mask


class DTM:
    def __init__(self, x0, y0, nx, ny, cell, z_grid, observed, filled):
        self.x0, self.y0 = x0, y0
        self.nx, self.ny, self.cell = nx, ny, cell
        self.z_grid = z_grid
        self.observed = observed
        self.filled = filled

    def height(self, x, y):
        ix = np.clip(np.floor((x - self.x0) / self.cell).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.floor((y - self.y0) / self.cell).astype(np.int64), 0, self.ny - 1)
        return self.z_grid[ix, iy]


def build_dtm_improved(xg, yg, zg, xmin, ymin, xmax, ymax, cell, fill_max_dist, fill_k):
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    ny = int(np.ceil((ymax - ymin) / cell)) + 1
    ix = np.clip(np.floor((xg - xmin) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((yg - ymin) / cell).astype(np.int64), 0, ny - 1)
    flat = ix * ny + iy
    order = np.argsort(flat)
    f_sorted, z_sorted = flat[order], zg[order]
    splits = np.flatnonzero(np.diff(f_sorted)) + 1
    starts = np.insert(splits, 0, 0)
    ends = np.append(splits, len(z_sorted))
    z_grid = np.full((nx, ny), np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        c = int(f_sorted[s])
        z_grid[c // ny, c % ny] = float(np.median(z_sorted[s:e]))

    observed = np.isfinite(z_grid)
    filled = np.zeros((nx, ny), dtype=bool)
    n_observed = int(observed.sum())
    n_gap = int((~observed).sum())
    if n_observed == 0:
        raise RuntimeError("No CSF ground points in any DTM cell.")
    if n_gap == 0:
        return DTM(xmin, ymin, nx, ny, cell, z_grid, observed, filled)

    obs_ii, obs_jj = np.where(observed)
    obs_z = z_grid[observed]
    tree = cKDTree(np.column_stack((obs_ii.astype(np.float64), obs_jj.astype(np.float64))))
    gap_ii, gap_jj = np.where(~observed)
    k = min(fill_k, len(obs_z))
    dists_ij, nn_idx = tree.query(
        np.column_stack((gap_ii.astype(np.float64), gap_jj.astype(np.float64))), k=k,
    )
    if k == 1:
        dists_ij = dists_ij.reshape(-1, 1)
        nn_idx = nn_idx.reshape(-1, 1)
    within = (dists_ij * cell) <= fill_max_dist
    z_fill = np.nanmin(np.where(within, obs_z[nn_idx], np.nan), axis=1)
    fill_ok = np.isfinite(z_fill)
    z_grid[gap_ii[fill_ok], gap_jj[fill_ok]] = z_fill[fill_ok]
    filled[gap_ii[fill_ok], gap_jj[fill_ok]] = True
    print(f"  DTM observed cells : {n_observed:,}")
    print(f"  DTM filled cells   : {int(fill_ok.sum()):,}")
    print(f"  DTM invalid cells  : {n_gap - int(fill_ok.sum()):,}")
    return DTM(xmin, ymin, nx, ny, cell, z_grid, observed, filled)


def hag_masks(hag: np.ndarray):
    finite = np.isfinite(hag)
    soil = finite & (hag <= HAG_SOIL_MAX)
    mixed = finite & (hag > HAG_SOIL_MAX) & (hag < HAG_VEG_MIN)
    plant = finite & (hag >= HAG_VEG_MIN)
    invalid = ~finite
    ground = finite & (np.abs(hag) <= CSF_THRESHOLD)
    return soil, mixed, plant, invalid, ground


def xy_to_ij(x, y, dtm: DTM):
    ix = np.clip(np.floor((x - dtm.x0) / dtm.cell).astype(np.int64), 0, dtm.nx - 1)
    iy = np.clip(np.floor((y - dtm.y0) / dtm.cell).astype(np.int64), 0, dtm.ny - 1)
    return ix, iy


def count_plant_grid(dtm: DTM, las_path: Path, total_points: int) -> np.ndarray:
    grid = np.zeros((dtm.nx, dtm.ny), dtype=np.int32)
    processed = 0
    print("\nPass 1 — plant density grid ...")
    with laspy.open(str(las_path)) as reader:
        for chunk in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            _, _, plant, _, _ = hag_masks(z - dtm.height(x, y))
            if plant.any():
                ix, iy = xy_to_ij(x[plant], y[plant], dtm)
                np.add.at(grid, (ix, iy), 1)
            processed += len(x)
            print(f"  {processed:,} / {total_points:,}", end="\r")
    print()
    return grid


def build_plant_cluster_mask(plant_grid: np.ndarray, min_neighborhood: int) -> np.ndarray:
    if min_neighborhood <= 1:
        return np.ones_like(plant_grid, dtype=bool)
    neigh = convolve(plant_grid, np.ones((3, 3), dtype=np.int32), mode="constant")
    return neigh >= min_neighborhood


def estimate_bed_width_frame(x, y, trim_frac: float):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    step = max(1, n // 200_000)
    xs, ys = x[::step], y[::step]
    ox, oy = float(xs.mean()), float(ys.mean())
    xy = np.column_stack([xs - ox, ys - oy])
    cov = (xy.T @ xy) / max(len(xy), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    across = eigvecs[:, 0]
    along = eigvecs[:, 1]
    angle_deg = float(np.degrees(np.arctan2(along[1], along[0])))
    w = (xs - ox) * across[0] + (ys - oy) * across[1]
    w_min, w_max = float(np.percentile(w, 0.5)), float(np.percentile(w, 99.5))
    width = w_max - w_min
    w_lo = w_min + trim_frac * width
    w_hi = w_max - trim_frac * width
    return np.array([ox, oy]), across, w_lo, w_hi, angle_deg, width


def side_band_mask_bed(x, y, origin, across, w_lo, w_hi) -> np.ndarray:
    w = (x - origin[0]) * across[0] + (y - origin[1]) * across[1]
    return (w < w_lo) | (w > w_hi)


class PlyWriter:
    HEADER_COUNT_DIGITS = 12

    def __init__(self, path):
        self.path = Path(path)
        self.count = 0
        header_bytes = self._make_header(0)
        self._count_offset = header_bytes.find(b"element vertex ") + len(b"element vertex ")
        self.file = open(self.path, "wb")
        self.file.write(header_bytes)

    def _make_header(self, count):
        count_str = str(count).zfill(self.HEADER_COUNT_DIGITS)
        lines = [
            "ply", "format binary_little_endian 1.0",
            f"element vertex {count_str}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue",
            "end_header",
        ]
        return ("\n".join(lines) + "\n").encode("ascii")

    def write(self, x, y, z, r, g, b):
        n = len(x)
        if n == 0:
            return
        rows = np.empty(n, dtype=_PLY_DTYPE)
        rows["x"] = x.astype(np.float32, copy=False)
        rows["y"] = y.astype(np.float32, copy=False)
        rows["z"] = z.astype(np.float32, copy=False)
        rows["r"], rows["g"], rows["b"] = r, g, b
        self.file.write(rows.tobytes())
        self.count += n

    def close(self):
        self.file.seek(self._count_offset)
        self.file.write(str(self.count).zfill(self.HEADER_COUNT_DIGITS).encode("ascii"))
        self.file.close()


def main():
    las_path = resolve_path(INPUT_LAS)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS not found: {las_path}")

    out_dir = (
        Path(OUTPUT_DIR).resolve()
        if OUTPUT_DIR
        else (PROJECT_ROOT / DEFAULT_OUTPUT_DIR).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = las_path.stem
    tag = "sideband" if USE_SIDEBAND else "nosideband"
    out_classified = out_dir / f"{stem}_classified_all_{tag}.ply"
    out_plants = out_dir / f"{stem}_plants_only_{tag}.ply"
    out_ground = out_dir / f"{stem}_ground_only_{tag}.ply"

    print(f"INPUT_LAS     : {las_path}")
    print(f"USE_SIDEBAND  : {USE_SIDEBAND}")
    print(f"OUTPUT_DIR    : {out_dir}")

    with laspy.open(str(las_path)) as reader:
        h = reader.header
        full_xmin, full_ymin = float(h.x_min), float(h.y_min)
        full_xmax, full_ymax = float(h.x_max), float(h.y_max)
        full_zmin = float(h.z_min)
        total_points = int(h.point_count)
    print(f"LAS points    : {total_points:,}")

    x_sub, y_sub, z_sub, _ = load_las_subsample(las_path, CSF_MAX_POINTS)

    origin = across = mid_lo = mid_hi = None
    if USE_SIDEBAND:
        origin, across, mid_lo, mid_hi, bed_angle, bed_width = estimate_bed_width_frame(
            x_sub, y_sub, X_TRIM_FRAC,
        )
        print(f"\nSideband ON (PCA bed width):")
        print(f"  angle ~ {bed_angle:.1f} deg, width ~ {bed_width:.3f} m")
        print(f"  trim each side = {X_TRIM_FRAC*100:.0f}%")
        print(f"  HARD_SIDE_CROP = {HARD_SIDE_CROP}")
    else:
        print("\nSideband OFF — keeping full bed width")

    ground_mask_sub = csf_classify(x_sub, y_sub, z_sub)
    print("\nBuilding improved DTM ...")
    dtm = build_dtm_improved(
        xg=x_sub[ground_mask_sub], yg=y_sub[ground_mask_sub], zg=z_sub[ground_mask_sub],
        xmin=full_xmin, ymin=full_ymin, xmax=full_xmax, ymax=full_ymax,
        cell=DTM_CELL, fill_max_dist=DTM_FILL_MAX_DIST, fill_k=DTM_FILL_K_NEIGHBORS,
    )
    print(f"DTM grid: {dtm.nx} x {dtm.ny} @ {dtm.cell} m")
    del x_sub, y_sub, z_sub, ground_mask_sub

    plant_grid = count_plant_grid(dtm, las_path, total_points)
    cluster_ok = build_plant_cluster_mask(plant_grid, PLANT_MIN_NEIGHBORHOOD)
    print(f"  Plant cluster cells: {int(cluster_ok.sum()):,}")

    x_off, y_off, z_off = full_xmin, full_ymin, full_zmin
    writers = {}
    if SAVE_CLASSIFIED:
        writers["classified"] = PlyWriter(out_classified)
    if SAVE_PLANTS_ONLY:
        writers["plants"] = PlyWriter(out_plants)
    if SAVE_GROUND_ONLY:
        writers["ground"] = PlyWriter(out_ground)

    print("\nPass 2 — classify + write PLYs ...")
    soil_count = mixed_count = plant_count = invalid_count = ground_count = 0
    demoted_cluster = demoted_side = 0
    processed = 0

    with laspy.open(str(las_path)) as reader:
        for chunk in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)

            hag = z - dtm.height(x, y)
            soil_mask, mixed_mask, plant_raw, invalid_mask, ground_mask = hag_masks(hag)

            if plant_raw.any():
                ix, iy = xy_to_ij(x[plant_raw], y[plant_raw], dtm)
                keep = cluster_ok[ix, iy]
                plant_mask = plant_raw.copy()
                plant_mask[plant_raw] = keep
                demoted = plant_raw & ~plant_mask
                demoted_cluster += int(demoted.sum())
                mixed_mask = mixed_mask | demoted
            else:
                plant_mask = plant_raw

            if USE_SIDEBAND and X_TRIM_FRAC > 0:
                on_side = side_band_mask_bed(x, y, origin, across, mid_lo, mid_hi)
                kill = on_side if HARD_SIDE_CROP else (plant_mask & on_side)
                if kill.any():
                    demoted_side += int(kill.sum())
                    plant_mask = plant_mask & ~kill
                    mixed_mask = mixed_mask & ~kill
                    invalid_mask = invalid_mask & ~kill
                    soil_mask = (soil_mask & ~kill) | kill

            soil_count += int(soil_mask.sum())
            mixed_count += int(mixed_mask.sum())
            plant_count += int(plant_mask.sum())
            invalid_count += int(invalid_mask.sum())
            ground_count += int(ground_mask.sum())

            xs, ys, zs = x - x_off, y - y_off, z - z_off

            if "classified" in writers:
                n = len(x)
                r = np.empty(n, dtype=np.uint8)
                g = np.empty(n, dtype=np.uint8)
                b = np.empty(n, dtype=np.uint8)
                r[soil_mask], g[soil_mask], b[soil_mask] = COLOR_SOIL
                r[mixed_mask], g[mixed_mask], b[mixed_mask] = COLOR_MIXED
                r[plant_mask], g[plant_mask], b[plant_mask] = COLOR_PLANT
                r[invalid_mask], g[invalid_mask], b[invalid_mask] = COLOR_INVALID
                writers["classified"].write(xs, ys, zs, r, g, b)

            if "plants" in writers and plant_mask.any():
                n = int(plant_mask.sum())
                cr, cg, cb = COLOR_PLANT
                writers["plants"].write(
                    xs[plant_mask], ys[plant_mask], zs[plant_mask],
                    np.full(n, cr, np.uint8), np.full(n, cg, np.uint8), np.full(n, cb, np.uint8),
                )

            if "ground" in writers and ground_mask.any():
                n = int(ground_mask.sum())
                cr, cg, cb = COLOR_SOIL
                writers["ground"].write(
                    xs[ground_mask], ys[ground_mask], zs[ground_mask],
                    np.full(n, cr, np.uint8), np.full(n, cg, np.uint8), np.full(n, cb, np.uint8),
                )

            processed += len(x)
            print(f"  {processed:,} / {total_points:,} ({100 * processed / total_points:5.1f}%)", end="\r")

    print("\n\nFinalizing ...")
    for name, w in writers.items():
        w.close()
        print(f"  {name:11s} : {w.count:>12,} -> {w.path}")

    print("\n--- Summary ---")
    print(f"USE_SIDEBAND     : {USE_SIDEBAND}")
    print(f"Soil             : {soil_count:,}")
    print(f"Plant            : {plant_count:,}")
    print(f"Mixed            : {mixed_count:,}")
    print(f"Invalid          : {invalid_count:,}")
    print(f"Demoted cluster  : {demoted_cluster:,}")
    print(f"Forced side soil : {demoted_side:,}")
    print("Done.")


if __name__ == "__main__":
    main()
