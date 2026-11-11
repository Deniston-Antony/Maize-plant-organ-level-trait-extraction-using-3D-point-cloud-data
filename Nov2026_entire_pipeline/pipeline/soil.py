"""Stage 1 + 2: load one scan and separate soil from plants.

Uses the user's script maize_soil_segmentation_main/soil_segmentation/classify_plants.py
directly (its functions and its tuned constants, read from the file at run time):
  CSF on the scan subsampled to CSF_MAX_POINTS (csf_classify)
  -> ground model per DTM_CELL, gaps filled with the lowest nearby ground (build_dtm_improved)
  -> height above ground, soil / plant classes (hag_masks)
  -> isolated-cell filter (build_plant_cluster_mask)
  -> optional side bands as soil (USE_SIDEBAND, estimate_bed_width_frame, side_band_mask_bed).
The script reads the LAS file twice in chunks; here the same functions run on
the scan in memory, in one pass. Coordinates stay in UTM (relative to
PROJECT_ORIGIN) so dates can be matched by position.

Additions that change the plant points are OFF by default (keep the script's
result): keep_low_points (leaf tips 1.2-2 cm above the soil) and
borrow_hidden_ground (ground under a closed canopy taken from an earlier date).
The 1 cm soil raster is only used to align the dates.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import laspy
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

from .config import PROJECT_ORIGIN

Log = callable


def _script_value(name: str, default):
    try:
        from .user_code import soil_module
        return getattr(soil_module(), name)
    except Exception:  # noqa: BLE001
        return default


@dataclass
class SoilConfig:
    # values of classify_plants.py (read from the script when the config is made)
    csf_max_points: int = field(default_factory=lambda: _script_value("CSF_MAX_POINTS", 9_000_000))
    csf_threshold: float = field(default_factory=lambda: _script_value("CSF_THRESHOLD", 0.012))
    dtm_cell: float = field(default_factory=lambda: _script_value("DTM_CELL", 0.03))
    fill_max_dist: float = field(default_factory=lambda: _script_value("DTM_FILL_MAX_DIST", 0.12))
    fill_k: int = field(default_factory=lambda: _script_value("DTM_FILL_K_NEIGHBORS", 12))
    hag_plant_min: float = field(default_factory=lambda: _script_value("HAG_VEG_MIN", 0.020))
    plant_min_neighborhood: int = field(default_factory=lambda: _script_value("PLANT_MIN_NEIGHBORHOOD", 8))
    use_sideband: bool = field(default_factory=lambda: _script_value("USE_SIDEBAND", False))
    x_trim_frac: float = field(default_factory=lambda: _script_value("X_TRIM_FRAC", 0.38))
    hard_side_crop: bool = field(default_factory=lambda: _script_value("HARD_SIDE_CROP", True))
    csf_cloth_res: float = field(default_factory=lambda: _script_value("CSF_CLOTH_RES", 0.02))
    csf_rigidness: int = field(default_factory=lambda: _script_value("CSF_RIGIDNESS", 1))
    csf_time_step: float = field(default_factory=lambda: _script_value("CSF_TIME_STEP", 0.50))
    csf_iterations: int = field(default_factory=lambda: _script_value("CSF_ITERATIONS", 1000))
    # pipeline additions (off = exactly the script's plant points)
    keep_low_points: bool = False    # keep low points (keep_low_min .. 2 cm) touching plant points
    keep_low_min: float = 0.012
    keep_low_dist: float = 0.01
    borrow_hidden_ground: bool = False  # late dates: ground hidden under the canopy from an earlier date
    borrow_when_filled: float = 0.10
    borrow_src_max_filled: float = 0.05
    ground_raster_cell: float = 0.01 # fine soil-height raster for date alignment only
    # ground outlier clean-up after the soil separation (removed from soil AND plants)
    clean_ground_outliers: bool = False   # off: plant points exactly as classify_plants.py gives them
    ground_band_height: float = 0.04  # m above the ground in which outliers are searched
                                      # (01.06: clods < 4 cm, seedlings 6-8 cm)
    below_ground_max: float = 0.03    # points more than 3 cm below the ground model = noise
    outlier_radius: float = 0.01      # isolated point: fewer than outlier_min_neighbors plant
    outlier_min_neighbors: int = 8    # points within outlier_radius (inside the band)
    blob_voxel: float = 0.01          # blobs: connected 1 cm voxels of plant points; a blob that
                                      # stays inside the band (never rises above it) is removed
    save_soil_voxel: float = 0.005    # soil PLY thinned to 5 mm for viewing (0 = all points)
    method: str = "classify_plants.py"
    version: int = 5                 # bump when the stage output changes (invalidates cached results)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scan:
    date: str
    path: Path
    xyz: np.ndarray            # (N, 3) float64, relative to PROJECT_ORIGIN
    rgb: np.ndarray | None     # (N, 3) uint8
    gps_time: np.ndarray | None


@dataclass
class DTM:
    x0: float
    y0: float
    cell: float
    z: np.ndarray              # (nx, ny) ground height, nan = unknown
    observed: np.ndarray       # cell had ground points
    fill_dist: np.ndarray      # metres to the nearest observed cell used (0 = observed, nan = unknown)

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.clip(((x - self.x0) / self.cell).astype(np.int64), 0, self.z.shape[0] - 1)
        iy = np.clip(((y - self.y0) / self.cell).astype(np.int64), 0, self.z.shape[1] - 1)
        return self.z[ix, iy]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, x0=self.x0, y0=self.y0, cell=self.cell, z=self.z.astype(np.float32),
                            observed=self.observed, fill_dist=self.fill_dist.astype(np.float32),
                            origin=np.array(PROJECT_ORIGIN))

    @classmethod
    def load(cls, path: Path) -> "DTM":
        d = np.load(path)
        return cls(float(d["x0"]), float(d["y0"]), float(d["cell"]), d["z"].astype(np.float64),
                   d["observed"], d["fill_dist"].astype(np.float64))


def load_scan(path: Path, date: str) -> Scan:
    las = laspy.read(str(path))
    o = np.asarray(PROJECT_ORIGIN)
    xyz = np.column_stack([np.asarray(las.x) - o[0], np.asarray(las.y) - o[1], np.asarray(las.z) - o[2]])
    dims = set(las.point_format.dimension_names)
    rgb = None
    if {"red", "green", "blue"} <= dims:
        r = np.column_stack([las.red, las.green, las.blue])
        if r.max() > 0:
            rgb = (r >> 8).astype(np.uint8) if r.max() > 255 else r.astype(np.uint8)
    gps = np.asarray(las.gps_time, dtype=np.float64) if "gps_time" in dims else None
    return Scan(date, Path(path), xyz, rgb, gps)


def _thin_index(xyz: np.ndarray, voxel: float) -> np.ndarray:
    k = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, idx = np.unique(key, return_index=True)
    return idx


def csf_ground(xyz: np.ndarray, cfg: SoilConfig) -> np.ndarray:
    import CSF
    c = CSF.CSF()
    c.params.bSloopSmooth = False
    c.params.cloth_resolution = cfg.csf_cloth_res
    c.params.rigidness = cfg.csf_rigidness
    c.params.time_step = cfg.csf_time_step
    c.params.class_threshold = cfg.csf_threshold
    try:
        c.params.iterations = cfg.csf_iterations
    except AttributeError:
        c.params.interations = cfg.csf_iterations
    c.setPointCloud(xyz.astype(np.float64))
    g, n = CSF.VecInt(), CSF.VecInt()
    c.do_filtering(g, n, exportCloth=False)
    mask = np.zeros(len(xyz), bool)
    mask[np.asarray(g, dtype=np.int64)] = True
    return mask


def build_dtm(ground: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float,
              cell: float, fill_max_dist: float, fill_k: int) -> DTM:
    """Median ground height per cell; empty cells take the lowest of the k
    nearest observed cells within fill_max_dist (as in classify_plants.py)."""
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    ny = int(np.ceil((ymax - ymin) / cell)) + 1
    ix = np.clip(((ground[:, 0] - xmin) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((ground[:, 1] - ymin) / cell).astype(np.int64), 0, ny - 1)
    flat = ix * ny + iy
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], ground[order, 2]
    starts = np.flatnonzero(np.r_[True, np.diff(fs) != 0])
    ends = np.r_[starts[1:], len(fs)]
    z = np.full(nx * ny, np.nan)
    z[fs[starts]] = [np.median(zs[s:e]) for s, e in zip(starts, ends)]
    z = z.reshape(nx, ny)
    observed = np.isfinite(z)
    fill = np.where(observed, 0.0, np.nan)
    oi, oj = np.where(observed)
    gi, gj = np.where(~observed)
    if len(oi) and len(gi):
        k = min(fill_k, len(oi))
        d, nn = cKDTree(np.column_stack([oi, oj]).astype(float)).query(np.column_stack([gi, gj]).astype(float), k=k)
        d, nn = d.reshape(len(gi), k) * cell, nn.reshape(len(gi), k)
        zo = z[oi, oj]
        cand = np.where(d <= fill_max_dist, zo[nn], np.nan)
        zf = np.nanmin(np.where(np.isfinite(cand), cand, np.inf), axis=1)
        ok = np.isfinite(zf)
        z[gi[ok], gj[ok]] = zf[ok]
        fill[gi[ok], gj[ok]] = d[ok, 0]
    return DTM(xmin, ymin, cell, z, observed, fill)


@dataclass
class SoilResult:
    date: str
    plant_idx: np.ndarray      # indices into scan.xyz
    hag: np.ndarray            # (N,) float32, nan = no ground model
    dtm: DTM
    stats: dict
    ground_raster: dict | None = None   # ix, iy (cell index from the project origin), z (median soil height)
    low_flag: np.ndarray | None = None  # (N,) True = low point kept as plant (leaf tip near the soil)
    outlier: np.ndarray | None = None   # (N,) True = ground outlier, removed from soil and plants


def _script_with(cfg: SoilConfig):
    """classify_plants.py with this run's values set (hag_masks / csf_classify read module constants)."""
    from .user_code import soil_module
    cp = soil_module()
    cp.CSF_CLOTH_RES, cp.CSF_RIGIDNESS, cp.CSF_TIME_STEP = cfg.csf_cloth_res, cfg.csf_rigidness, cfg.csf_time_step
    cp.CSF_ITERATIONS, cp.CSF_THRESHOLD = cfg.csf_iterations, cfg.csf_threshold
    cp.HAG_VEG_MIN = cfg.hag_plant_min
    cp.HAG_SOIL_MAX = min(cp.HAG_SOIL_MAX, cfg.hag_plant_min)
    return cp


def _to_pipeline_dtm(sdtm) -> DTM:
    """Script DTM -> pipeline DTM (adds how far each filled cell's height came from)."""
    from scipy.ndimage import distance_transform_edt
    observed = sdtm.observed.copy()
    dist = distance_transform_edt(~observed) * sdtm.cell
    fill = np.where(observed, 0.0, np.where(np.isfinite(sdtm.z_grid), dist, np.nan))
    return DTM(float(sdtm.x0), float(sdtm.y0), float(sdtm.cell), sdtm.z_grid.astype(np.float64), observed, fill)


def remove_ground_outliers(xyz: np.ndarray, hag: np.ndarray, plant: np.ndarray, cfg: SoilConfig):
    """Outliers near the ground, removed from both soil and plants.

    1. points more than below_ground_max below the ground model;
    2. plant points inside the band (HAG < ground_band_height) with fewer than
       outlier_min_neighbors plant points within outlier_radius;
    3. plant blobs (connected blob_voxel voxels) that stay inside the band:
       clods, debris, stones. Leaf tips lying on the soil are kept, because
       they are connected to their plant above the band.
    Returns (new plant mask, removed mask, stats).
    """
    from scipy.ndimage import label
    removed = np.isfinite(hag) & (hag < -cfg.below_ground_max)
    n_below = int(removed.sum())
    band = plant & (hag < cfg.ground_band_height)
    n_iso = 0
    if band.any():
        near = np.where(plant & (hag < cfg.ground_band_height + 2 * cfg.outlier_radius))[0]
        bi = np.where(band)[0]
        cnt = cKDTree(xyz[near]).query_ball_point(xyz[bi], cfg.outlier_radius, return_length=True, workers=-1) - 1
        iso = bi[cnt < cfg.outlier_min_neighbors]
        removed[iso] = True
        n_iso = len(iso)
    keep = plant & ~removed
    n_blob_pts = n_blobs = 0
    pi = np.where(keep)[0]
    if len(pi):
        v = cfg.blob_voxel
        lo = xyz[pi].min(axis=0)
        k = np.floor((xyz[pi] - lo) / v).astype(np.int64)
        shape = k.max(axis=0) + 1
        grid = np.zeros(shape, bool)
        grid[k[:, 0], k[:, 1], k[:, 2]] = True
        lab, n = label(grid, structure=np.ones((3, 3, 3), bool))
        pl = lab[k[:, 0], k[:, 1], k[:, 2]]
        del grid, lab
        top = np.full(n + 1, -np.inf)
        np.maximum.at(top, pl, hag[pi].astype(np.float64))
        low = top < cfg.ground_band_height
        low[0] = False
        drop = low[pl]
        removed[pi[drop]] = True
        n_blob_pts, n_blobs = int(drop.sum()), int(low.sum())
    stats = {"ground_band_height_m": cfg.ground_band_height, "n_removed_below_ground": n_below,
             "n_removed_isolated_in_band": n_iso, "n_low_blobs_removed": n_blobs,
             "n_removed_low_blob_points": n_blob_pts, "n_ground_outliers_removed": int(removed.sum())}
    return plant & ~removed, removed, stats


def separate_soil(scan: Scan, cfg: SoilConfig | None = None, log=print) -> SoilResult:
    """classify_plants.py on one scan (in memory)."""
    cfg = cfg or SoilConfig()
    cp = _script_with(cfg)
    t0 = time.time()
    xyz = scan.xyz
    # exactly as the script: raw UTM coordinates from the file, grid bounds from the LAS header
    with laspy.open(str(scan.path)) as f:
        h = f.header
        bounds = (float(h.x_min), float(h.y_min), float(h.x_max), float(h.y_max))
    las = laspy.read(str(scan.path))
    x, y, z = (np.asarray(las.x, np.float64), np.asarray(las.y, np.float64), np.asarray(las.z, np.float64))
    del las
    # CSF subsample: every stride-th point, as load_las_subsample() in the script
    stride = max(1, (len(x) + cfg.csf_max_points - 1) // cfg.csf_max_points)
    sub = slice(None, None, stride)
    with contextlib.redirect_stdout(io.StringIO()):
        g = cp.csf_classify(x[sub], y[sub], z[sub])
        sdtm = cp.build_dtm_improved(xg=x[sub][g], yg=y[sub][g], zg=z[sub][g], xmin=bounds[0], ymin=bounds[1],
                                     xmax=bounds[2], ymax=bounds[3], cell=cfg.dtm_cell,
                                     fill_max_dist=cfg.fill_max_dist, fill_k=cfg.fill_k)
    hag64 = z - sdtm.height(x, y)
    _, _, plant_raw, _, _ = cp.hag_masks(hag64)
    ix, iy = cp.xy_to_ij(x, y, sdtm)
    grid = np.zeros((sdtm.nx, sdtm.ny), np.int32)
    np.add.at(grid, (ix[plant_raw], iy[plant_raw]), 1)
    ok_cell = cp.build_plant_cluster_mask(grid, cfg.plant_min_neighborhood)
    plant = plant_raw & ok_cell[ix, iy]
    n_side = 0
    if cfg.use_sideband and cfg.x_trim_frac > 0:
        o_, across, w_lo, w_hi, _, _ = cp.estimate_bed_width_frame(x[sub], y[sub], cfg.x_trim_frac)
        side = cp.side_band_mask_bed(x, y, o_, across, w_lo, w_hi)
        n_side = int((plant & side).sum())
        plant &= ~side
    del x, y, z
    # pipeline frame (relative to PROJECT_ORIGIN) for everything after this
    O = np.asarray(PROJECT_ORIGIN)
    sdtm.x0, sdtm.y0 = sdtm.x0 - O[0], sdtm.y0 - O[1]
    sdtm.z_grid = sdtm.z_grid - O[2]
    dtm = _to_pipeline_dtm(sdtm)
    hag = hag64.astype(np.float32)
    del hag64
    n_raw = int(plant_raw.sum())

    # keep low points that touch plant points (leaf tips near the soil)
    low = np.isfinite(hag) & (hag >= cfg.keep_low_min) & (hag < cfg.hag_plant_min) & ok_cell[ix, iy]
    n_low_kept = 0
    low_flag = np.zeros(len(xyz), bool)
    if cfg.keep_low_points and low.any() and plant.any():
        li = np.where(low)[0]
        pv = xyz[plant][hag[plant] < cfg.hag_plant_min + 0.03]       # plant points near the soil only
        if len(pv):
            d, _ = cKDTree(pv[_thin_index(pv, 0.004)]).query(xyz[li], k=1, distance_upper_bound=cfg.keep_low_dist,
                                                               workers=-1)
            keep = li[np.isfinite(d)]
            plant[keep] = True
            low_flag[keep] = True
            n_low_kept = len(keep)

    # ground outliers (removed from soil and plants)
    outlier = np.zeros(len(xyz), bool)
    o_stats = {}
    if cfg.clean_ground_outliers:
        plant, outlier, o_stats = remove_ground_outliers(xyz, hag, plant, cfg)
        low_flag &= plant
        log(f"  ground outliers {scan.date}: {o_stats['n_ground_outliers_removed']:,} points removed "
            f"({o_stats['n_removed_below_ground']:,} below ground, {o_stats['n_removed_isolated_in_band']:,} isolated, "
            f"{o_stats['n_low_blobs_removed']:,} blobs below {cfg.ground_band_height * 100:.0f} cm with "
            f"{o_stats['n_removed_low_blob_points']:,} points)")

    # fine soil raster (median height of soil points per 1 cm cell, grid anchored
    # at the project origin so every date uses the same cells)
    soil_pts = np.isfinite(hag) & (np.abs(hag) <= cfg.csf_threshold) & ~outlier
    gc_ = cfg.ground_raster_cell
    gx = np.floor(xyz[soil_pts, 0] / gc_).astype(np.int64)
    gy = np.floor(xyz[soil_pts, 1] / gc_).astype(np.int64)
    key = (gx << 32) + gy
    order = np.argsort(key, kind="stable")
    ks, zs_ = key[order], xyz[soil_pts, 2][order]
    st_ = np.flatnonzero(np.r_[True, np.diff(ks) != 0])
    zmed = np.array([np.median(zs_[a:b]) for a, b in zip(st_, np.r_[st_[1:], len(ks)])], np.float32)
    ground_raster = {"ix": (ks[st_] >> 32).astype(np.int32), "iy": (ks[st_] & 0xFFFFFFFF).astype(np.int32),
                     "z": zmed, "cell": gc_}

    pidx = np.where(plant)[0]
    near_plants = ok_cell & (grid > 0)
    stats = {
        "date": scan.date, "n_points": int(len(xyz)), "n_plant": int(len(pidx)),
        "plant_share": float(len(pidx) / len(xyz)), "n_plant_raw": n_raw, "n_low_kept_as_plant": n_low_kept,
        "n_no_ground_model": int((~np.isfinite(hag)).sum()), "fill_max_dist_used": cfg.fill_max_dist,
        "n_side_band_removed": n_side, "method": "classify_plants.py",
        "csf_points": int(len(g)), "csf_subsample_stride": stride, "csf_ground_points": int(g.sum()),
        "dtm_cells_observed_share": float(dtm.observed.mean()),
        "dtm_cells_under_plants_filled_share": float(np.mean(~dtm.observed[near_plants])) if near_plants.any() else 0.0,
        "dtm_fill_dist_under_plants_p95_m": float(np.nanpercentile(dtm.fill_dist[near_plants], 95)) if near_plants.any() else 0.0,
        **o_stats,
        "seconds": round(time.time() - t0, 1),
    }
    log(f"  soil {scan.date}: {len(pidx):,} plant points ({stats['plant_share']:.1%}), "
        f"{n_low_kept:,} low points kept, ground under plants filled in "
        f"{stats['dtm_cells_under_plants_filled_share']:.0%} of cells, {stats['seconds']} s")
    return SoilResult(scan.date, pidx, hag, dtm, stats, ground_raster, low_flag, outlier)


def save_soil(out_dir: Path, scan: Scan, res: SoilResult, cfg: SoilConfig) -> None:
    from .io_utils import write_ply
    p = res.plant_idx
    np.savez(out_dir / "plants.npz", xyz=scan.xyz[p].astype(np.float32), hag=res.hag[p],
             rgb=scan.rgb[p] if scan.rgb is not None else np.zeros((0, 3), np.uint8),
             gps_time=scan.gps_time[p] if scan.gps_time is not None else np.zeros(0),
             low_kept=res.low_flag[p] if res.low_flag is not None else np.zeros(len(p), bool),
             origin=np.array(PROJECT_ORIGIN))
    write_ply(out_dir / f"{scan.date}_plants.ply", scan.xyz[p] + np.asarray(PROJECT_ORIGIN),
              rgb=scan.rgb[p] if scan.rgb is not None else None)
    O = np.asarray(PROJECT_ORIGIN)
    is_plant = np.zeros(len(scan.xyz), bool)
    is_plant[p] = True
    out_m = res.outlier if res.outlier is not None else np.zeros(len(scan.xyz), bool)
    soil = np.where(~is_plant & ~out_m & np.isfinite(res.hag))[0]
    if cfg.save_soil_voxel > 0:
        soil = soil[_thin_index(scan.xyz[soil], cfg.save_soil_voxel)]
    write_ply(out_dir / f"{scan.date}_soil.ply", scan.xyz[soil] + O, rgb=scan.rgb[soil] if scan.rgb is not None else None)
    oi = np.where(out_m)[0]
    write_ply(out_dir / f"{scan.date}_ground_outliers.ply", scan.xyz[oi] + O,
              rgb=scan.rgb[oi] if scan.rgb is not None else None)
    res.dtm.save(out_dir / "dtm.npz")
    if res.ground_raster is not None:
        np.savez_compressed(out_dir / "ground_1cm.npz", **res.ground_raster)
    (out_dir / "soil_summary.json").write_text(json.dumps({**res.stats, "config": cfg.to_dict(),
                                                            "source": str(scan.path)}, indent=2))


def load_ground_raster(out_dir: Path) -> dict | None:
    f = out_dir / "ground_1cm.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {k: d[k] for k in d.files}


def load_plants(out_dir: Path) -> dict:
    d = np.load(out_dir / "plants.npz")
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- ground from an earlier date
def borrow_ground(dtm: DTM, src: DTM, to_src_xy, dz_src_minus_date: float, near_plants_only: np.ndarray | None = None,
                  bias_radius_cells: int = 10) -> tuple[DTM, dict]:
    """Replace the filled (hidden) cells of `dtm` with the ground model of an
    earlier date `src` where that date saw the soil. `to_src_xy(xy)` maps
    this date's coordinates into the source date's frame; heights are moved
    by the z offset between the dates and by the local difference between the
    two ground models around the cell (soil can settle between dates).

    The original fill takes the LOWEST nearby ground, so under a closed canopy
    the row furrow becomes flat and too deep (seen on 05.07 / 11.07)."""
    nx, ny = dtm.z.shape
    gi, gj = np.where(~dtm.observed & np.isfinite(dtm.z))
    out = DTM(dtm.x0, dtm.y0, dtm.cell, dtm.z.copy(), dtm.observed.copy(), dtm.fill_dist.copy())
    if not len(gi):
        return out, {"cells_borrowed": 0}
    def src_z(xy):
        sxy = to_src_xy(xy)
        ix = np.clip(((sxy[:, 0] - src.x0) / src.cell).astype(np.int64), 0, src.z.shape[0] - 1)
        iy = np.clip(((sxy[:, 1] - src.y0) / src.cell).astype(np.int64), 0, src.z.shape[1] - 1)
        z = src.z[ix, iy] + dz_src_minus_date
        return np.where(src.observed[ix, iy], z, np.nan)
    cx = dtm.x0 + (np.arange(nx) + 0.5) * dtm.cell
    cy = dtm.y0 + (np.arange(ny) + 0.5) * dtm.cell
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    zb = src_z(np.column_stack([X.ravel(), Y.ravel()])).reshape(nx, ny)
    # local bias: observed ground of this date minus borrowed ground, smoothed over ~30 cm
    from scipy.ndimage import uniform_filter
    diff = np.where(dtm.observed & np.isfinite(zb), dtm.z - zb, 0.0)
    cnt = uniform_filter((dtm.observed & np.isfinite(zb)).astype(float), 2 * bias_radius_cells + 1)
    bias = uniform_filter(diff, 2 * bias_radius_cells + 1) / np.maximum(cnt, 1e-6)
    glob = float(np.median((dtm.z - zb)[dtm.observed & np.isfinite(zb)])) if np.any(dtm.observed & np.isfinite(zb)) else 0.0
    bias = np.where(cnt > 0.05, bias, glob)
    take = np.isfinite(zb[gi, gj])
    if near_plants_only is not None:
        take &= near_plants_only[gi, gj]
    new = zb[gi, gj] + bias[gi, gj]
    old = dtm.z[gi, gj]
    out.z[gi[take], gj[take]] = new[take]
    out.fill_dist[gi[take], gj[take]] = -1.0          # -1 = borrowed from an earlier date
    ch = (new - old)[take]
    return out, {"cells_borrowed": int(take.sum()), "cells_hidden": int(len(gi)),
                 "ground_change_median_cm": float(np.median(ch) * 100) if len(ch) else 0.0,
                 "ground_change_p95_cm": float(np.percentile(ch, 95) * 100) if len(ch) else 0.0,
                 "global_bias_cm": glob * 100}
