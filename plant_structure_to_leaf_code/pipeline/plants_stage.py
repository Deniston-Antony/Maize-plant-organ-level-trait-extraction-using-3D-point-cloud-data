"""Stage 5 + 6: plant separation and per-plant structure for one date.

For every plant of the register present on this date, its points go through
the user's structure script (Research_project/copy_of_New_optimized_plant_structure.py,
run_structure_from_xyz: cleaning, stem/leaf segmentation, leaf clusters,
stem centreline, midribs, measurements). Only the stem radius is added
(median circle fit along the stem; the script does not measure it). The
plant is saved in the format the ground-truth checker reads
(<plant_id>_labelled.npz, <plant_id>_traits.json, plants_index.csv), plus a
labelled PLY, the skeleton (<plant_id>_skeleton.npz) and the script's own
outputs in structure_output/.
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import PLANT_WORKERS, PROJECT_ORIGIN

SEEDLING_HEIGHT = 0.15   # m; below this no stem radius is reported
MIN_SEG_POINTS = 800      # below this the plant is too small to segment (seedlings: height only)


def _structure_job(args):
    """Worker: the user's structure code on one plant (local coordinates)."""
    pid, xyz, base, out_dir = args
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from .user_code import run_user_structure
        from .config import USE_LEAF_CHECK
        r = run_user_structure(xyz, out_dir, leaf_check=USE_LEAF_CHECK)
        skel = {"centerline": r["centerline"], "leaf_paths": r["leaf_paths"], "leaf_labels": r["leaf_labels"],
                "scene": r.get("scene")}
        return pid, xyz, r["labels"], r["traits"], r["leaves"], [], [], None, skel
    except Exception:
        err = traceback.format_exc(limit=3)
        # the script stops on plants it cannot structure (e.g. < 10 stem points): keep height only
        t = _height_only(xyz, base)
        t["stem_status"] = "structure_failed"
        t["note"] = "structure script failed: " + err.strip().splitlines()[-1]
        return pid, xyz, np.full(len(xyz), -1, np.int32), t, [], [], [t["note"]], None, None


def _height_only(xyz: np.ndarray, base: np.ndarray) -> dict:
    """Seedlings / tiny plants: only height and point count."""
    return {"base_x": float(base[0] + PROJECT_ORIGIN[0]), "base_y": float(base[1] + PROJECT_ORIGIN[1]),
            "base_z": float(base[2] + PROJECT_ORIGIN[2]), "plant_height_m": float(xyz[:, 2].max() - base[2]),
            "stem_status": "too_small_to_segment", "n_leaves": float("nan")}


def run_structures(jobs: list[tuple], log=print, workers: int = PLANT_WORKERS,
                   on_done=None) -> dict:
    """jobs: (plant_id, xyz local, base local, folder for the script's own outputs). Returns plant_id -> result tuple."""
    out = {}
    big = [j for j in jobs if len(j[1]) >= MIN_SEG_POINTS]
    for pid, xyz, base, _ in jobs:
        if len(xyz) < MIN_SEG_POINTS:
            out[pid] = (pid, xyz, np.full(len(xyz), -1, np.int32), _height_only(xyz, base), [], [], ["too few points"], None, None)
    if not big:
        return out
    # largest first so the long jobs start early
    big.sort(key=lambda j: -len(j[1]))
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=min(workers, len(big)), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_structure_job, j): j[0] for j in big}
        for k, f in enumerate(as_completed(futs)):
            r = f.result()
            out[r[0]] = r
            if r[3].get("stem_status") == "structure_failed":
                log(f"    ! {r[0]}: {r[3]['note']}")
            if on_done:
                on_done(k + 1, len(big))
    return out


def save_plant(run_p, date: str, pid: str, res, base_local: np.ndarray, presence: dict) -> dict:
    from .ground_truth import PredPlant, save_predicted_plant
    from .io_utils import write_ply
    O = np.asarray(PROJECT_ORIGIN)
    _, sx, sl, traits, leaves, merges, warns, err, skel = res
    if err or sx is None:
        return {"plant_id": pid, "date": date, "status": "structure_failed", "error": (err or "").strip().splitlines()[-1:]}
    traits = dict(traits)
    if "base_x" not in traits:                                  # plant position = stem base at the soil
        traits.update(base_x=float(base_local[0]), base_y=float(base_local[1]), base_z=float(base_local[2]))
    if abs(traits["base_x"]) < 1e5:                             # local -> UTM
        for k, o_ in zip(("base_x", "base_y", "base_z"), O):
            traits[k] = float(traits[k] + o_)
    # seedlings: stems of 3-5 mm are at the scan noise level -> no stem radius (agreed 2026-11-01)
    if float(traits.get("plant_height_m", 1.0)) < SEEDLING_HEIGHT:
        for k in ("stem_radius_m", "stem_diameter_mm"):
            if k in traits:
                traits[k] = float("nan")
        traits["note"] = "seedling: height and leaf count only"
    for k, v in presence.items():
        if k != "axis":
            traits[f"presence_{k}"] = v
    xyz_utm = sx + O
    base_utm = base_local + O
    save_predicted_plant(run_p, date, PredPlant(pid, xyz_utm, sl.astype(np.int32), base_utm, traits, leaves, merges))
    write_ply(run_p.plant_dir(date, pid) / f"{pid}_labelled.ply", xyz_utm, labels=sl.astype(np.int32))
    if skel is not None:
        paths = skel["leaf_paths"]
        np.savez_compressed(run_p.plant_dir(date, pid) / f"{pid}_skeleton.npz",
                            centerline=np.asarray(skel["centerline"]) + O,
                            leaf_points=np.concatenate([np.asarray(p) + O for p in paths]) if paths else np.zeros((0, 3)),
                            leaf_sizes=np.array([len(p) for p in paths], int),
                            leaf_labels=np.array(skel["leaf_labels"], int), origin=O)
        if skel.get("scene") is not None:      # for the app: the script's own 3D view without recomputing
            np.savez_compressed(run_p.plant_dir(date, pid) / f"{pid}_user_scene.npz", **skel["scene"])
    if warns:
        (run_p.plant_dir(date, pid) / f"{pid}_warnings.txt").write_text("\n".join(map(str, warns)))
    return {"plant_id": pid, "date": date, "status": "ok", **{k: v for k, v in traits.items() if not isinstance(v, (list, dict))}}
