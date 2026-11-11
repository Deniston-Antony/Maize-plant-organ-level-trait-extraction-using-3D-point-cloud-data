"""Run the saved ExtraTrees classifier and stem-guided instance proposal on one LAS scan.

The result is a reviewable set of candidate plant axes, not verified biological
plant IDs. The output cloud contains the observed, downsampled vegetation points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
import rfpipe as R
import step1_downsample
import step2_remove_ground


def scan_name(path: Path) -> str:
    """Separate files with the same name and invalidate caches if a scan changes."""
    stat = path.stat()
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)[:70]
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{clean}_{suffix}"


def stem_location_hints(xyz: np.ndarray, semantic: np.ndarray) -> list[list[float]]:
    """Approximate bases from connected stem predictions in the same scan.

    This reproduces the earlier stem-clustering *idea*, not the older C
    experiment's full geometric repair. The later detector requires observed
    upright tracks before it accepts a hint as a candidate axis.
    """
    stem = np.flatnonzero(semantic == R.CLASS_STEM)
    if len(stem) < 40:
        return []
    cluster = DBSCAN(eps=.05, min_samples=10, n_jobs=2).fit_predict(xyz[stem])
    seeds = [stem[cluster == label] for label in np.unique(cluster)
             if label >= 0 and np.count_nonzero(cluster == label) >= 40]
    if not seeds:
        return []
    centers = np.array([xyz[part, :2].mean(axis=0) for part in seeds])
    used = np.zeros(len(seeds), bool)
    merged = []
    for i in range(len(seeds)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(seeds)):
            if not used[j] and np.linalg.norm(centers[i] - centers[j]) < .12:
                group.append(j)
                used[j] = True
        part = np.concatenate([seeds[j] for j in group])
        if np.ptp(xyz[part, 2]) >= .08:
            low = xyz[part, 2] <= np.percentile(xyz[part, 2], 25)
            merged.append(np.median(xyz[part[low], :2], axis=0))
    merged.sort(key=lambda point: point[1])
    return [point.tolist() for point in merged]


def save_inputs(name: str, xyz: np.ndarray, semantic: np.ndarray,
                centers: list[list[float]], source: Path, ground: Path,
                origin: np.ndarray) -> None:
    target = ROOT / "inputs" / name
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "xyz.npy", xyz.astype(np.float32))
    np.save(target / "semantic.npy", semantic.astype(np.int8))
    np.save(target / "previous_plant_id.npy", np.zeros(len(xyz), np.int32))
    np.save(target / "input_index.npy", np.arange(len(xyz), dtype=np.int32))
    (target / "stem_location_hints.json").write_text(json.dumps({
        "source": "stem clusters estimated from current ExtraTrees prediction",
        "centers_xy": centers,
        "seed_count": len(centers),
        "note": "These hints differ from the earlier C experiment and are not verified plants.",
    }, indent=2), encoding="utf-8")
    (target / "work_config.json").write_text(json.dumps({
        "source": str(source), "ground_removed_cloud": str(ground),
        "local_coordinate_origin_xyz_m": origin.tolist(),
        "coordinate_note": "Add this origin to output XYZ to recover source map coordinates.",
        "downsample_voxel_m": .004,
        "input_index_meaning": "Index into the downsampled, soil-removed cloud; not raw LAS.",
    }, indent=2), encoding="utf-8")
    (target / "semantic_config.json").write_text(json.dumps({
        "source": "bundled ExtraTrees model",
        "model": str(ROOT / "model" / "et_model.joblib"),
        "class_ids": {"0": "stem", "1": "leaf", "-1": "unknown"},
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--las", type=Path, help="Raw maize-row LAS scan")
    source_group.add_argument("--prepared-ground-ply", type=Path,
                              help="Previously prepared RFMF soil-removed ASCII PLY")
    args = parser.parse_args()
    source = (args.las or args.prepared_ground_ply).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Input not found: {source}")
    if args.las and source.suffix.lower() != ".las":
        parser.error("--las requires a .las file")

    name = scan_name(source)
    review_csv = ROOT / "review" / name / "stem_axes_to_review.csv"
    if review_csv.exists():
        print(f"Review already exists: {review_csv}")
        print("The program left it unchanged. Open it or use a different input file.")
        return

    work = ROOT / "runs" / name
    R.DOWN_DIR = work / "downsampled"
    R.GDIR = work / "ground_removed"
    R.CACHE = work / "feature_cache"
    R.MODEL_FILE = ROOT / "model" / "et_model.joblib"
    R.DOWN_DIR.mkdir(parents=True, exist_ok=True)
    R.GDIR.mkdir(parents=True, exist_ok=True)

    ground = R.GDIR / f"G-DRD-{name}.ply"
    if args.las:
        down = R.DOWN_DIR / f"DRD-{name}.ply"
        if not R.ply_is_complete(down):
            print("Step 1/4: downsampling LAS to 4 mm", flush=True)
            step1_downsample.downsample(source, down)
        if not R.ply_is_complete(ground):
            print("Step 2/4: removing soil with CSF and height above ground", flush=True)
            step2_remove_ground.process(down)
    elif not R.ply_is_complete(ground):
        shutil.copy2(source, ground)

    print("Step 3/4: predicting stem and leaf labels with the saved ExtraTrees model", flush=True)
    # Older scikit-learn emits one compatibility warning per tree in this
    # 500-tree model. Keep the warning visible once without flooding the log.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = R.load_model()
    if any(issubclass(item.category, InconsistentVersionWarning) for item in caught):
        print("Warning: model trained with scikit-learn 1.7.2; use the pinned requirements for reproducible predictions.", flush=True)
    xyz, semantic = R.predict_cloud(ground, bundle)
    # LAS map coordinates can be millions of metres. Geometry works in local
    # metres so float32 output retains millimetre-scale point spacing.
    origin = xyz.min(axis=0)
    local_xyz = xyz - origin
    centers = stem_location_hints(local_xyz, semantic)
    save_inputs(name, local_xyz, semantic, centers, source, ground, origin)
    print(f"  {len(xyz):,} vegetation points; {len(centers)} approximate stem hints", flush=True)

    print("Step 4/4: proposing plant instances with stem-guided geometry", flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / "run_review.py"),
                    "--dataset", name], check=True)
    print(f"Open this cloud in CloudCompare: {ROOT / 'review' / name / 'candidate_axes_all_points.ply'}")
    print(f"Review candidate axes here: {review_csv}")


if __name__ == "__main__":
    main()
