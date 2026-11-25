"""Run the user's structure code (exactly as the pipeline does, no leaf check)
on the hand-labelled ground-truth plants of 21.06. and cache the result, so
the leaf-piece settings (step 7) can be tuned against the true leaves.

Usage (from the pipeline folder): python -m pipeline.cross_date.gt_structure [gt_cache_dir] [out_dir]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

GT_NAMES = ("LtR_10th_Plant_230621pc_kinematic_calibration", "LtR_11th_Plant_230621pc_kinematic_calibration",
            "LtR_12th_Plant_230621pc_kinematic_calibration", "RL_plant01")


def run(gt_dir: Path, out_dir: Path, log=print) -> None:
    from pipeline.config import PROJECT_ORIGIN, USE_LEAF_CHECK
    from pipeline.ground_truth import load_ground_truth
    from pipeline.user_code import run_user_structure
    O = np.asarray(PROJECT_ORIGIN)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in GT_NAMES:
        f = out_dir / f"{n}.npz"
        if f.exists():
            log(f"  {n}: cached")
            continue
        src = gt_dir / f"{n}_asc"
        if not src.exists():
            log(f"  {n}: no ground truth at {src}")
            continue
        g = load_ground_truth(src, date="230621", log=lambda *a: None)
        t0 = time.time()
        xyz = g.xyz - O
        r = run_user_structure(xyz, None, leaf_check=USE_LEAF_CHECK)
        np.savez_compressed(f, xyz=xyz, gt_labels=g.labels, pred_labels=r["labels"],
                            centerline=np.asarray(r["centerline"]),
                            leaf_points=np.concatenate(r["leaf_paths"]) if r["leaf_paths"] else np.zeros((0, 3)),
                            leaf_sizes=np.array([len(p) for p in r["leaf_paths"]], int),
                            leaf_labels=np.array(r["leaf_labels"], int))
        (out_dir / f"{n}.json").write_text(json.dumps({"traits": r["traits"], "leaves": r["leaves"]}, indent=1, default=float))
        log(f"  {n}: {len(xyz):,} points, {len(r['leaves'])} leaf pieces, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    from pipeline.config import OUTPUT_ROOT
    gt_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_ROOT / "_gt_cache"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_ROOT / "_gt_structure"
    run(gt_dir, out)
