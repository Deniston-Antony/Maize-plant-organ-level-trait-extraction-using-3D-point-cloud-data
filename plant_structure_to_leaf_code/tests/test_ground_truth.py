"""Self-test for the ground-truth checker.

Builds a small fake 'run' from the 4 hand-labelled plants of 21.06
(P147/manual_segmented_leaf_file/iou_evaluation/*_asc): each plant is
segmented automatically with the reference method and stored as a
prediction. Then several GT input styles are checked against it:

  A  CloudCompare ASC folders (Classification column)       -> position_overlap
  B  one PLY file with a 'label' field                      -> position_overlap
  C  folder named by organ, one leaf split in two pieces    -> pieces merged
  D  no stem file + no date in the name                     -> stem N/A, all dates tried
  E  coordinates shifted by (-357000, -5610000, -200)       -> shape_icp
  F  unlabelled single plant (no labels)                    -> plant-level only

Run:  venv/bin/python plant_structure_to_leaf_code/tests/test_ground_truth.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from pipeline import reference_structure as ref  # noqa: E402
from pipeline.config import OUTPUT_ROOT, RunPaths  # noqa: E402
from pipeline.ground_truth import PredPlant, load_ground_truth, run_ground_truth_check, save_predicted_plant  # noqa: E402
from pipeline.io_utils import write_ply  # noqa: E402
from pipeline.structure import structure_from_labels  # noqa: E402

GT_SRC = CODE.parent / "P147" / "manual_segmented_leaf_file" / "iou_evaluation"
PLANTS = ["LtR_10th_Plant_230621pc_kinematic_calibration", "LtR_11th_Plant_230621pc_kinematic_calibration",
          "LtR_12th_Plant_230621pc_kinematic_calibration", "RL_plant01"]


def build_fake_run() -> RunPaths:
    run = RunPaths(OUTPUT_ROOT / "_selftest_run")
    if run.root.exists():
        shutil.rmtree(run.root)
    cfg = ref.PipelineConfig()
    for k, name in enumerate(PLANTS, start=1):
        gt = load_ground_truth(GT_SRC / f"{name}_asc", date="230621", log=lambda *_: None)
        xyz, lab = ref.classify_plant_for_iou(gt.xyz, cfg)
        st = structure_from_labels(xyz, lab)
        pid = f"group{k:02d}_plant01"
        save_predicted_plant(run, "230621", PredPlant(pid, xyz, lab, np.array([st.traits[c] for c in ("base_x", "base_y", "base_z")]),
                                                      st.traits, st.leaves, st.merges))
        print(f"prediction {pid} <- {name}: {len(xyz):,} pts, {st.traits['n_leaves']} leaves, stem={st.stem_status}")
    return run


def build_variants(run: RunPaths) -> list[tuple[Path, str | None, str]]:
    vdir = run.root / "_gt_inputs"
    vdir.mkdir(parents=True, exist_ok=True)
    q = lambda *_: None
    items = []
    # A: original ASC folders
    for k, name in enumerate(PLANTS[:3], start=1):
        items.append((GT_SRC / f"{name}_asc", None, f"group{k:02d}_plant01"))
    # B: single labelled PLY
    g = load_ground_truth(GT_SRC / f"{PLANTS[1]}_asc", log=q)
    write_ply(vdir / "230621_LtR11_single_labelled.ply", g.xyz, labels=g.labels)
    items.append((vdir / "230621_LtR11_single_labelled.ply", None, "group02_plant01"))
    # C: organ-named files, largest leaf split into two pieces
    g = load_ground_truth(GT_SRC / f"{PLANTS[2]}_asc", log=q)
    cdir = vdir / "230621_LtR12_organ_named"
    cdir.mkdir(exist_ok=True)
    write_ply(cdir / "LtR12_stem.ply", g.xyz[g.labels == 0])
    big = max(g.leaf_ids, key=lambda i: np.count_nonzero(g.labels == i))
    for lid in g.leaf_ids:
        pts = g.xyz[g.labels == lid]
        if lid == big:
            d = np.linalg.norm(pts[:, :2] - pts[:, :2].mean(0), axis=1)
            write_ply(cdir / f"LtR12_leaf{lid:02d}_part1.ply", pts[d <= np.median(d)])
            write_ply(cdir / f"LtR12_leaf{lid:02d}_part2.ply", pts[d > np.median(d)])
        else:
            write_ply(cdir / f"LtR12_leaf{lid:02d}.ply", pts)
    items.append((cdir, None, "group03_plant01"))
    # D: RL_plant01 without stem, no date in name
    g = load_ground_truth(GT_SRC / f"{PLANTS[3]}_asc", log=q)
    m = g.labels > 0
    write_ply(vdir / "RL_plant01_nostem.ply", g.xyz[m], labels=g.labels[m])
    items.append((vdir / "RL_plant01_nostem.ply", None, "group04_plant01"))
    # E: shifted coordinates (ASC with Classification column)
    g = load_ground_truth(GT_SRC / f"{PLANTS[0]}_asc", log=q)
    s = g.xyz - [357000.0, 5610000.0, 200.0]
    np.savetxt(vdir / "230621_LtR10_shifted.asc", np.column_stack([s, g.labels]), fmt="%.6f",
               header="X Y Z Classification", comments="//")
    items.append((vdir / "230621_LtR10_shifted.asc", None, "group01_plant01"))
    # F: unlabelled plant
    g = load_ground_truth(GT_SRC / f"{PLANTS[1]}_asc", log=q)
    write_ply(vdir / "unlabelled_plant.ply", g.xyz)
    items.append((vdir / "unlabelled_plant.ply", None, "group02_plant01"))
    return items


def main() -> None:
    run = build_fake_run()
    items = build_variants(run)
    res = run_ground_truth_check(run.root, [(p, d) for p, d, _ in items])
    print("\n=== RESULT ===")
    ok_all = True
    for (p, _, expect), r in zip(items, res):
        ok = r.get("matched_plant") == expect
        ok_all &= ok
        print(f"{'OK ' if ok else 'BAD'} {Path(p).name:<55} -> {r.get('matched_plant')!s:<16} "
              f"{r.get('match_method')!s:<17} plantIoU={r.get('plant_iou', float('nan')):.3f} "
              f"stemIoU={r.get('stem_iou', float('nan')):.3f} leafIoU={r.get('mean_leaf_iou_with_fragments', float('nan')):.3f}")
    print("ALL MATCHES CORRECT" if ok_all else "SOME MATCHES WRONG")


if __name__ == "__main__":
    main()
