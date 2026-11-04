#!/usr/bin/env python3
"""Run the whole pipeline from the terminal (no app).

    python run_pipeline.py                         # all 5 dates + ground-truth check
    python run_pipeline.py 230621 230629           # only these dates
    python run_pipeline.py --no-gt                 # skip the ground-truth check

Output: outputs/run_<date>_<time>_<dates>/ (open it in the app with "Open run...").
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = HERE / "plant_structure_to_leaf_datasets"
GT_DIR = HERE.parent / "P147" / "manual_segmented_leaf_file"
GT_NAMES = ("LtR_10th_Plant_230621pc_kinematic_calibration", "LtR_11th_Plant_230621pc_kinematic_calibration",
            "LtR_12th_Plant_230621pc_kinematic_calibration", "RL_plant01")
REFERENCE = "230629"


def main() -> None:
    from pipeline.runner import run

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dates", nargs="*", help="YYMMDD dates to run (default: all scans in the dataset folder)")
    p.add_argument("--no-gt", action="store_true", help="skip the ground-truth check")
    p.add_argument("--step", default="all", choices=("soil", "alignment", "stems", "plants", "all"),
                   help="stop after this step")
    a = p.parse_args()

    scans = sorted((f, f.name[:6]) for f in DATA.glob("*.las"))
    if a.dates:
        scans = [s for s in scans if s[1] in a.dates]
    if not scans:
        sys.exit(f"no scans found for {a.dates} in {DATA}")
    dates = [d for _, d in scans]
    ref = REFERENCE if REFERENCE in dates else dates[len(dates) // 2]
    gts = None
    if not a.no_gt:
        gts = [(GT_DIR / f"{n}.bin", "230621") for n in GT_NAMES if (GT_DIR / f"{n}.bin").exists()] or None
    tag = "_".join(dates)                      # dates of the point clouds at the end of the name
    root = HERE / "outputs" / f"run_{time.strftime('%Y%m%d_%H%M')}_{tag}"
    print(f"dates {dates}, reference {ref}, ground truth: {'yes' if gts else 'no'}\noutput {root}\n")
    run(scans, root, ref, gt_items=gts, reuse=False, stop_after=a.step)


if __name__ == "__main__":
    main()
