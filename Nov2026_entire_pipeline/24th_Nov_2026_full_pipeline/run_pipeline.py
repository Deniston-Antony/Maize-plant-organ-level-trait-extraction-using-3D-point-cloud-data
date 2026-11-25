#!/usr/bin/env python3
"""Run the whole pipeline from the terminal (no app).

    python run_pipeline.py                         # all 5 dates + ground-truth check
    python run_pipeline.py 230621 230629           # only these dates
    python run_pipeline.py --no-gt                 # skip the ground-truth check
    python run_pipeline.py --small-plants 230601   # seedling date: original separation for its plants

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
from pipeline.config import find_up  # noqa: E402
GT_DIR = find_up("P147") / "manual_segmented_leaf_file"
GT_NAMES = ("LtR_10th_Plant_230621pc_kinematic_calibration", "LtR_11th_Plant_230621pc_kinematic_calibration",
            "LtR_12th_Plant_230621pc_kinematic_calibration", "RL_plant01")
REFERENCE = "230629"


def main() -> None:
    from pipeline.runner import run

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dates", nargs="*", help="YYMMDD dates to run (default: all scans in the dataset folder)")
    p.add_argument("--no-gt", action="store_true", help="skip the ground-truth check")
    p.add_argument("--small-plants", nargs="+", default=[], metavar="YYMMDD",
                   help="dates with small plants (seedlings), e.g. 230601")
    p.add_argument("--small-method", default="own", choices=("own", "fadly_small"),
                   help="plant instances on small-plant dates: own = original separation (default), "
                        "fadly_small = instance code with heights x1/4 and a 4 mm graph voxel")
    p.add_argument("--step", default="all", choices=("soil", "alignment", "stems", "plants", "all"),
                   help="stop after this step")
    p.add_argument("--no-cluster-shift", action="store_true",
                   help="stage 4: no fine shift per plant cluster before the register (3b)")
    p.add_argument("--no-identity-check", action="store_true",
                   help="no identity check after the structure (6b): tracking uses the register names only")
    p.add_argument("--no-cross-date", action="store_true", help="skip stage 9 (cross-date matching)")
    p.add_argument("--no-stem-rule", action="store_true",
                   help="register without the one-stem-per-date rule (as before 25.11.2026)")
    p.add_argument("--tag", default="", help="extra text at the end of the run folder name")
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
    root = HERE / "outputs" / (f"run_{time.strftime('%Y%m%d_%H%M')}_{tag}" + (f"_{a.tag}" if a.tag else ""))
    print(f"dates {dates}, reference {ref}, ground truth: {'yes' if gts else 'no'}\noutput {root}\n")
    run(scans, root, ref, gt_items=gts, reuse=False, stop_after=a.step,
        small_plant_dates=a.small_plants, small_plant_method=a.small_method,
        cluster_fine_shift=not a.no_cluster_shift, identity_check=not a.no_identity_check,
        cross_date=not a.no_cross_date, one_stem_per_date=not a.no_stem_rule)


if __name__ == "__main__":
    main()
