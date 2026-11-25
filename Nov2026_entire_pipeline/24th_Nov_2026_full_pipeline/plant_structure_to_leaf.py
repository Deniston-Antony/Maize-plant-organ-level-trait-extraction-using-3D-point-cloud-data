#!/usr/bin/env python3
"""Maize field pipeline - entry point.

    python plant_structure_to_leaf.py                 # open the Tkinter app
    python plant_structure_to_leaf.py gt-check RUN_DIR GT [GT ...] [--date 230621]
                                                      # ground-truth check without the app

Ground truth (GT) = a folder with one file per organ (..._stem, ..._leafNN,
..._leafNN_partK) or one file with a Classification/label field, or a
CloudCompare .bin. See ui/help_content.py (or the (?) button in the app).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gt-check", help="compare ground truth with the predictions of a run")
    g.add_argument("run_dir", type=Path)
    g.add_argument("gt", type=Path, nargs="+")
    g.add_argument("--date", default=None, help="YYMMDD for all GT inputs (default: from the name, else all dates)")
    args = p.parse_args()

    if args.cmd == "gt-check":
        from pipeline.ground_truth import run_ground_truth_check
        run_ground_truth_check(args.run_dir, [(x, args.date) for x in args.gt])
    else:
        from ui.app import main as app_main
        app_main()


if __name__ == "__main__":
    main()
