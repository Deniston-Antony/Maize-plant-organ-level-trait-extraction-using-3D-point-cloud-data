"""Stage 9 on its own: cross-date matching of a finished run (clusters ->
plants -> stems -> leaves). A full run_pipeline.py run already does this at the
end; use this to (re)run it on an existing run.

    ../../venv/bin/python run_matching.py                       # newest finished run
    ../../venv/bin/python run_matching.py --run outputs/run_... # a given run
    ../../venv/bin/python run_matching.py --until clusters      # steps 1-4 only (~25 s)
    ../../venv/bin/python run_matching.py --retune              # choose the leaf rule again

Output: <run>/cross_date_matching/ (report: cross_date_matching_report.pdf).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline.config import OUTPUT_ROOT  # noqa: E402


def newest_run() -> Path:
    runs = sorted(p for p in OUTPUT_ROOT.glob("run_*") if (p / "alignment" / "alignment.json").exists()
                  and (p / "tracking" / "plant_register.csv").exists())
    if not runs:
        raise SystemExit(f"No finished run in {OUTPUT_ROOT}")
    return runs[-1]


def main() -> None:
    from pipeline.cross_date.stage import run_cross_date
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None, help="run folder (default: newest finished run)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <run>/cross_date_matching)")
    ap.add_argument("--until", default="all", choices=("clusters", "stems", "all"), help="stop after this checkpoint")
    ap.add_argument("--retune", action="store_true", help="choose the leaf-piece rule again (step 7)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    run_dir = a.run or newest_run()
    run_dir = run_dir if run_dir.is_absolute() else (Path.cwd() / run_dir)
    run_cross_date(run_dir, a.out, until=a.until, retune=a.retune, figures=not a.no_figures)


if __name__ == "__main__":
    main()
