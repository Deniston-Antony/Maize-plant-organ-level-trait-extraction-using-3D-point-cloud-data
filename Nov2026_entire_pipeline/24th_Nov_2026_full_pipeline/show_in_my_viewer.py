#!/usr/bin/env python3
"""Open one plant of a run in the PyVista structure viewer of the user's script
(copy_of_New_optimized_plant_structure.py, run_structure_from_xyz with show_pyvista=True).

    python show_in_my_viewer.py RUN_DIR DATE PLANT_ID
    python show_in_my_viewer.py outputs/run_20261103_2101 230629 group01_plant01

Keys in the window (from the script): 1=cloud 2=stem 3=leaves 4=stem skel
5=leaf skels 6=junctions 7=tips 8=labels.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    from pipeline.user_code import structure_module
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    run, date, pid = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    f = run / date / "03_plants" / pid.split("_")[0] / pid / f"{pid}_labelled.npz"
    from pipeline.config import PROJECT_ORIGIN
    xyz = np.load(f)["xyz"].astype(np.float64) - np.asarray(PROJECT_ORIGIN)   # same input as the pipeline
    ref = structure_module()
    cfg = ref.PipelineConfig()
    cfg.output_dir = f.parent / "structure_output"
    print(f"{pid} {date}: {len(xyz):,} points -> {Path(ref.__file__).name}")
    ref.run_structure_from_xyz(xyz, cfg, show_o3d_steps=False, show_pyvista=True)


if __name__ == "__main__":
    main()
