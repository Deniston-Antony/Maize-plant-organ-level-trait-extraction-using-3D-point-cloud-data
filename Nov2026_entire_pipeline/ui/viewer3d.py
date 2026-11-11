"""Interactive 3D viewer (PyVista) for one ground-truth comparison.

Started by the app in its own process (VTK and Tk must not share a main loop):
    python ui/viewer3d.py <ground_truth/<name>/viewer_data.npz>
Left: ground truth. Right: matched prediction, coloured with the GT leaf ids
it was paired with (grey = not matched). Cameras are linked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import reference_structure as ref  # noqa: E402


def colors(labels: np.ndarray) -> np.ndarray:
    lut = ref._COLOR_LUT
    out = np.full((len(labels), 3), 0.6)
    ok = (labels >= 0) & (labels < len(lut))
    out[ok] = lut[labels[ok]]
    return (out * 255).astype(np.uint8)


def main(npz: str) -> None:
    d = np.load(npz, allow_pickle=False)
    c0 = d["gt_xyz"].mean(axis=0)
    pl = pv.Plotter(shape=(1, 2), window_size=(1500, 800), title=f"GT vs prediction - {Path(npz).parent.name}")
    for k, (xyz, lab, ttl) in enumerate([
        (d["gt_xyz"], d["gt_labels"], f"Ground truth: {Path(npz).parent.name}"),
        (d["pred_xyz"], d["pred_labels"], f"Prediction: {d['pred_id']}  (grey = unmatched)"),
    ]):
        pl.subplot(0, k)
        cloud = pv.PolyData(xyz - c0)
        cloud["rgb"] = colors(lab)
        pl.add_points(cloud, scalars="rgb", rgb=True, point_size=3, render_points_as_spheres=True)
        pl.add_text(ttl, font_size=10)
    pl.link_views()
    pl.show()


if __name__ == "__main__":
    main(sys.argv[1])
