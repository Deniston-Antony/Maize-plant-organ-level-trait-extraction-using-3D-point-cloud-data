"""The user's PyVista structure scene inside the Tkinter app.

The scene is built by the user's own script (copy_of_New_optimized_plant_structure.py:
run_structure_from_xyz + _populate_structure_scene, the same layers and
colours as its pyvista_viewer). PyVista cannot be embedded in Tkinter, so the
scene is rendered off-screen and shown as an image; dragging rotates,
the wheel zooms, and the check boxes replace the script's keys 1-8.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from ui.offscreen_view import CheckMenu, OffscreenView

# key -> (label, layer names in the script's scene); keys as in pyvista_viewer()
LAYERS = [
    ("1", "cloud (all points, grey)", ("cloud",)),
    ("2", "stem points", ("stem_pts",)),
    ("3", "leaf points", ("leaf_pts",)),
    ("4", "stem skeleton", ("stem_skel",)),
    ("5", "leaf skeletons", ("leaf_skel",)),
    ("6", "junctions", ("junctions", "junction_hover")),
    ("7", "tips", ("tips",)),
    ("8", "labels", ("leaf_labels", "bottom_label", "top_label")),
]

_lock = threading.Lock()


def layer_vars() -> dict:
    """One check box variable per layer (all on)."""
    return {k: tk.BooleanVar(value=True) for k, _, _ in LAYERS}


def scene_file(npz: Path) -> Path:
    return npz.with_name(npz.name.replace("_labelled.npz", "_user_scene.npz"))


def user_result(npz: Path):
    """(result, prep) of the user's script for a saved plant.

    New runs save <plant>_user_scene.npz during the structure step, so this only
    reads a file. Older runs: the script runs once on the plant and the file is
    written, so the next time is fast too."""
    from pipeline.user_code import scene_arrays, scene_from_arrays, structure_module
    sf = scene_file(npz)
    with _lock:
        if not sf.exists():
            from pipeline.config import PROJECT_ORIGIN
            ref = structure_module()
            xyz = np.load(npz)["xyz"].astype(np.float64) - np.asarray(PROJECT_ORIGIN)   # same input as the pipeline
            cfg = ref.PipelineConfig()
            cfg.output_dir = Path(tempfile.mkdtemp(prefix="structure_view_"))   # keep the run's own files untouched
            with contextlib.redirect_stdout(io.StringIO()):
                res = ref.run_structure_from_xyz(xyz, cfg, show_o3d_steps=False, show_pyvista=False)
            np.savez_compressed(sf, **scene_arrays(res))
        d = np.load(sf)
        return scene_from_arrays({k: d[k] for k in d.files})


def prefetch(npz_files: list[Path]) -> None:
    """Prepare the scene files of the next plants in the background (older runs)."""
    def work():
        for f in npz_files:
            if not scene_file(f).exists():
                try:
                    user_result(f)
                except Exception:  # noqa: BLE001
                    pass
    threading.Thread(target=work, daemon=True).start()


class UserStructureView(OffscreenView):
    loading_text = "Loading the structure of this plant ..."

    def __init__(self, master, npz: Path, title: str, log=print, vis: dict | None = None):
        super().__init__(master, title, log)
        self.npz = Path(npz)
        # layer choices are kept by the caller, so the next plant opens with the same layers
        self.vis = vis if vis is not None else layer_vars()
        bar = self.layer_bar()
        CheckMenu(bar, "Layers", [(f"{k}  {name}", self.vis[k]) for k, name, _ in LAYERS],
                  self._apply_visibility).pack(side="left")
        self.make_view()
        self.in_background(lambda: user_result(self.npz), lambda r: self._build(*r), self._failed)

    def _failed(self, e):
        msg = f"Your structure code failed on this plant: {e}"
        self.show_message(msg)
        self.log(msg)

    def _build(self, res, prep):
        from pipeline.user_code import structure_module
        ref = structure_module()
        pl = self.new_plotter()
        self.actors = ref._populate_structure_scene(pl, res, prep)
        # display only: thicker skeleton lines (the script's scene is unchanged otherwise)
        for name in ("stem_skel", "leaf_skel"):
            a = self.actors.get(name)
            for x in (a if isinstance(a, list) else [a] if a is not None else []):
                x.GetProperty().SetLineWidth(7)
                x.GetProperty().SetRenderLinesAsTubes(True)
        pl.add_text(self.title, font_size=10)
        pl.view_isometric()
        pl.reset_camera()
        self._home = pl.camera_position
        self._apply_visibility()

    def _apply_visibility(self):
        if self.plotter is None:
            return
        for k, _, names in LAYERS:
            for n in names:
                a = self.actors.get(n)
                for x in (a if isinstance(a, list) else [a] if a is not None else []):
                    x.SetVisibility(self.vis[k].get())
        self._render()
