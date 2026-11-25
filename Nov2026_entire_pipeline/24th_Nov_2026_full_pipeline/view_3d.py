"""3D viewer for the cross-date matching: one plant on all its dates, leaves
coloured by leaf ID (IDs from 29.06.), stem black, small untracked leaves grey.

    # from this folder (24th_Nov_2026_full_pipeline/)
    ../../venv/bin/python view_3d.py --list
    ../../venv/bin/python view_3d.py 16/03
    ../../venv/bin/python view_3d.py 16/03 --overlay
    ../../venv/bin/python view_3d.py 16/03 --save plant.png
    ../../venv/bin/python view_3d.py 16/03 --run outputs/run_...   # another run (default: newest)

The plant is a 29.06. plant: '16/03', 'group16_plant03' or
'plant_cluster_16_plant_instance_03'. Side-by-side views are linked: turning
one turns all. --overlay puts all dates in one view, coloured by date (shows
how well they line up). Mouse: left = rotate, right / wheel = zoom,
shift + left = pan, r = reset, q = quit.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import names  # noqa: E402
from pipeline.config import OUTPUT_ROOT, RUN_LEVEL_DIRS  # noqa: E402
from pipeline.cross_date.config import DATE_COLOURS, date_label  # noqa: E402
from pipeline.align import Alignment, load_alignments  # noqa: E402
from pipeline.config import PROJECT_ORIGIN  # noqa: E402

LEAF_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b5cd6", "#3fb8c9", "#a8641f",
                "#5a8f29", "#c2410c", "#6b7280", "#b8336a"]
STEM, SMALL, OTHER = "#111111", "#9a9a92", "#dcdbd4"


def leaf_colour(leaf_id: str) -> str:
    return LEAF_COLOURS[(int(leaf_id[1:]) - 1) % len(LEAF_COLOURS)] if leaf_id.startswith("L") else SMALL


def to_raw(text: str) -> str:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if m:
        return f"group{int(m.group(1)):02d}_plant{int(m.group(2)):02d}"
    return names.raw(text.strip())


def matching_folder(run: Path | None) -> Path:
    """<run>/cross_date_matching of the given run (or of the newest run that has one)."""
    sub = RUN_LEVEL_DIRS["cross_date"]
    if run is not None:
        run = run if run.is_absolute() else Path.cwd() / run
        f = run if (run / "03_leaves").exists() else run / sub
        if not (f / "03_leaves" / "leaf_tracks.csv").exists():
            raise SystemExit(f"No cross-date matching in {run} - run run_matching.py --run {run}")
        return f
    outs = sorted(p / sub for p in OUTPUT_ROOT.glob("run_*") if (p / sub / "03_leaves" / "leaf_tracks.csv").exists())
    if not outs:
        raise SystemExit("No cross-date matching output yet - run run_matching.py first.")
    return outs[-1]


def load_plant_points(run_dir: Path, date: str, pid: str, al, fine: tuple, voxel: float):
    idx = run_dir / date / "03_plants" / "plants_index.csv"
    row = next((r for r in csv.DictReader(open(idx)) if r["plant_id"] == pid), None)
    if row is None:
        return None, None
    L = np.load(run_dir / date / "03_plants" / row["file"])
    xyz = L["xyz"] - np.asarray(PROJECT_ORIGIN)
    if voxel > 0:
        k = np.floor(xyz / voxel).astype(np.int64)
        _, i = np.unique(k, axis=0, return_index=True)
        i = np.sort(i)
        xyz, lab = xyz[i], L["labels"][i]
    else:
        lab = L["labels"]
    xyz = al.apply(xyz)
    xyz[:, 0] += fine[0]
    xyz[:, 1] += fine[1]
    return xyz, lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plant", nargs="?", help="29.06. plant, e.g. 16/03")
    ap.add_argument("--run", type=Path, default=None, help="run folder (default: newest run with a matching)")
    ap.add_argument("--list", action="store_true", help="list the plants with their leaf counts per date")
    ap.add_argument("--overlay", action="store_true", help="all dates in one view, coloured by date")
    ap.add_argument("--save", type=Path, default=None, help="write a screenshot instead of opening a window")
    ap.add_argument("--voxel", type=float, default=0.003, help="thin points to this voxel (m); 0 = all points")
    ap.add_argument("--point-size", type=float, default=3.0)
    a = ap.parse_args()

    out = matching_folder(a.run)
    summ = json.loads((out / "01_clusters" / "summary.json").read_text())
    run_dir, ref, dates = Path(summ["run"]), summ["reference_date"], summ["dates"]
    assign = json.loads((out / "02_plants_stems" / "assignment.json").read_text())
    tracks = list(csv.DictReader(open(out / "03_leaves" / "leaf_tracks.csv")))
    shifts = {(r["date"], r["plant_id"]): (float(r["fine_dx_m"]), float(r["fine_dy_m"]))
              for r in csv.DictReader(open(out / "02_plants_stems" / "plant_shifts.csv"))} \
        if (out / "02_plants_stems" / "plant_shifts.csv").exists() else {}

    if a.list or not a.plant:
        refs = sorted({r["plant"] for r in tracks})
        print(f"29.06. plants in {out.parent.name} (tracked leaves per date):")
        for p in refs:
            per = {d: sum(1 for r in tracks if r["plant"] == p and r["date"] == d and r["leaf_id"].startswith("L"))
                   for d in dates}
            short = p.replace("plant_cluster_", "").replace("_plant_instance_", "/")
            print(f"  {short:7s} {p:38s} " + "  ".join(f"{date_label(d)} {n:2d}" for d, n in per.items() if n))
        return

    rp = to_raw(a.plant)
    nice = names.nice(rp)
    if not any(r["plant"] == nice for r in tracks):
        raise SystemExit(f"{nice} has no leaves in {out.parent.name}; see --list")
    als = load_alignments(run_dir / "alignment" / "alignment.json")
    als.setdefault(ref, Alignment(ref, ref, [0.0, 0.0], [0.0, 1.0]))

    views = []
    for d in dates:
        qp = rp if d == ref else assign.get(d, {}).get(rp)
        if not qp:
            continue
        xyz, lab = load_plant_points(run_dir, d, qp, als[d], shifts.get((d, qp), (0.0, 0.0)), a.voxel)
        if xyz is None:
            continue
        rows = [r for r in tracks if r["plant"] == nice and r["date"] == d]
        piece_to_id = {}
        for r in rows:
            for pc in r["pieces"].split():
                piece_to_id[int(pc)] = r["leaf_id"]
        views.append((d, qp, xyz, lab, piece_to_id))
    if not views:
        raise SystemExit(f"no plant data for {nice}")

    import pyvista as pv
    from matplotlib.colors import to_rgb

    # common origin: the 29.06. stem base (or the first date's), ground = 0
    base_v = next((v for v in views if v[0] == ref), views[0])
    stem = base_v[2][base_v[3] == 0]
    o = np.r_[np.median(stem[:, :2], axis=0) if len(stem) else base_v[2][:, :2].mean(0),
              np.percentile(base_v[2][:, 2], 0.5)]

    off = a.save is not None
    if a.overlay:
        pl = pv.Plotter(off_screen=off, window_size=(1100, 900))
        for d, qp, xyz, lab, _ in views:
            pl.add_mesh(pv.PolyData(100 * (xyz - o)), color=DATE_COLOURS[d], point_size=a.point_size,
                        render_points_as_spheres=True, label=date_label(d))
        pl.add_legend(bcolor="white", face="circle", size=(0.16, 0.2))
        pl.add_text(f"{nice}: all dates, aligned (cm)", font_size=10)
        pl.show_grid(xtitle="x (cm)", ytitle="y (cm)", ztitle="height (cm)", font_size=8)
    else:
        pl = pv.Plotter(shape=(1, len(views)), off_screen=off, window_size=(430 * len(views), 760))
        for k, (d, qp, xyz, lab, p2id) in enumerate(views):
            pl.subplot(0, k)
            cols = np.empty((len(lab), 3))
            cols[:] = to_rgb(OTHER)
            cols[lab == 0] = to_rgb(STEM)
            ids = {}
            for pc, lid in p2id.items():
                m = lab == pc
                if m.any():
                    cols[m] = to_rgb(leaf_colour(lid))
                    if lid.startswith("L"):
                        ids.setdefault(lid, []).append(np.flatnonzero(m))
            P = pv.PolyData(100 * (xyz - o))
            P["rgb"] = (cols * 255).astype(np.uint8)
            pl.add_mesh(P, scalars="rgb", rgb=True, point_size=a.point_size, render_points_as_spheres=True)
            # leaf ID label at the leaf point farthest from the stem
            if ids:
                sx = np.median(xyz[lab == 0, :2], axis=0) if (lab == 0).any() else xyz[:, :2].mean(0)
                pts, txt = [], []
                for lid, parts in sorted(ids.items()):
                    i = np.concatenate(parts)
                    far = i[np.argmax(np.hypot(*(xyz[i, :2] - sx).T))]
                    pts.append(100 * (xyz[far] - o))
                    txt.append(lid)
                pl.add_point_labels(np.array(pts), txt, font_size=11, point_size=1, shape_opacity=0.75,
                                    always_visible=True, text_color="black")
            title = date_label(d) + (" (reference)" if d == ref else "") + ("" if qp == rp else f"\nas {names.nice(qp)}")
            pl.add_text(title, font_size=10, color=DATE_COLOURS[d])
            pl.show_grid(xtitle="x (cm)", ytitle="y (cm)", ztitle="height (cm)", font_size=7)
        pl.link_views()
        pl.subplot(0, 0)
        big = int(np.argmax([np.ptp(v[2], axis=0).max() for v in views]))
        pl.add_text(f"{nice}   black = stem, colours = leaf IDs, dark grey = small leaves, light grey = not counted",
                    position="lower_left", font_size=8)
    pl.set_background("white")
    if not a.overlay:
        pl.subplot(0, big)          # frame the linked camera on the largest date
    pl.camera_position = "xz"
    pl.camera.elevation = 20
    pl.camera.azimuth = -30
    pl.reset_camera()
    pl.camera.zoom(1.5 if not a.overlay else 1.1)
    if off:
        pl.screenshot(str(a.save))
        print(f"saved {a.save}")
    else:
        pl.show(title=f"{nice} - cross-date matching")


if __name__ == "__main__":
    main()
