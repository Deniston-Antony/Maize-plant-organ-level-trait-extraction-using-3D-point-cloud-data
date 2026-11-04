"""Evaluate classified PLY from classify_plants.py against GT_manual.las.

Edit DATASET / USE_SIDEBAND below, then:
    .venv\\Scripts\\python.exe soil_segmentation/evaluate_classification.py

1. Opens Open3D viewer (errors in red).
2. After you close it, prints precision / recall / IoU / OA.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ============================================================
# USER SETTINGS — match what you used in classify_plants.py
# ============================================================

# Dataset folder name under data/  (all files share this stem)
DATASET = "230601pc_kinematic_calibration"

# Must match classify_plants.py (picks *_sideband.ply vs *_nosideband.ply)
USE_SIDEBAND = False

SHOW = True
CHUNK_SIZE = 500_000
MATCH_TOLERANCE = 1e-3
POINT_SIZE = 2.0

_TAG = "sideband" if USE_SIDEBAND else "nosideband"
INPUT_LAS = rf"data/{DATASET}/{DATASET}.las"
GT_LAS = rf"data/{DATASET}/{DATASET}_GT_manual.las"
CLASSIFIED_PLY = (
    rf"data/{DATASET}/output/soil_segmentation/"
    rf"{DATASET}_classified_all_{_TAG}.ply"
)

# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# Final_code/ (parent of soil_segmentation/) — where data/ and .venv live
PROJECT_ROOT = SCRIPT_DIR.parent

COLOR_SOIL = (140, 140, 140)
COLOR_MIXED = (255, 140, 0)
COLOR_PLANT = (13, 217, 26)
COLOR_INVALID = (242, 217, 26)
COLOR_ERROR = (1.0, 0.0, 0.0)

_PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        for base in (PROJECT_ROOT, SCRIPT_DIR):
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return (PROJECT_ROOT / p).resolve()
    return p.resolve()


def load_gt(gt_path: Path):
    las = laspy.read(str(gt_path))
    names = list(las.point_format.dimension_names)
    xyz = np.column_stack((
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ))

    if "gt_label" in names:
        labels = (np.asarray(las.gt_label) >= 0.5).astype(np.uint8)
    elif "ids" in names:
        # Manual GT export used ids: 1=ground, 2=plant
        labels = (np.asarray(las.ids) >= 2).astype(np.uint8)
    else:
        raise ValueError(
            f"No GT labels in {gt_path.name}. Expected 'gt_label' or 'ids'. Found: {names}"
        )
    return xyz, labels


def ply_vertex_count(ply_path: Path) -> int:
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"Cannot read vertex count from {ply_path}")


def rgb_to_pred(r, g, b):
    rgb = np.column_stack((r, g, b))
    soil = np.all(rgb == COLOR_SOIL, axis=1)
    plant = np.all(rgb == COLOR_PLANT, axis=1)
    pred = np.zeros(len(r), dtype=np.uint8)
    pred[plant] = 1
    return pred, soil | plant


def compute_metrics(cm: np.ndarray) -> dict:
    tn, fp, fn, tp = cm.ravel()
    total = int(cm.sum())
    oa = (tn + tp) / total if total else 0.0

    def prf1(tp_k, fp_k, fn_k):
        p = tp_k / (tp_k + fp_k) if (tp_k + fp_k) else 0.0
        r = tp_k / (tp_k + fn_k) if (tp_k + fn_k) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    plant_iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    g_p, g_r, g_f1 = prf1(tn, fn, fp)
    p_p, p_r, p_f1 = prf1(tp, fp, fn)
    return {
        "oa": oa, "cm": cm, "total": total, "plant_iou": plant_iou,
        "ground": {"precision": g_p, "recall": g_r, "f1": g_f1},
        "plant": {"precision": p_p, "recall": p_r, "f1": p_f1, "iou": plant_iou},
    }


def show_point_cloud(xyz, colors, title, n_errors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"\nOpening viewer — {len(pcd.points):,} points  ({n_errors:,} errors in red)")
    print("  Gray=ground  Green=plant  Orange=mixed  Yellow=invalid  Red=error vs GT")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.background_color = np.array([0.15, 0.15, 0.15])
    vis.run()
    vis.destroy_window()


def print_accuracy(m: dict, skipped: int, n_mixed_invalid: int, ply_name: str):
    cm = m["cm"]
    print(f"\n{'=' * 55}")
    print(f"  ACCURACY — {ply_name}")
    print(f"{'=' * 55}")
    print(f"  Overall accuracy : {m['oa'] * 100:.2f} %")
    print(f"  Plant IoU        : {m['plant_iou'] * 100:.2f} %")
    print(f"  Points evaluated : {m['total']:,}")
    if skipped:
        print(f"  Skipped (unknown colour): {skipped:,}")
    if n_mixed_invalid:
        print(f"  Excluded (mixed/invalid): {n_mixed_invalid:,}")
    print()
    print("  Confusion matrix (rows=GT, cols=pred):")
    print(f"                    pred ground   pred plant")
    print(f"    GT ground       {cm[0, 0]:>11,}  {cm[0, 1]:>11,}")
    print(f"    GT plant        {cm[1, 0]:>11,}  {cm[1, 1]:>11,}")
    print()
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
    g, p = m["ground"], m["plant"]
    print(f"  {'Ground':<8} {g['precision']*100:>9.2f}% {g['recall']*100:>9.2f}% {g['f1']*100:>9.2f}%")
    print(f"  {'Plant':<8} {p['precision']*100:>9.2f}% {p['recall']*100:>9.2f}% "
          f"{p['f1']*100:>9.2f}% {p['iou']*100:>9.2f}%")
    print(f"{'=' * 55}\n")


def main():
    las_path = resolve_path(INPUT_LAS)
    gt_path = resolve_path(GT_LAS)
    ply_path = resolve_path(CLASSIFIED_PLY)
    for p in (las_path, gt_path, ply_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print(f"\nClassified : {ply_path}")
    print(f"GT         : {gt_path}")
    print(f"LAS offset : {las_path.name}")

    xyz_gt, labels_gt = load_gt(gt_path)
    print(f"  GT points: {len(labels_gt):,}  "
          f"(ground={(labels_gt == 0).sum():,}  plant={(labels_gt == 1).sum():,})")

    with laspy.open(str(las_path)) as r:
        offset = np.array([r.header.x_min, r.header.y_min, r.header.z_min], dtype=np.float64)

    tree = cKDTree(xyz_gt)
    n_verts = ply_vertex_count(ply_path)
    xyz_all = np.empty((n_verts, 3), dtype=np.float32)
    col_all = np.empty((n_verts, 3), dtype=np.float32)
    cm = np.zeros((2, 2), dtype=np.int64)
    n_errors = skipped = n_mixed_invalid = processed = 0

    print(f"Matching PLY to GT ...")
    with open(ply_path, "rb") as f:
        while f.readline().decode("ascii").strip() != "end_header":
            pass
        while processed < n_verts:
            n = min(CHUNK_SIZE, n_verts - processed)
            chunk = np.fromfile(f, dtype=_PLY_DTYPE, count=n)
            xyz_chunk = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(np.float32)
            col_chunk = np.column_stack((chunk["r"], chunk["g"], chunk["b"])).astype(np.float32) / 255.0
            rgb = np.column_stack((chunk["r"], chunk["g"], chunk["b"]))
            mixed_invalid = (
                np.all(rgb == COLOR_MIXED, axis=1) | np.all(rgb == COLOR_INVALID, axis=1)
            )
            n_mixed_invalid += int(mixed_invalid.sum())

            xyz_abs = np.column_stack((
                chunk["x"].astype(np.float64) + offset[0],
                chunk["y"].astype(np.float64) + offset[1],
                chunk["z"].astype(np.float64) + offset[2],
            ))
            pred, valid = rgb_to_pred(chunk["r"], chunk["g"], chunk["b"])
            skipped += int((~valid & ~mixed_invalid).sum())

            if valid.any():
                dist, idx = tree.query(xyz_abs[valid], k=1, workers=-1)
                good = dist <= MATCH_TOLERANCE
                if good.any():
                    valid_idx = np.flatnonzero(valid)
                    matched_idx = valid_idx[good]
                    gt = labels_gt[idx[good]]
                    pr = pred[valid][good]
                    np.add.at(cm, (gt, pr), 1)
                    err = gt != pr
                    n_errors += int(err.sum())
                    col_chunk[matched_idx[err]] = COLOR_ERROR

            xyz_all[processed:processed + n] = xyz_chunk
            col_all[processed:processed + n] = col_chunk
            processed += n
            print(f"  {processed:,} / {n_verts:,}", end="\r", flush=True)

    print()
    metrics = compute_metrics(cm)
    if SHOW:
        show_point_cloud(xyz_all, col_all, f"{ply_path.name} | red = error", n_errors)
    print_accuracy(metrics, skipped, n_mixed_invalid, ply_path.name)


if __name__ == "__main__":
    main()
