"""
====================================================================
 STEP 2 — Remove the soil:  CSF -> DTM -> kNN -> HAG
====================================================================

 IN   data/2_downsampled/DRD0N-<date>.ply
 OUT  data/3_ground_removed/G-DRD0N-<date>.ply     plant points only

 WHY THIS COMES BEFORE THE CLASSIFIER
   Soil and stem bases are both low, compact and roughly vertical, so
   while soil is still in the cloud it competes directly with the stem
   class. Measured on this project: removing the ground first raised
   stem IoU from 0.76 to 0.86. It also means the classifier only has
   two classes to learn instead of three.

 THE FOUR STEPS  (the same chain the soil-segmentation team used,
 running the real cloth-simulation-filter library, not an imitation)

   1. CSF — Cloth Simulation Filter
      The cloud is flipped upside down and a stiff virtual cloth is
      dropped on it. Because the cloth is rigid it drapes over the
      terrain and bridges across plants rather than sagging between
      them. Points close to where it settles are ground.

   2. DTM — Digital Terrain Model
      Those ground points are gridded into 10 cm cells, each cell
      keeping the MEDIAN height inside it. Median, not mean, so one
      stray low point cannot drag a whole cell down.

   3. kNN gap filling
      Cells the cloth never reached (fully hidden under canopy) take
      an inverse-distance weighted height from the nearest filled
      cells, so the terrain model covers the whole plot.

   4. HAG — Height Above Ground
          hag = z - DTM(x, y)
      Points below the threshold are soil and are dropped.

 WHY NOT JUST THRESHOLD THE 'height' FIELD ALREADY IN THE FILE
   On this data that field behaves as if it came from a near-flat
   terrain model: on DRD05 it correlates with raw z at 1.00, i.e. it
   is essentially z minus a constant. The field is tilled and uneven,
   so soil clods sit up to ~17 cm above that flat reference and would
   survive a 2-3 cm cut. Rebuilding the DTM locally at 10 cm follows
   the real ground surface and removes them.

 RUN   python pipeline/step2_remove_ground.py
====================================================================
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rfpipe as R

# --- cloth ---
CLOTH_RES     = 0.05   # m, cloth grid spacing
RIGIDNESS     = 3      # 1 steep, 2 relief, 3 flat. Higher = stiffer = less sag
TIME_STEP     = 0.65
CLASS_THRES   = 0.03   # m, distance from the cloth still counted as ground
ITERATIONS    = 500
# --- terrain model and cut ---
DTM_CELL      = 0.10   # m
HAG_THRESHOLD = 0.03   # m, below this is soil
SAVE_PNG      = False  # review plots are generated after instance segmentation


def run_csf(pts):
    import CSF
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = CLOTH_RES
    csf.params.rigidness = RIGIDNESS
    csf.params.time_step = TIME_STEP
    csf.params.class_threshold = CLASS_THRES
    csf.params.interations = ITERATIONS      # the library's own spelling
    csf.setPointCloud(pts.tolist())
    g, ng = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(g, ng)
    mask = np.zeros(len(pts), bool)
    mask[np.asarray(g, dtype=np.int64)] = True
    return mask


def build_dtm(gpts, x0, y0, nx, ny, cell):
    gx = np.clip(((gpts[:, 0] - x0) / cell).astype(np.int64), 0, nx - 1)
    gy = np.clip(((gpts[:, 1] - y0) / cell).astype(np.int64), 0, ny - 1)
    flat = gx * ny + gy
    order = np.argsort(flat)
    fs, zs = flat[order], gpts[order, 2]
    uniq, starts = np.unique(fs, return_index=True)
    bounds = np.append(starts, len(fs))
    dtm = np.full(nx * ny, np.nan)
    for i, cid in enumerate(uniq):
        dtm[cid] = np.median(zs[bounds[i]:bounds[i + 1]])
    return dtm.reshape(nx, ny)


def fill_gaps(dtm, k=8):
    from scipy.spatial import cKDTree
    nx, ny = dtm.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    filled = ~np.isnan(dtm)
    if filled.sum() == 0:
        raise RuntimeError("CSF found no ground at all")
    if filled.all():
        return dtm
    src = np.column_stack([ii[filled], jj[filled]]).astype(float)
    tgt = np.column_stack([ii[~filled], jj[~filled]]).astype(float)
    kk = min(k, len(src))
    d, idx = cKDTree(src).query(tgt, k=kk, workers=-1)
    if kk == 1:
        d, idx = d[:, None], idx[:, None]
    w = 1.0 / np.maximum(d, 1e-6)
    out = dtm.copy()
    out[~filled] = (dtm[filled][idx] * w).sum(axis=1) / w.sum(axis=1)
    return out


def process(src):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    out = R.GDIR / f"G-{src.stem}.ply"
    print(f"\n[{src.name}]")
    pts, _ = R.load_ply_xyz_hag(src)
    print(f"  {len(pts):,} points")

    print(f"  1. CSF  (res {CLOTH_RES} m, rigidness {RIGIDNESS})")
    gmask = run_csf(pts)
    print(f"     ground seeds {int(gmask.sum()):,}  ({100*gmask.mean():.1f}%)")

    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    nx = int(np.ceil((pts[:, 0].max() - x0) / DTM_CELL)) + 1
    ny = int(np.ceil((pts[:, 1].max() - y0) / DTM_CELL)) + 1
    print(f"  2. DTM  {nx} x {ny} cells @ {DTM_CELL*100:.0f} cm")
    dtm = build_dtm(pts[gmask], x0, y0, nx, ny, DTM_CELL)
    print(f"     empty cells {int(np.isnan(dtm).sum()):,}")
    print("  3. kNN gap filling")
    dtm = fill_gaps(dtm)

    print("  4. HAG")
    px = np.clip(((pts[:, 0] - x0) / DTM_CELL).astype(np.int64), 0, nx - 1)
    py = np.clip(((pts[:, 1] - y0) / DTM_CELL).astype(np.int64), 0, ny - 1)
    hag = pts[:, 2] - dtm[px, py]
    keep = hag > HAG_THRESHOLD
    print(f"     soil removed {int((~keep).sum()):,} ({100*(~keep).mean():.1f}%)   "
          f"plant kept {int(keep.sum()):,}")

    R.write_ply_height(out, pts[keep], hag[keep])
    print(f"  -> {out.name}")

    if SAVE_PNG:
        fig, ax = plt.subplots(figsize=(16, 4.5))
        ax.scatter(pts[~keep, 1], pts[~keep, 2], s=0.3, c="#B8935A", linewidths=0,
                   label=f"removed soil ({int((~keep).sum()):,})")
        ax.scatter(pts[keep, 1], pts[keep, 2], s=0.3, c="#2E7D46", linewidths=0,
                   label=f"kept plant ({int(keep.sum()):,})")
        ax.set_title(f"{src.stem} — CSF + DTM + kNN + HAG")
        ax.set_xlabel("Y [m]"); ax.set_ylabel("Z [m]"); ax.set_aspect("equal")
        ax.legend(markerscale=16, fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(R.GDIR / f"G-{src.stem}__removal.png", dpi=140)
        plt.close(fig)

    print(f"  [{time.time()-t0:.1f}s]")
    return {"file": src.name, "points": len(pts),
            "ground_seeds": int(gmask.sum()),
            "soil_removed": int((~keep).sum()),
            "plant_kept": int(keep.sum()),
            "pct_removed": round(100 * float((~keep).mean()), 1)}


def main():
    R.GDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print(" GROUND REMOVAL — CSF -> DTM -> kNN -> HAG")
    print("=" * 66)
    rows = []
    for src in sorted(R.DOWN_DIR.glob("*.ply")):
        out = R.GDIR / f"G-{src.stem}.ply"
        if R.ply_is_complete(out):
            print(f"\n[{src.name}] already done")
            continue
        rows.append(process(src))
    if rows:
        p = R.GDIR / "ground_removal_summary.csv"
        old = list(csv.DictReader(open(p))) if p.exists() else []
        done = {r["file"] for r in rows}
        allr = [r for r in old if r["file"] not in done] + rows
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
            for r in allr:
                w.writerow({k: r.get(k, "") for k in rows[0]})
    print(f"\nDone -> {R.GDIR}")


if __name__ == "__main__":
    main()
