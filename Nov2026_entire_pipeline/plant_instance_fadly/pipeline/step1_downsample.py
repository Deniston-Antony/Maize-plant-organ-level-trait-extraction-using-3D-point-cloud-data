"""
====================================================================
 STEP 1 — Downsample the raw scans
====================================================================

 IN   data/1_raw_las/*.las          22-37 million points each
 OUT  data/2_downsampled/DRD0N-<date>.ply

 WHY
   The raw scans are far too heavy to open in CloudCompare for hand
   labelling, and far too heavy to compute geometric features on.

 HOW — voxel downsampling
   Space is divided into 4 mm cubes and only the first point found in
   each cube is kept, so the surviving points are roughly evenly
   spaced instead of clumped wherever the scanner happened to dwell.

 MEMORY
   A 1.7 GB file will not fit in memory here, so the las is streamed
   in 4-million-point chunks. Within a chunk numpy dedupes fast; a
   running set of already-used voxel keys handles the boundaries
   between chunks. Keys are packed into a single int64 rather than
   python tuples, which is what makes the set lookups quick.

 WHAT IT KEEPS
   x, y, z, RGB when present, and the 'height' field from the las --
   that field is the CSF+HAG result from Task 1.1 and is worth
   carrying through.

 RUN   python pipeline/step1_downsample.py
====================================================================
"""

import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rfpipe as R

VOXEL = 0.004        # metres
CHUNK = 500_000      # smaller chunks for laptops; voxel result is unchanged


def date_of(name):
    m = re.search(r"(\d{6})", name)
    if not m:
        raise ValueError(f"no YYMMDD date in {name}")
    return m.group(1)


def write_ply(path, x, y, z, rgb=None, height=None):
    cols = [x, y, z]
    fmt = ["%.6f"] * 3
    head = ["ply", "format ascii 1.0", f"element vertex {len(x)}",
            "property double x", "property double y", "property double z"]
    if rgb is not None:
        cols += list(rgb); fmt += ["%d"] * 3
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    if height is not None:
        cols.append(height); fmt.append("%.4f")
        head.append("property float height")
    head.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode())
        np.savetxt(fh, np.column_stack(cols), fmt=" ".join(fmt))


def downsample(las_path, out_path):
    import laspy
    t0 = time.time()
    print(f"\n[{las_path.name}]")
    seen = set()
    ox, oy, oz, orr, og, ob, oh = [], [], [], [], [], [], []

    with laspy.open(las_path) as fh:
        total = fh.header.point_count
        dims = fh.header.point_format.dimension_names
        has_rgb, has_h = "red" in dims, "height" in dims
        print(f"  {total:,} points   rgb={has_rgb}  height={has_h}")
        # A fixed LAS-header origin keeps voxel keys stable across chunks.
        x0, y0, z0 = (float(v) for v in fh.header.mins)
        seen_n = 0
        for pts in fh.chunk_iterator(CHUNK):
            x = np.asarray(pts.x); y = np.asarray(pts.y); z = np.asarray(pts.z)
            if has_rgb:
                r = (np.asarray(pts.red) / 256).astype(np.uint8)
                g = (np.asarray(pts.green) / 256).astype(np.uint8)
                b = (np.asarray(pts.blue) / 256).astype(np.uint8)
            if has_h:
                h = np.asarray(pts["height"], dtype=np.float64)
            vx = np.floor((x - x0) / VOXEL).astype(np.int64)
            vy = np.floor((y - y0) / VOXEL).astype(np.int64)
            vz = np.floor((z - z0) / VOXEL).astype(np.int64)
            if (min(vx.min(), vy.min(), vz.min()) < 0 or
                    max(vx.max(), vy.max(), vz.max()) >= 1 << 21):
                raise ValueError("LAS extent is outside the supported 4 mm voxel-key range")
            key = (vx << 42) | (vy << 21) | vz

            uk, first = np.unique(key, return_index=True)
            x, y, z = x[first], y[first], z[first]
            if has_rgb: r, g, b = r[first], g[first], b[first]
            if has_h:   h = h[first]

            already = np.fromiter((k in seen for k in uk.tolist()),
                                  dtype=bool, count=len(uk))
            new = ~already
            seen.update(uk[new].tolist())

            ox.append(x[new]); oy.append(y[new]); oz.append(z[new])
            if has_rgb:
                orr.append(r[new]); og.append(g[new]); ob.append(b[new])
            if has_h:
                oh.append(h[new])
            seen_n += len(pts)
            print(f"  {seen_n:,}/{total:,}   kept {sum(len(a) for a in ox):,}", end="\r")

    x = np.concatenate(ox); y = np.concatenate(oy); z = np.concatenate(oz)
    rgb = (np.concatenate(orr), np.concatenate(og), np.concatenate(ob)) if orr else None
    height = np.concatenate(oh) if oh else None
    print(f"\n  kept {len(x):,}  ({total/max(1,len(x)):.1f}x reduction)")
    write_ply(out_path, x, y, z, rgb, height)
    print(f"  -> {out_path.name}   [{time.time()-t0:.1f}s]")


def main():
    R.DOWN_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(R.RAW_DIR.glob("*.las"), key=lambda p: date_of(p.name))
    if not files:
        raise SystemExit(f"no .las found in {R.RAW_DIR}")
    print(f"{len(files)} raw scans, in date order")
    for i, f in enumerate(files, 1):
        out = R.DOWN_DIR / f"DRD{i:02d}-{date_of(f.name)}.ply"
        if R.ply_is_complete(out):
            print(f"\n[{f.name}] already done")
            continue
        downsample(f, out)
    print(f"\nDone -> {R.DOWN_DIR}")


if __name__ == "__main__":
    main()
