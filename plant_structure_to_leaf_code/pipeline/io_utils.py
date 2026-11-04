"""Reading / writing point clouds with optional per-point labels.

Supported inputs: .las/.laz, .ply (ascii or binary, any scalar fields),
.asc/.txt/.xyz/.csv/.pts (CloudCompare style, header optional).
Coordinates are always returned as float64 in the file's own frame
(UTM for this project). PLY files written here store coordinates relative
to an origin that is saved in the header comment and restored on read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PROJECT_ORIGIN

POINT_SUFFIXES = {".las", ".laz", ".ply", ".asc", ".txt", ".xyz", ".csv", ".pts"}

# Column / field names that are understood as a stem-leaf label (lower case).
LABEL_NAMES = (
    "label", "labels", "class", "classification", "scalar_classification",
    "leaf_id", "organ", "organ_id", "scalar_label", "segment", "scalar_segment",
)


@dataclass
class Cloud:
    xyz: np.ndarray                      # (N, 3) float64
    labels: np.ndarray | None = None     # (N,) int32 or None
    rgb: np.ndarray | None = None        # (N, 3) float 0..1 or None
    label_field: str | None = None       # which field the labels came from
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# LAS
# ----------------------------------------------------------------------------

def _read_las(path: Path) -> Cloud:
    import laspy

    las = laspy.read(str(path))
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    names = {n.lower(): n for n in las.point_format.dimension_names}
    labels = None
    label_field = None
    for cand in LABEL_NAMES:
        if cand in names:
            vals = np.asarray(las[names[cand]])
            if cand == "classification" and len(np.unique(vals)) <= 1:
                continue  # unused LAS classification (all 0) is not a label
            labels = vals.astype(np.int32)
            label_field = names[cand]
            break
    rgb = None
    if {"red", "green", "blue"} <= set(names):
        rgb = np.column_stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)]).astype(np.float64)
        mx = rgb.max() if len(rgb) else 0
        rgb = rgb / (65535.0 if mx > 255 else 255.0) if mx > 0 else None
    return Cloud(xyz, labels, rgb, label_field)


# ----------------------------------------------------------------------------
# PLY (own parser so scalar fields such as 'label' are kept)
# ----------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_ply(path: Path) -> Cloud:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Broken PLY header: {path}")
            s = line.decode("ascii", errors="ignore").strip()
            header_lines.append(s)
            if s == "end_header":
                break
        fmt = "ascii"
        props: list[tuple[str, str]] = []
        n_vertex = 0
        in_vertex = False
        origin = np.zeros(3)
        for s in header_lines:
            parts = s.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "comment" and len(parts) >= 5 and parts[1] == "origin":
                origin = np.array([float(v) for v in parts[2:5]])
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertex = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("PLY vertex list properties are not supported")
                props.append((parts[2], _PLY_TYPES[parts[1]]))
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_vertex, ndmin=2)
            arr = {name: data[:, i] for i, (name, _) in enumerate(props)}
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(n, endian + t) for n, t in props])
            raw = np.frombuffer(f.read(dt.itemsize * n_vertex), dtype=dt, count=n_vertex)
            arr = {n: raw[n] for n, _ in props}
    lower = {k.lower(): k for k in arr}
    xyz = np.column_stack([arr[lower["x"]], arr[lower["y"]], arr[lower["z"]]]).astype(np.float64) + origin
    labels, label_field = None, None
    for cand in LABEL_NAMES:
        if cand in lower:
            labels = np.round(np.asarray(arr[lower[cand]], dtype=np.float64)).astype(np.int32)
            label_field = lower[cand]
            break
    rgb = None
    for r, g, b in (("red", "green", "blue"), ("r", "g", "b")):
        if r in lower and g in lower and b in lower:
            rgb = np.column_stack([arr[lower[r]], arr[lower[g]], arr[lower[b]]]).astype(np.float64)
            rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
            break
    return Cloud(xyz, labels, rgb, label_field)


# ----------------------------------------------------------------------------
# ASCII tables (CloudCompare ASC export, CSV, TXT)
# ----------------------------------------------------------------------------

def _read_ascii(path: Path) -> Cloud:
    import pandas as pd

    header: list[str] | None = None
    skip = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                skip += 1
                continue
            first = re.split(r"[\s,;]+", s.lstrip("/#").strip())
            try:
                float(first[0])
                break                       # first numeric row reached
            except ValueError:
                header = [h.strip().lower() for h in first if h.strip()]
                skip += 1
    df = pd.read_csv(path, sep=r"[\s,;]+", engine="python", header=None, skiprows=skip, comment=None)
    df = df.dropna(axis=1, how="all")
    data = df.to_numpy(dtype=np.float64)
    if data.shape[1] < 3:
        raise ValueError(f"Need at least X Y Z columns: {path}")
    cols = header if header and len(header) == data.shape[1] else None
    if cols:
        ix = cols.index("x") if "x" in cols else 0
        iy = cols.index("y") if "y" in cols else 1
        iz = cols.index("z") if "z" in cols else 2
    else:
        ix, iy, iz = 0, 1, 2
    xyz = data[:, [ix, iy, iz]]
    labels, label_field = None, None
    if cols:
        for cand in LABEL_NAMES:
            if cand in cols:
                labels = np.round(data[:, cols.index(cand)]).astype(np.int32)
                label_field = cand
                break
    rgb = None
    if cols and {"r", "g", "b"} <= set(cols):
        rgb = data[:, [cols.index("r"), cols.index("g"), cols.index("b")]]
        rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
    return Cloud(xyz, labels, rgb, label_field, extra={"columns": cols})


def read_cloud(path: str | Path) -> Cloud:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in {".las", ".laz"}:
        return _read_las(path)
    if suf == ".ply":
        return _read_ply(path)
    if suf in {".asc", ".txt", ".xyz", ".csv", ".pts"}:
        return _read_ascii(path)
    raise ValueError(f"Unsupported point cloud format '{suf}': {path}")


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------

def write_ply(
    path: str | Path, xyz: np.ndarray, *, labels: np.ndarray | None = None,
    rgb: np.ndarray | None = None, origin=PROJECT_ORIGIN,
) -> Path:
    """Binary PLY, xyz relative to `origin` (float32), origin stored in the header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if labels is not None:
        fields += [("label", "<i4")]
    rows = np.empty(n, dtype=np.dtype(fields))
    rel = np.asarray(xyz, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    rows["x"], rows["y"], rows["z"] = rel[:, 0], rel[:, 1], rel[:, 2]
    if rgb is not None:
        c = np.asarray(rgb, dtype=np.float64)
        c = np.clip(c * 255.0 if c.max() <= 1.0 else c, 0, 255).astype(np.uint8)
        rows["red"], rows["green"], rows["blue"] = c[:, 0], c[:, 1], c[:, 2]
    if labels is not None:
        rows["label"] = np.asarray(labels, dtype=np.int32)
    type_name = {"<f4": "float", "u1": "uchar", "<i4": "int"}
    header = ["ply", "format binary_little_endian 1.0",
              f"comment origin {origin[0]:.3f} {origin[1]:.3f} {origin[2]:.3f}",
              f"element vertex {n}"]
    header += [f"property {type_name[t]} {name}" for name, t in fields]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(rows.tobytes())
    return path


def voxel_keys(xyz: np.ndarray, voxel: float, ref=None) -> np.ndarray:
    """Integer voxel index per point packed into one int64 (for set operations).

    Coordinates are taken relative to `ref` (default: the cloud's own minimum).
    Pass the SAME ref for clouds whose voxels are compared with each other.
    21 bits per axis cover about +-5 km at a 5 mm voxel.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if ref is None:
        ref = np.floor(xyz.min(axis=0)) if len(xyz) else np.zeros(3)
    rel = xyz - np.asarray(ref, dtype=np.float64)
    ijk = np.floor(rel / voxel).astype(np.int64) + (1 << 20)
    ijk = np.clip(ijk, 0, (1 << 21) - 1)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
