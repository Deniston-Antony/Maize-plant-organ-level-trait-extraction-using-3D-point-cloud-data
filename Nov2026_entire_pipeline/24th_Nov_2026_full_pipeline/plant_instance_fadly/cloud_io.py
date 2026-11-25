"""Small PLY reader/writer for the workspace; no optional plyfile dependency."""
from pathlib import Path
import numpy as np

PLY_TYPES = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
             "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
             "float": "f4", "double": "f8", "float32": "f4", "float64": "f8"}


def read_ply(path):
    path = Path(path)
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        props, count, fmt, vertex = [], None, None, False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            tokens = line.decode("ascii").strip().split()
            if not tokens:
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                vertex = tokens[1] == "vertex"
                if vertex:
                    count = int(tokens[2])
            elif tokens[0] == "property" and vertex:
                if tokens[1] == "list":
                    raise ValueError("List-valued vertex properties are unsupported")
                props.append((tokens[2], PLY_TYPES[tokens[1]]))
            elif tokens[0] == "end_header":
                break
        offset = stream.tell()
    if count is None:
        raise ValueError("Missing vertex count")
    if fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(n, endian + t) for n, t in props])
        if path.stat().st_size < offset + count * dtype.itemsize:
            raise ValueError(f"Truncated binary vertices: {path}")
        return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))
    if fmt == "ascii":
        with path.open("rb") as stream:
            stream.seek(offset)
            values = np.loadtxt(stream, max_rows=count, ndmin=2)
        if values.shape != (count, len(props)):
            raise ValueError(f"ASCII vertex count mismatch: {path}")
        result = np.empty(count, dtype=props)
        for j, (name, _) in enumerate(props):
            result[name] = values[:, j]
        return result
    raise ValueError(f"Unsupported PLY encoding: {fmt}")


def xyz_array(vertices, dtype=np.float32):
    return np.column_stack([vertices[c] for c in ("x", "y", "z")]).astype(dtype, copy=False)


def write_ply(path, xyz, colors=None, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        props += [(c, "u1") for c in ("red", "green", "blue")]
        header += [f"property uchar {c}" for c in ("red", "green", "blue")]
    fields = fields or {}
    for name, values in fields.items():
        if len(values) != len(xyz):
            raise ValueError(f"Wrong length for field {name}")
        integer = np.issubdtype(np.asarray(values).dtype, np.integer)
        props.append((name, "<i4" if integer else "<f4"))
        header.append(f"property {'int' if integer else 'float'} {name}")
    with path.open("wb") as stream:
        stream.write(("\n".join(header + ["end_header"]) + "\n").encode("ascii"))
        for start in range(0, len(xyz), 500_000):
            stop = min(start + 500_000, len(xyz))
            block = np.empty(stop - start, dtype=props)
            for axis, c in enumerate(("x", "y", "z")):
                block[c] = xyz[start:stop, axis]
            if colors is not None:
                for axis, c in enumerate(("red", "green", "blue")):
                    block[c] = colors[start:stop, axis]
            for name, values in fields.items():
                block[name] = values[start:stop]
            block.tofile(stream)


def instance_colors(labels):
    labels = np.asarray(labels)
    rng = np.random.default_rng(42)
    palette = rng.integers(40, 245, (max(int(labels.max(initial=0)), 1) + 1, 3), dtype=np.uint8)
    palette[0] = 100
    return palette[np.maximum(labels, 0)]
