"""
====================================================================
 rfpipe.py — the one shared library for this project
====================================================================

 Everything the four step-scripts and the notebook need lives here,
 so a change to (say) how a feature is computed changes everywhere at
 once and the steps can never disagree with each other.

 CONTENTS
   paths and constants        where the data lives, class colours
   ply read / write           point clouds in and out
   feature computation        the 14 geometric features, multi-scale
   caching                    so nothing is ever computed twice
   smoothing                  spatial post-processing
   metrics                    IoU / mIoU

 CLASS COLOURS  (used in every output .ply, open them in CloudCompare)
   stem    brown   (150, 75, 40)
   leaf    green   (60, 150, 45)
   ground  purple  (128, 0, 128)     only in the ground-removal step
   unknown black   (0, 0, 0)         model was not confident enough
====================================================================
"""

from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "1_raw_las"
DOWN_DIR  = ROOT / "data" / "2_downsampled"
GDIR      = ROOT / "data" / "3_ground_removed"
LABEL_DIR = ROOT / "data" / "4_labels"
RESULTS   = ROOT / "results"
CACHE     = RESULTS / "_feature_cache"
MODEL_FILE = RESULTS / "model" / "et_model.joblib"

# growth stage -> ground-removed cloud
STAGES = {
    "P1": "G-DRD01-230601.ply",   # seedlings, leaf labels only
    "P2": "G-DRD02-230621.ply",
    "P3": "G-DRD03-230629.ply",
    "P4": "G-DRD04-230705.ply",
    "P5": "G-DRD05-230711.ply",
}

# ─────────────────────────────────────────────────────────────────
#  CLASSES
# ─────────────────────────────────────────────────────────────────
CLASS_STEM, CLASS_LEAF, CLASS_GROUND = 0, 1, 2
CLASS_UNKNOWN = -1
CLASS_NAMES = {CLASS_STEM: "stem", CLASS_LEAF: "leaf", CLASS_GROUND: "ground"}
COLOURS = {
    CLASS_STEM:    (150, 75, 40),
    CLASS_LEAF:    (60, 150, 45),
    CLASS_GROUND:  (128, 0, 128),
    CLASS_UNKNOWN: (0, 0, 0),
}

# 14 features, computed at each neighbourhood size
FEAT_PER_SCALE = ["linearity", "planarity", "sphericity", "curvature",
                  "verticality", "anisotropy", "eigenentropy", "omnivariance",
                  "sum_eigen", "z_range", "z_std", "hag", "radial", "density"]

SCALES = (20, 50, 100)     # chosen by the search — see README


def feature_names(ks=SCALES):
    return [f"{n}_k{k}" for k in ks for n in FEAT_PER_SCALE]


# ─────────────────────────────────────────────────────────────────
#  PLY IN / OUT
# ─────────────────────────────────────────────────────────────────
def load_ply_xyz(path):
    return _read_ascii_ply(path)[0]


def load_ply_xyz_hag(path):
    """xyz plus the stored height-above-ground, when the file has it."""
    pts, height = _read_ascii_ply(path)
    hag = height if height is not None else pts[:, 2] - pts[:, 2].min()
    return pts, hag


def _read_ascii_ply(path):
    """Read the ASCII PLYs written by this package's downsampling stages."""
    properties = []
    header_lines = 0
    vertex_count = None
    in_vertex = False
    with open(path, "rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}")
            header_lines += 1
            line = raw.decode("ascii").strip()
            if line.startswith("format ") and line != "format ascii 1.0":
                raise ValueError(f"Expected ASCII PLY: {path}")
            if line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
    if vertex_count is None or not all(name in properties for name in ("x", "y", "z")):
        raise ValueError(f"PLY has no XYZ vertex properties: {path}")
    columns = [properties.index(name) for name in ("x", "y", "z")]
    has_height = "height" in properties
    if has_height:
        columns.append(properties.index("height"))
    values = np.loadtxt(path, skiprows=header_lines, usecols=columns,
                        dtype=np.float64, ndmin=2)
    if len(values) != vertex_count:
        raise ValueError(f"PLY point count changed: {path}")
    return values[:, :3], values[:, 3] if has_height else None


def write_ply(path, pts, rgb):
    """
    Colour ply, written with np.savetxt.
    A python loop over 1.7 million points takes minutes; this takes
    about a second.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(pts)}\n"
              "property double x\nproperty double y\nproperty double z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        np.savetxt(fh, np.column_stack([np.asarray(pts, np.float64),
                                        np.asarray(rgb, np.int32)]),
                   fmt="%.6f %.6f %.6f %d %d %d")


def write_ply_height(path, pts, hag):
    """ply carrying a height scalar instead of colour (ground-removal output)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(pts)}\n"
              "property double x\nproperty double y\nproperty double z\n"
              "property float height\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        np.savetxt(fh, np.column_stack([pts, hag]), fmt="%.6f %.6f %.6f %.4f")


def labels_to_rgb(labels):
    rgb = np.zeros((len(labels), 3), dtype=np.uint8)
    for cls, col in COLOURS.items():
        rgb[labels == cls] = col
    return rgb


def ply_is_complete(path):
    """True only if the body has as many rows as the header promises.
    Checking that a file merely exists would treat a half-written file
    as finished and skip it forever."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        declared = None
        with open(path) as fh:
            for line in fh:
                if line.startswith("element vertex"):
                    declared = int(line.split()[2])
                if line.strip() == "end_header":
                    break
            return declared is not None and sum(1 for _ in fh) == declared
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
#  FEATURES
# ─────────────────────────────────────────────────────────────────
def radial_distance(pts):
    """Horizontal distance from the local centre-line of the row.
    Stems sit near the middle of their plant, leaf tips reach out."""
    y = pts[:, 1]
    bands = np.floor((y - y.min()) / 0.10).astype(int)
    out = np.zeros(len(pts))
    for b in np.unique(bands):
        m = bands == b
        out[m] = np.abs(pts[m, 0] - np.median(pts[m, 0]))
    return out


def compute_scale(pts, k, chunk=20_000, verbose=False):
    """
    The 14 features at ONE neighbourhood size.

    For each point, its k nearest neighbours are found and a 3x3
    covariance matrix of that little cloud is built. Its three
    eigenvalues L1>=L2>=L3 describe the local shape: one large
    eigenvalue means a line (stem), two large and one small means a
    flat sheet (leaf), three similar means a blob.

    Processed in chunks so peak memory stays bounded no matter how
    large the cloud is.
    """
    from scipy.spatial import cKDTree

    n = len(pts)
    tree = cKDTree(pts)
    k_eff = max(4, min(k, n - 1))
    o = {nm: np.empty(n) for nm in FEAT_PER_SCALE[:11]}

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, idx = tree.query(pts[s:e], k=k_eff, workers=-1)
        nb = pts[idx]

        z = nb[:, :, 2]
        o["z_range"][s:e] = z.max(axis=1) - z.min(axis=1)
        o["z_std"][s:e] = z.std(axis=1)

        c = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", c, c) / k_eff
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals[:, ::-1], 1e-12, None)
        nz = evecs[:, :, ::-1][:, 2, 2]          # z of the smallest-eigenvalue axis

        L1, L2, L3 = evals[:, 0], evals[:, 1], evals[:, 2]
        tot = L1 + L2 + L3
        o["linearity"][s:e]    = (L1 - L2) / L1
        o["planarity"][s:e]    = (L2 - L3) / L1
        o["sphericity"][s:e]   = L3 / L1
        o["curvature"][s:e]    = L3 / tot
        o["verticality"][s:e]  = 1.0 - np.abs(nz)
        o["anisotropy"][s:e]   = (L1 - L3) / L1
        nrm = evals / tot[:, None]
        o["eigenentropy"][s:e] = -(nrm * np.log(nrm)).sum(axis=1)
        o["omnivariance"][s:e] = np.cbrt(L1 * L2 * L3)
        o["sum_eigen"][s:e]    = tot
        del nb, c, cov, evals, evecs, idx
        if verbose:
            print(f"      k={k}  {e:,}/{n:,}", end="\r")

    o["hag"] = pts[:, 2] - pts[:, 2].min()
    o["radial"] = radial_distance(pts)

    d, _ = tree.query(pts[:min(n, 5000)], k=2, workers=-1)
    r = max(4.0 * float(np.median(d[:, 1])), 0.005)
    dens = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dens[s:e] = [len(c_) for c_ in tree.query_ball_point(pts[s:e], r=r, workers=-1)]
    o["density"] = dens

    for a in o.values():
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = 0.0
    return np.column_stack([o[nm] for nm in FEAT_PER_SCALE])


# ─────────────────────────────────────────────────────────────────
#  CACHES — geometry is expensive, labels are not
# ─────────────────────────────────────────────────────────────────
def cached_points(path):
    path = Path(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / f"{path.stem}__xyz.npy"
    if c.exists():
        return np.load(c)
    pts = load_ply_xyz(path)
    np.save(c, pts)
    return pts


def cached_scale(path, k, verbose=False):
    path = Path(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / f"{path.stem}__k{k}_v3.npy"
    if c.exists():
        return np.load(c)
    X = compute_scale(cached_points(path), k, verbose=verbose)
    np.save(c, X)
    return X


def cached_features(path, ks=SCALES, verbose=False):
    """Features at every scale, joined side by side (14 x len(ks) columns)."""
    return np.hstack([cached_scale(path, k, verbose) for k in ks])


def match_indices(cloud_pts, label_pts, tol=1e-3):
    """Which rows of the cloud correspond to a set of labelled points.
    Labels drawn in CloudCompare match to well under a millimetre."""
    from scipy.spatial import cKDTree
    d, i = cKDTree(cloud_pts).query(label_pts, k=1, workers=-1)
    ok = d < tol
    return np.unique(i[ok]), int(ok.sum()), int((~ok).sum())


def build_training_set(ks=SCALES, stages=tuple(STAGES), verbose=False):
    """
    Assemble (X, y, stage) from the hand labels.

    Features are computed on the FULL cloud and the labelled rows are
    then pulled out. Computing them on an extracted subset instead
    would give every point the wrong neighbours.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "-".join(map(str, ks)) + "__" + "".join(stages)
    c = CACHE / f"trainset__{tag}.npz"
    if c.exists():
        d = np.load(c, allow_pickle=True)
        return d["X"], d["y"], d["stage"]

    Xp, yp, sp = [], [], []
    for st in stages:
        f = GDIR / STAGES[st]
        if not f.exists():
            continue
        pts = cached_points(f)
        feats = cached_features(f, ks)
        for cls, nm in [(CLASS_STEM, "stem"), (CLASS_LEAF, "leaf")]:
            lp = LABEL_DIR / f"{st}-{nm}.ply"
            if not lp.exists():
                continue
            idx, _, _ = match_indices(pts, load_ply_xyz(lp))
            Xp.append(feats[idx])
            yp.append(np.full(len(idx), cls, dtype=int))
            sp.append(np.full(len(idx), st))
            if verbose:
                print(f"  {st} {nm:<5} {len(idx):>7,}")
    X = np.vstack(Xp); y = np.concatenate(yp); sg = np.concatenate(sp)
    np.savez_compressed(c, X=X, y=y, stage=sg)
    return X, y, sg


# ─────────────────────────────────────────────────────────────────
#  SMOOTHING
# ─────────────────────────────────────────────────────────────────
def smooth_proba(pts, proba, k=96, rounds=1, nidx=None):
    """
    Average class probabilities over each point's k nearest neighbours.

    The classifier judges every point alone, so its raw output contains
    physically impossible points -- a lone stem point inside a leaf
    blade. Averaging over the neighbourhood removes that speckle and
    leaves real boundaries alone, because at a true boundary the
    neighbourhood genuinely is mixed.

    Averaging PROBABILITIES beats voting on hard labels: an uncertain
    point contributes proportionally instead of casting a full vote.
    Measured: soft won at every k tried.
    """
    from scipy.spatial import cKDTree
    P = proba.copy()
    if nidx is not None:
        for _ in range(rounds):
            P = P[nidx[:, :k]].mean(axis=1)
        return P
    tree = cKDTree(pts)
    k_eff = min(k, len(pts))
    for _ in range(rounds):
        smoothed = np.empty_like(P)
        for start in range(0, len(pts), 20_000):
            stop = min(start + 20_000, len(pts))
            _, nearby = tree.query(pts[start:stop], k=k_eff, workers=2)
            if k_eff == 1:
                nearby = nearby[:, None]
            smoothed[start:stop] = P[nearby].mean(axis=1)
        P = smoothed
    return P


# ─────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────
def iou_scores(cm, class_ids):
    """
    IoU per class from a confusion matrix, and their mean (mIoU).

        IoU = TP / (TP + FP + FN)

    1.0 is a perfect overlap, 0 is none. Stricter than accuracy, which
    can look high just because one class dominates the point count.
    """
    out = {}
    for i, c in enumerate(class_ids):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        u = tp + fp + fn
        out[CLASS_NAMES[c]] = tp / u if u > 0 else float("nan")
    return out, float(np.nanmean(list(out.values())))


def load_model():
    import joblib
    return joblib.load(MODEL_FILE)


def predict_cloud(cloud_path, bundle=None, conf_threshold=None):
    """
    Full inference for one cloud: features -> forest -> smoothing ->
    confidence threshold. Returns (points, labels).
    """
    bundle = bundle or load_model()
    clf = bundle["model"]
    ids = bundle["class_ids"]
    C = bundle["config"]
    ks = bundle.get("ks", SCALES)
    thr = conf_threshold if conf_threshold is not None else C["conf_threshold"]

    pts = cached_points(cloud_path)
    X = cached_features(cloud_path, ks)
    P = smooth_proba(pts, clf.predict_proba(X),
                     k=C["smooth_k"], rounds=C["smooth_rounds"])
    lab = np.array([ids[i] for i in P.argmax(axis=1)])
    return pts, np.where(P.max(axis=1) >= thr, lab, CLASS_UNKNOWN)
