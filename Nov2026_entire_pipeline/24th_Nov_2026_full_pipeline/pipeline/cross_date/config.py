"""Settings of the cross-date matching (clusters -> plants -> stems -> leaves).

Everything is geometry: density profiles, Gaussian fits, order-preserving
sequence alignment, DBSCAN (unsupervised) and translation-only ICP. No trained
model is used anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ClusterConfig:
    # ---- step 1: which points describe a cluster
    low_hag: tuple = (0.03, 0.15)     # low layer (m above ground): stem bases, same as alignment
    row_band: float = 0.10            # keep points within this distance of the row centre line (m)
    column_cell: float = 0.02         # column filter: 2 cm xy cells ...
    column_min_extent: float = 0.03   # ... kept only if their points span >= 3 cm in height (stems are
                                      # upright columns; low leaves lying in the layer are nearly flat)
    seedling_height: float = 0.15     # dates whose plants are mostly lower than this skip the column
                                      # filter (seedlings are too short to form columns; clod test instead)
    # ---- step 2: density profile along the row
    bin: float = 0.01                 # profile bin (m)
    smooth: float = 0.015             # Gaussian smoothing of the profile (m)
    thin_voxel: float = 0.005         # thin points to this voxel first, so all dates have the same density scale
    bg_k: float = 3.0                 # threshold = background mean + bg_k * background std
    bg_min_frac: float = 0.03         # ... but at least this share of the date's 99th-percentile density
    join_gap: float = 0.03            # runs closer than this along the row are one cluster (m)
    min_width: float = 0.02           # narrower runs are noise (m)
    min_mass_frac: float = 0.01       # clusters with less than this share of the date's median cluster mass are noise
    sigma_window: float = 2.0         # cluster points = within +- this many sigma of the cluster centre
    clod_top: float = 0.05            # a cluster whose plant points never rise above this (m) is a clod / low blob
    stem_gmm_max: int = 8             # max Gaussians per cluster for the per-stem fit
    stem_gmm_points: int = 4000       # subsample for the per-stem fit
    # ---- step 3: matching clusters across dates
    bc_min: float = 0.10              # Gaussians overlapping less than this are never matched
    max_centre_dist: float = 0.08     # (m) centres farther apart are never matched
    gap_cost: float = 0.45            # cost of leaving a cluster unmatched (missing / extra)
    split_penalty: float = 0.10       # extra cost per additional cluster in a split / merge
    max_split: int = 3                # one cluster may correspond to at most this many on the other date
    dbscan_eps: float = 0.08          # (m) DBSCAN radius on cluster centres; < half the closest reference spacing (20 cm)
    dbscan_min_samples: int = 2       # seen on at least 2 dates (same rule as the plant register)
    # ---- step 4: fine alignment per cluster (image cross-correlation of the stem-base layer)
    fine_search: float = 0.05         # search the shift within +- this (m)
    fine_max_accept: float = 0.03     # accept at most this (m): ~3x the global alignment error
    fine_min_gain: float = 0.02       # the correlation must improve by at least this

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlantConfig:
    # ---- step 5: plant check
    search_radius: float = 0.10       # candidate plants: stem bases within this (m, reference frame)
    cluster_margin: float = 0.03      # a plant belongs to a cluster if its base lies within the stretch +- this
    base_scale: float = 0.05          # cost: base distance / this (bases are noisy by 3-5 cm)
    lean_scale: float = 0.27          # cost: lean-vector difference / this (0.27 = tan 15 deg)
    contain_weight: float = 2.0       # cost: weight * (1 - containment)
    low_region_h: float = 0.25        # containment uses the earlier plant's points up to this height ...
    low_region_frac: float = 0.6      # ... and at most this share of its height
    low_region_r: float = 0.06        # ... within this distance of its stem line (m)
    contain_tol: float = 0.02         # an earlier point is "inside" if a later point is within this (m)
    max_cost: float = 5.0             # pairs above this cost are not assigned
    ambiguous_margin: float = 0.30    # ambiguous if the second-best candidate is within this cost
    # ---- step 6: stem matching
    stem_step: float = 0.01           # compare the centre lines every 1 cm of height
    stem_min_h: float = 0.03          # ... from this height above ground
    stem_flag_cm: float = 3.0         # flag a stem pair whose mean deviation is above this (cm)

    def to_dict(self) -> dict:
        return asdict(self)


DATE_COLOURS = {          # same validated palette as the alignment report
    "230601": "#2a78d6", "230621": "#eb6834", "230629": "#1baf7a",
    "230705": "#eda100", "230711": "#e87ba4",
}
DATE_MARKERS = {"230601": "o", "230621": "s", "230629": "^", "230705": "D", "230711": "v"}
STATUS_COLOURS = {        # dataviz status palette; always shown with a text label
    "confirmed": "#0ca30c", "split": "#fab219", "merged": "#fab219",
    "flagged": "#ec835a", "missing": "#9a9a92", "extra": "#9a9a92",
    "clod": "#c3c2b7", "reference": "#1baf7a",
}


def date_label(d: str) -> str:
    return f"{d[4:6]}.{d[2:4]}."
