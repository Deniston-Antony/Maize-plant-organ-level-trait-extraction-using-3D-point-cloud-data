"""Shared settings, paths and output-folder naming for the maize pipeline.

All distances are in metres. All point coordinates inside the pipeline are
kept in the original UTM frame of the scans (float64), so plants can be
matched across dates and to ground truth by position. Files written to disk
store coordinates relative to PROJECT_ORIGIN (float32-safe) and the origin is
written next to them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent          # plant_structure_to_leaf_code/
DATASET_DIR = CODE_DIR / "plant_structure_to_leaf_datasets"
OUTPUT_ROOT = CODE_DIR / "outputs"


def find_up(name: str, start: Path = CODE_DIR) -> Path:
    """`name` in the code folder or the nearest folder above it (Research_project,
    P147, ... live next to the project, at any depth below Research_Project_1)."""
    for p in (start, *start.parents):
        if (p / name).exists():
            return p / name
    return start.parent / name
ASSETS_DIR = CODE_DIR / "assets"

# One fixed origin for every date. float32 has ~7 significant digits, so raw
# UTM values (5 610 175 m) would lose ~0.5 m of precision if saved as float32.
PROJECT_ORIGIN = (357659.0, 5610170.0, 222.0)

REFERENCE_DATE = "230629"   # date whose groups/plants define the names

# Folder names inside one run (numbered so they sort in processing order).
STAGE_DIRS = {
    "soil": "01_soil",
    "stems_groups": "02_stems_groups",
    "plants": "03_plants",
}
RUN_LEVEL_DIRS = {
    "alignment": "alignment",
    "ground_truth": "ground_truth",
    "tracking": "tracking",
    "growth": "growth",
    "cross_date": "cross_date_matching",     # stage 9 (pipeline/cross_date)
}

_DATE_RE = re.compile(r"(?<!\d)(2\d)(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)")


def date_from_name(name: str) -> str | None:
    """Return the first YYMMDD date found in a file/folder name, e.g. '230621'."""
    m = _DATE_RE.search(name)
    return "".join(m.groups()) if m else None


def pretty_date(yymmdd: str) -> str:
    """'230621' -> '21.06.2023'."""
    return f"{yymmdd[4:6]}.{yymmdd[2:4]}.20{yymmdd[0:2]}"


@dataclass
class RunPaths:
    """Every folder of one pipeline run. Created on demand."""

    root: Path

    @classmethod
    def new(cls, output_root: Path = OUTPUT_ROOT, name: str | None = None) -> "RunPaths":
        name = name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        return cls(Path(output_root) / name)

    def date_dir(self, date: str) -> Path:
        return self._mk(self.root / date)

    def stage_dir(self, date: str, stage: str) -> Path:
        return self._mk(self.root / date / STAGE_DIRS[stage])

    def plant_dir(self, date: str, plant_id: str) -> Path:
        group = plant_id.split("_")[0]
        return self._mk(self.root / date / STAGE_DIRS["plants"] / group / plant_id)

    def run_dir(self, key: str) -> Path:
        return self._mk(self.root / RUN_LEVEL_DIRS[key])

    def gt_dir(self, gt_name: str) -> Path:
        return self._mk(self.root / RUN_LEVEL_DIRS["ground_truth"] / gt_name)

    def plants_index(self, date: str) -> Path:
        return self.root / date / STAGE_DIRS["plants"] / "plants_index.csv"

    @staticmethod
    def _mk(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p


def plant_id(group: int, plant: int) -> str:
    return f"group{group:02d}_plant{plant:02d}"


@dataclass
class GroundTruthConfig:
    """Settings for comparing manual ground truth with predictions."""

    voxel_size: float = 0.004          # both clouds thinned to one point per 4 mm voxel before IoU
    iou_tolerance: float = 0.008       # a point counts as matched if the other cloud has a point within 8 mm
    match_radius: float = 0.015        # GT point "hits" a predicted point within 1.5 cm
    min_match_share: float = 0.20      # best predicted plant must hold >= 20 % of GT points
    base_fallback_radius: float = 0.15 # if no overlap: nearest predicted base within 15 cm
    icp_search_radius: float = 1.0     # ICP fallback: try predicted plants within 1 m of GT centre
    icp_max_corr: float = 0.02         # ICP correspondence distance (2 cm)
    min_leaf_iou: float = 0.05         # leaf pairs below this IoU are "unmatched"
    fragment_min_share: float = 0.5    # a predicted piece whose voxels are >= 50 % inside a GT leaf
                                       # is counted as a fragment of that leaf
    unlabelled_value: int = -2         # label for GT points without stem/leaf labels

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructureConfig:
    """Extra structure settings on top of the reference PipelineConfig."""

    min_stem_points: int = 40          # fewer stem points -> stem axis is rebuilt from leaf bases
    stem_radius_slice: float = 0.02    # stem radius: circle fit per 2 cm height slice
    stem_radius_min_points: int = 12   # slices with fewer points are skipped
    stem_radius_max: float = 0.035     # fits larger than 3.5 cm are rejected (not a maize stem)
    fragment_merge_gap: float = 0.06   # leaf pieces whose ends are < 6 cm apart ...
    fragment_merge_angle_deg: float = 35.0  # ... and point the same way (< 35 deg) are one leaf
    leaf_attach_dist: float = 0.05     # a leaf touches the stem if its closest point is < 5 cm away
    leaf_touch_stem_points: float = 0.04  # leaf check after the user's structure code: a leaf must come
                                       # within 4 cm of the script's stem points (4 mm work cloud; GT 21.06:
                                       # real leaves <= 3.4 cm, pieces of neighbouring plants >= 5.7 cm)
    leaf_check_min_stem_points: int = 50  # fewer stem points -> use leaf_attach_dist to the stem axis instead
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- run settings
# Every run uses the complete scan at full point density (no thinning of the
# plants before segmentation). Thinning to 3 mm was tested: 3-5x faster but
# leaf count / leaf area changed by up to 40 %, so it was dropped.
PLANT_WORKERS = 8   # parallel plant processes (10-core laptop)
# Plant structure exactly as the user's script gives it (copy_of_New_optimized_plant_structure.py).
# True = also run the leaf check after it (join leaf pieces, drop pieces not reaching the stem).
USE_LEAF_CHECK = False

# Nov2026 pipeline: plant instances (stage 5) from the plant_instance_fadly code
# (../plant_instance_fadly, see pipeline/instances_fadly.py). "own" = the original separate.py.
INSTANCE_METHOD = "fadly_nov2026"

# 24.11.2026: parts of stage 9 (cross-date matching) moved earlier, so later stages use them.
# Stage 4 "3b": clusters per date are matched to the reference date and each matched cluster gets a
# fine shift (stem-base image cross-correlation, <= 3 cm) on top of the date's alignment, BEFORE the
# stems of all dates are pooled into the plant register. Stored in alignment.json (local_shifts), so
# every later step that moves points between dates uses it.
CLUSTER_FINE_SHIFT = False   # tested 25.11.: no clear gain (2 plants lost, fewer leaf matches) - off
# Stage "6b" (after the plant structure): plant check + stem matching against the reference date.
# Where one plant carries two register names (each separated on only one of two dates, same plant by
# geometry), tracking and growth use the reference name. Results are reused by stage 9.
IDENTITY_CHECK = True
IDENTITY_MERGE_MIN_CONTAINMENT = 0.8   # merge two names only if the earlier plant lies this much inside the later
IDENTITY_MERGE_MAX_STEM_DEV_CM = 3.0   # ... and their stem lines are at most this far apart AT THE SAME HEIGHTS
IDENTITY_MERGE_MIN_COMMON_CM = 5.0     # ... compared over at least this much common stem height
# (the lowest points of two stem lines are no measure: a hidden stem base starts the line higher up)
# Stage 4 register: one register plant holds at most one stem per date. A pooled group with two stems
# of the same date is split (seeds = stems of the date with the most stems; the other dates' stems are
# assigned one-to-one, Hungarian). Two stems seen on the same date are always two plants.
REGISTER_ONE_STEM_PER_DATE = True
