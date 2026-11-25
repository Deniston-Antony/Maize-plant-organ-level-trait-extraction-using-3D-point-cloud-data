"""Help text and sample picture for preparing ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

GT_HELP_TITLE = "How to prepare ground truth"

GT_HELP_TEXT = """\
Ground truth = ONE plant, cut out of the scan by hand, with its stem and leaves labelled.
You can load one plant or many. Each one is matched automatically to the plant the pipeline
predicted at the same position (UTM coordinates), so you do NOT need to know its group/plant id.

1. CUT OUT THE PLANT (CloudCompare)
   - Open the scan of the SAME date, use the Segment tool (scissors) to cut out one plant.
   - Do not move, rotate or scale it. If CloudCompare offers a "global shift", accept it:
     it is removed again when you export.
   - Export as LAS or ASC (ASC: precision 6). Those keep the original coordinates.
     If the coordinates still end up shifted, the checker falls back to shape matching
     (slower and less certain - it will say so in the log).

2. SPLIT IT INTO ORGANS
   One cloud for the stem, one cloud per leaf.

3. LABEL THE ORGANS (either way works)
   a) By name:
        <plant>_stem            -> stem
        <plant>_leaf01          -> leaf 1
        <plant>_leaf02          -> leaf 2 ...
        <plant>_leaf03_part1    \\  one leaf in several pieces
        <plant>_leaf03_part2    /   (same number = merged into leaf 3)
   b) By a scalar field named "Classification" or "label":
        0 = stem, 1, 2, 3 ... = leaf number. Pieces of one leaf get the SAME number.
        (CloudCompare: Edit > Scalar fields > Add constant SF, then merge the clouds
         or export each organ - the CloudCompare ASC export of a .bin also works.)

4. PUT THE DATE IN THE NAME   (YYMMDD, e.g. 230621)
   e.g.  230621_LtR_10th_Plant/   or   230621_LtR_10th_Plant.las
   No date? Double-click the date cell in the table and pick one, or leave "auto":
   then every loaded date is tried and the best match is used.

5. LOAD
   "Add GT folder"  - a folder with one file per organ, or a CloudCompare .bin
                      (exported to ASC automatically; CloudCompare must be installed)
   "Add GT file(s)" - one file per plant that has a label field (LAS / PLY / ASC / TXT / CSV)

SPECIAL CASES
   - No visible stem: leave out the stem file. Stem IoU and stem traits are reported as N/A.
   - Unlabelled plant (one file, no labels): only matching, plant IoU and height are checked.
   - Leaf numbers do not have to match the pipeline: leaves are paired automatically
     (best IoU). Predicted leaves that are split in pieces are joined to the GT leaf they lie in.

EXAMPLE
   230621_LtR_10th_Plant/
       LtR10_stem.las
       LtR10_leaf01.las
       LtR10_leaf02.las
       LtR10_leaf03_part1.las
       LtR10_leaf03_part2.las
   or one file:  230621_LtR_10th_Plant.las   (field "Classification": 0 stem, 1..n leaves)
"""


def make_sample_image(out_png: Path, sample_dir: Path | None = None) -> Path | None:
    """Render a labelled sample plant (from the project's own GT if available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pipeline import reference_structure as ref
    from pipeline.ground_truth import load_ground_truth

    if sample_dir is None or not Path(sample_dir).exists():
        return None
    gt = load_ground_truth(sample_dir, log=lambda *_: None)
    xyz = gt.xyz - gt.xyz.mean(axis=0)
    rng = np.random.default_rng(0)
    i = rng.choice(len(xyz), min(len(xyz), 30000), replace=False)
    lut = ref._COLOR_LUT
    cols = lut[np.clip(gt.labels[i], 0, len(lut) - 1)]
    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(xyz[i, 0], xyz[i, 1], xyz[i, 2], c=cols, s=0.6)
    ax.set_box_aspect(np.ptp(xyz[i], axis=0))
    ax.view_init(elev=12, azim=-60)
    ax.set_axis_off()
    handles = [plt.Line2D([], [], marker="o", ls="", color=lut[0], label="0 = stem  (<plant>_stem)")]
    for lid in gt.leaf_ids[:9]:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=lut[lid], label=f"{lid} = leaf {lid}  (<plant>_leaf{lid:02d})"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True)
    ax.set_title("Sample ground truth: one plant, one label per organ", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return out_png
