# 24th_Nov_2026_full_pipeline

The complete maize field pipeline (copy of `Nov2026_entire_pipeline`, made 24.11.2026) **plus stage 9, cross-date
matching**: the plant clusters, plants, stems and leaves of every date are matched to the reference date (29.06.),
and every leaf keeps the same leaf ID on all dates. Stages 1-8 are the same code as `Nov2026_entire_pipeline`
(soil and plant structure = your code; plant instances = `plant_instance_fadly/`). Nothing in
`Nov2026_entire_pipeline` was changed.

| Step | Code |
|---|---|
| 1-2 soil / plant | `classify_plants.py` (your code) |
| 3 alignment of dates | `pipeline/align.py` |
| 4 stems, plant register, clusters | `pipeline/stems.py` |
| 5 plant instances | `plant_instance_fadly/` via `pipeline/instances_fadly.py` |
| 6 structure + traits | your structure script (`Research_project/copy_of_New_optimized_plant_structure.py`) |
| 7-8 tracking, growth, report, GT check | `pipeline/tracking.py`, `pipeline/report.py`, `pipeline/ground_truth.py` |
| 4 · **3b cluster fine shift** | `pipeline/cross_date/early.py` (switch `CLUSTER_FINE_SHIFT`) |
| 6b **identity check** | `pipeline/cross_date/early.py` (switch `IDENTITY_CHECK`) |
| **9 cross-date matching** | **`pipeline/cross_date/`** (see [its README](pipeline/cross_date/README.md)) |

### Parts of stage 9 that run earlier (switches in `pipeline/config.py`)

**3b, cluster fine shift** (inside stage 4, `CLUSTER_FINE_SHIFT = True`).

When it runs: after each date's shift is final (including the 01.06. stem refinement), and before the stems of all dates are pooled into the plant register.

What it does:
1. Clusters are found on every date from the stem-base layer and matched to 29.06.
2. Each matched cluster gets a fine shift of at most 3 cm (stem-base image cross-correlation).
3. The shifts are stored in `alignment.json` (`local_shifts`), so every later step that moves points between dates uses them: the register, plant separation, stage 9 and the app.

Details are in `<run>/alignment/cluster_fine_shift/`.

**6b, identity check** (after stage 6, before tracking; `IDENTITY_CHECK = True`).

What it does:
1. It runs stage 9 steps 1–6 (cluster matching, plant check, stem matching).
2. Where one plant carries two register names, tracking and the growth curves use the 29.06. name. This needs all of:
   - each name was separated on only one of two dates;
   - containment ≥ 0.8;
   - stem lines ≤ 3 cm apart **at the same heights**, over ≥ 5 cm of common stem height.

   The lowest stem points are not used: a hidden stem base makes a stem line start higher up. The cases are drawn in `tracking/identity_check.png`.

Folders and files on disk keep their names. The merges are listed in `tracking/identity_check.csv`. Stage 9 then reuses steps 1–6 and only adds the leaves and the report.

**Register rule** (stage 4, `REGISTER_ONE_STEM_PER_DATE = True`): a register plant holds at most one stem per date.
- If pooling puts two stems of the same date into one group, the group is split: the stems of the date with the most stems become seeds, and the other dates' stems are assigned to them one-to-one (Hungarian).
- Without this rule, the 3b shifts could let stems of other dates bridge two real stems 4 cm apart.

Run without them (`--no-stem-rule` for the register rule):

```
../../venv/bin/python run_pipeline.py --no-cluster-shift --no-identity-check
```

Compare two runs:

```
../../venv/bin/python compare_runs.py outputs/run_A outputs/run_B
```

## Running

From this folder (the project venv is two folders up):

```
../../venv/bin/python plant_structure_to_leaf.py   # app
../../venv/bin/python run_pipeline.py              # terminal: all dates + ground-truth check + stage 9
../../venv/bin/python run_matching.py              # stage 9 only, on the newest finished run (~1 min)
../../venv/bin/python view_3d.py --list            # 3D viewer: plants with tracked leaves
../../venv/bin/python view_3d.py 16/03             # one plant on all dates, leaves coloured by leaf ID
open outputs/run_*/cross_date_matching/cross_date_matching_report.pdf
```

In the app, **Step viewer → 8 Cross-date matching** shows stage 9:
- **Plant cluster:** a 29.06. cluster on every date. The separated plants are coloured by the 29.06. plant they were matched to, so the same colour means the same plant on every date.
- **Plant: stem & leaves:** one plant on every date, with the stem in black and leaves coloured and labelled by leaf ID.

The 3D views are linked. Under them, a table gives the matching result per date. There are also buttons for "Re-run stage 9" and "Open report (PDF)".

Stage 9 runs at the end of every full run (`run(..., cross_date=True)` in `pipeline/runner.py`; a failure there is
logged and never loses the run). Its results are in `<run>/cross_date_matching/`, linked from the run's
`report.html`.

Locations: results go to `outputs/` here. The scans are not copied (`plant_structure_to_leaf_datasets` is a link
to the original folder). `Research_project/` (your structure script) and `P147/` (ground truth) are found in the
nearest folder above this one (`find_up` in `pipeline/config.py`), so the folder can sit at any depth.
`outputs/run_20261114_1513_...` is the earlier full run: its stage 1-8 folders link to
`Nov2026_entire_pipeline/outputs/`, its `cross_date_matching/` and `report.html` are real files here.
`outputs/_gt_cache` (ground-truth export) and `outputs/_gt_structure` (your structure code on the ground-truth
plants, used to choose the leaf rule) are caches.

## Flow diagram and settings

[docs/pipeline_flow_and_settings.pdf](docs/pipeline_flow_and_settings.pdf) (also as [HTML](docs/pipeline_flow_and_settings.html)): flow diagram of every step, and one table per step with each setting's default, what it does, and what happens when it is increased or decreased.

## Running

Same as the original, from this folder and with the same venv:

```
../venv/bin/python plant_structure_to_leaf.py     # app (window title says "Nov2026")
../venv/bin/python run_pipeline.py                # terminal, all dates + ground-truth check
```

Results go to `Nov2026_entire_pipeline/outputs/`. The scans are not copied: `plant_structure_to_leaf_datasets` is a link to the original folder.

To run this copy with the original separation instead, set `INSTANCE_METHOD = "own"` in `pipeline/config.py`.

### Small plants (seedling dates such as 01.06)

The instance code finds a plant only when it can trace an upright stem over at least 12 cm, so it finds no seedlings on 01.06. Before a run, tick the date in the **Small plants** column of the scan list (Setup tab) and choose what those dates use:

- **Original separation (geometry)** - default; the original pipeline's separation step for that date.
- **Instance code with small-plant heights** - the code's height settings times 1/4 (minimum stem 12 → 3 cm, slice 1.5 → 0.375 cm, gap 7.5 → 1.9 cm) and a 4 mm graph voxel.

Dates that are not ticked always use the instance code with its own settings. In the terminal:

```
../../venv/bin/python run_pipeline.py --small-plants 230601                            # original separation
../../venv/bin/python run_pipeline.py --small-plants 230601 --small-method fadly_small # scaled heights
```

The choice is saved in `run_config.json` (`small_plant_dates`, `small_plant_method`).

## How the plant instance code is used

`plant_instance_fadly/` is an unchanged copy of `Plant_instance_fadly_nov2026/plant_instances_from_las` (including the ExtraTrees model). `pipeline/instances_fadly.py` runs it the way its README describes for an already soil-removed cloud (`run_from_las.py --prepared-ground-ply`):

1. the plant points of the date (soil already removed by `classify_plants.py`) are thinned to 4 mm with the voxel keys of `step1_downsample.py` (first point per voxel, grid anchored at the LAS header minimum, map coordinates with 6 decimals as in its PLY files);
2. stem / leaf / unknown per point with the bundled model (`rfpipe` features, forest, smoothing, confidence threshold, all from the model file);
3. stem hints from the predicted stem points (`run_from_las.stem_location_hints`), local float32 coordinates as in its `xyz.npy`;
4. review pass as `run_review.py` (`gap_segmentation.segment` with `Settings()` defaults), then the confirmed pass as `segment_confirmed.py`.

All settings are the code's own; nothing is tuned.

Its own LAS downsampling and CSF soil removal (README steps 1-2) are not used, because soil stays the user's code.

**The manual review is filled in automatically.** The code's README asks a person to give each candidate stem axis a `plant_id` in `stem_axes_to_review.csv`. Here each axis is matched to the nearest plant of the run's plant register within 9 cm (the code's own gate for matching a stem-location prior to an axis), so a plant has the same name on every date. Axes with no register plant are left blank, which the README treats as rejected: they are not used.

**Labels used:** the conservative `plant_id` of `segment_confirmed.py` (points with an observed path to one stem and no near-tie). The other option, `candidate` (also points reached through gaps), can be chosen with `FadlyInstanceConfig.labels` in `pipeline/instances_fadly.py`.

Per date the code's own outputs are saved in `outputs/<run>/<date>/03_plants/instances_fadly/`:
`plants_conservative.ply`, `plants_candidate_including_gaps.ply`, `semantic_stem_leaf.ply`, `stem_axes_auto_review.csv` and `summary.json`. Coordinates there are local; `summary.json` gives the origin to add.

## Things to know

- The plant instance step now uses a **trained model** (ExtraTrees). The original pipeline is geometry-only.
- The model was saved with scikit-learn 1.7.2; the venv has 1.8.0. Tested on 21.06: 35 of 219,901 points get a different stem/leaf label (0.02 %).
- One difference stays: the code thins the whole LAS and then removes soil with its own CSF; here soil is removed first (your code) and only plant points are thinned, so a 4 mm voxel at the soil can keep a different point.
- A plant gets points only if the code finds its stem axis (an upright stem traced over at least 12 cm, starting near the ground). Plants without a found axis are reported as `no_stem_axis_found` and get no structure.
- The model file (`plant_instance_fadly/model/et_model.joblib`, 90 MB) is not tracked by git (only code is).

## Setup after cloning

The model file is not in git (90 MB; only code is tracked). Copy it from the plant_instance_fadly package:

```
mkdir -p Nov2026_entire_pipeline/plant_instance_fadly/model
cp Plant_instance_fadly_nov2026/plant_instances_from_las/model/et_model.joblib Nov2026_entire_pipeline/plant_instance_fadly/model/
```

The scans folder `plant_structure_to_leaf_datasets` is a link to `../plant_structure_to_leaf_code/plant_structure_to_leaf_datasets` and is not in git either. Create it again with:

```
ln -s ../plant_structure_to_leaf_code/plant_structure_to_leaf_datasets Nov2026_entire_pipeline/plant_structure_to_leaf_datasets
```

## Results (full run of all 5 dates, 10.11.2026)

Plants with points, of 55 register plants:

| Date | Instance code | 01.06 as small plants: original separation | 01.06 as small plants: instance code ×1/4 heights |
|---|---|---|---|
| 01.06 | 0 | 12 | 2 |
| 21.06 | 23 | 23 | 23 |
| 29.06 | 24 | 24 | 24 |
| 05.07 | 19 | 19 | 19 |
| 11.07 | 19 | 19 | 19 |

Ground truth, 21.06, plant IoU (original pipeline → Nov2026): LtR_10th 0.80 → 0.40, LtR_11th 0.70 → 0.48, LtR_12th 0.91 → 0.97, RL_plant01 0.02 → 0.02. The instance code is best on a plant standing on its own and weaker where plants touch. Full run time about 11–12 min.

## Change log

All changes are in `Nov2026_entire_pipeline/`; `plant_structure_to_leaf_code/` is unchanged.

| Date | Change | Files |
|---|---|---|
| 10.11.2026 | Copy of the whole pipeline made from `plant_structure_to_leaf_code` (outputs not copied, scans linked). | whole folder |
| 10.11.2026 | plant_instance_fadly code copied unchanged from `Plant_instance_fadly_nov2026/plant_instances_from_las`. | `plant_instance_fadly/` |
| 10.11.2026 | New stage 5: plant instances from the plant_instance_fadly code (4 mm thinning, ExtraTrees stem/leaf, stem hints, review pass, automatic review against the plant register, confirmed pass, conservative labels). Its outputs are saved per date. | `pipeline/instances_fadly.py`, `pipeline/runner.py`, `pipeline/config.py` (`INSTANCE_METHOD`) |
| 10.11.2026 | Old results of a date are deleted before stage 5 (not after), so the instance code's files are kept. | `pipeline/runner.py` |
| 10.11.2026 | Availability shows "no stem axis found (instance code)" for plants without a found stem. | `pipeline/tracking.py` |
| 10.11.2026 | App window title says "Nov2026". | `ui/app.py` |
| 10.11.2026 | Instance code run with its exact settings: 4 mm voxel keys from the LAS header (as `step1_downsample.py`), coordinates and number types as its files, review pass with the full `segment()`, matching distance 9 cm (the code's own gate), unmatched axes left blank, no extra point minimum. | `pipeline/instances_fadly.py` |
| 10.11.2026 | Small plants option: per-date tick box in the app's scan list and a method choice (original separation, or instance code with heights ×1/4 and a 4 mm graph voxel); `--small-plants` / `--small-method` in the terminal; saved in `run_config.json`. | `ui/app.py`, `pipeline/runner.py`, `pipeline/instances_fadly.py`, `run_pipeline.py` |
| 11.11.2026 | README with setup, results and this change log. | `README.md` |
| 11.11.2026 | Flow diagram and settings tables (PDF + HTML). | `docs/` |
