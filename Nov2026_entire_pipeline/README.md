# Nov2026_entire_pipeline

A copy of the maize field pipeline (`../plant_structure_to_leaf_code`, made 10.11.2026) in which **one step is replaced**: the plant instances (stage 5, splitting the plant points of a date into single plants) now come from the code in `../Plant_instance_fadly_nov2026`. Every other step is the same code as the original pipeline. The original pipeline was not changed.

| Step | Original pipeline | Nov2026_entire_pipeline |
|---|---|---|
| 1-2 soil / plant | `classify_plants.py` (user's code) | same |
| 3 alignment of dates | `align.py` | same |
| 4 stems, plant register, clusters | `stems.py` | same |
| **5 plant instances** | `separate.py` (voxel graph from the stem axes) | **`plant_instance_fadly/`** via `pipeline/instances_fadly.py` |
| 6 structure + traits | user's structure script | same |
| 7-8 tracking, growth, report, GT check | same | same |

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
../venv/bin/python run_pipeline.py --small-plants 230601                            # original separation
../venv/bin/python run_pipeline.py --small-plants 230601 --small-method fadly_small # scaled heights
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
