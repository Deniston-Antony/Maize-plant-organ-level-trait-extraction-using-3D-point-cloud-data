# Maize field pipeline: soil, plants and plant structure from LiDAR scans

Research Project 1, University of Bonn (Semester 2) - Deniston Antony.

This code takes LiDAR point clouds (`.las`) of one maize field plot, scanned on several dates, and works out the following for every date:

1. which points are **soil** and which are **plant**,
2. how to **align** the dates so the same plant sits in the same place on every date,
3. the **stems** and the **plant clusters** (plants standing close together along the row),
4. each single **plant instance**,
5. the **structure** of every plant instance (stem, leaves, skeleton) and its **traits** (height, stem length and diameter, leaf count, leaf area, leaf length and width, leaf angle),
6. how each plant instance **grows** from date to date,
7. how well the results match hand-labelled **ground truth**.

It only uses geometry (point positions); no colour and no machine-learning model. A Tkinter app runs the steps and shows every result. Each step can be checked in 2D figures and in 3D views you can rotate. At the end the app writes an HTML report.

The scans used during development (five dates in 2023: 01.06, 21.06, 29.06, 05.07 and 11.07) are **not** in this repository because they are too large.

---

## Repository layout

```
plant_structure_to_leaf_code/
  plant_structure_to_leaf.py   start the app (or run the ground-truth check without it)
  run_pipeline.py              run the whole pipeline in the terminal, without the app
  show_in_my_viewer.py         open one plant instance in the PyVista structure viewer
  pipeline/                    the processing steps
    soil.py                    1  soil / plant separation (uses classify_plants.py)
    align.py                   2  alignment of the dates onto a reference date
    stems.py, stem_tracking.py 3  stem detection and plant clusters
    separate.py                4  plant instance separation (voxel graph from the stem axes)
    plants_stage.py,           5  plant structure and traits
      user_code.py, structure*.py
    tracking.py                6  matching plant instances across dates, growth tables and figures
    ground_truth.py            ground-truth check (IoU per organ, trait errors)
    tolerance.py               deviation / error values shown in the report
    report.py                  HTML report
    names.py                   names shown to the user (plant_cluster_03_plant_instance_01)
    runner.py, config.py       runs the steps in order; all settings
  ui/                          the Tkinter app (step viewer, 3D views, compare dates)
  maize_soil_segmentation_main/soil_segmentation/
    classify_plants.py         soil / plant separation code (cloth simulation filter + terrain model)
  tests/                       ground-truth test and benchmarks
  assets/                      image used in the app's help
```

---

## Installation

Python 3.12 was used for development.

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy laspy open3d cloth_simulation_filter matplotlib pyvista vtk pillow scikit-learn networkx pandas
```

Tkinter comes with most Python installations. On macOS with Homebrew Python, install it with `brew install python-tk`.

### Data

Put the scans in

```
plant_structure_to_leaf_code/plant_structure_to_leaf_datasets/
```

The first six characters of each file name must be the date as `YYMMDD`, for example `230629_pc_kinematic_calibration.las`.

### Plant structure code

The structure step runs the script `copy_of_New_optimized_plant_structure.py`. The pipeline looks for it next to this folder, in `Research_project/` (a separate repository). If the script is not there, `pipeline/reference_structure.py` is used instead.

---

## Running

### With the app

```bash
cd plant_structure_to_leaf_code
python plant_structure_to_leaf.py
```

- **Setup tab:**
  - Tick the dates.
  - Choose the reference date. The other dates are moved onto it, and plant clusters are named from it.
  - Choose how far to run, then start.
  - The run name ends with the ticked dates, for example `run_20261104_0919_230601_230621_230629_230705_230711`.
- **Step viewer:** one page per step:
  1. Soil & plants
  2. Alignment
  3. Stems & plant clusters
  4. Plant instances
  5. Plant structure
  6. Growth
  7. Compare dates: the same plant instances of all dates on top of each other, with one linked 3D view per date

  In the 3D views:
  - drag to rotate
  - scroll or use two fingers to zoom
  - double-click to reset the view
- **Results tab:**
  - plant instances with their availability per date
  - the ground-truth check
  - the structure viewer
  - **Open report**

### In the terminal

```bash
cd plant_structure_to_leaf_code
python run_pipeline.py                      # all dates in the dataset folder, with the ground-truth check
python run_pipeline.py 230621 230629        # only these dates
python run_pipeline.py --no-gt              # without the ground-truth check
python run_pipeline.py --step plants        # stop after a step: soil, alignment, stems, plants, all
```

Open the finished run in the app with **Open run...**.

Other commands:

```bash
python plant_structure_to_leaf.py gt-check outputs/<run> <ground truth> [--date 230621]
python show_in_my_viewer.py outputs/<run> 230629 group01_plant01
```

---

## Output

Each run writes one folder, `plant_structure_to_leaf_code/outputs/run_<date>_<time>_<dates>/`:

```
<date>/01_soil/            soil and plant points, terrain model, check figure
<date>/02_stems_groups/    stems and plant clusters, check figure
<date>/03_plants/groupNN/groupNN_plantMM/
                           points with stem / leaf labels, skeleton, traits (json)
alignment/                 shift of every date onto the reference date, check figures
tracking/                  plant register, availability per date, traits per plant instance and date
growth/                    growth tables and figures per trait
ground_truth/              ground-truth check (if run)
report.html                report of the whole run
```

On disk, plant clusters and plant instances keep their short IDs (`group03`, `group03_plant01`). The app and the report show them as `plant_cluster_03` and `plant_cluster_03_plant_instance_01`.

### Report

- **Summary per date:**
  - number of plant clusters and plant instances
  - total leaf area
  - mean plant height, stem diameter, leaf angle at the junction and leaf width
- **Trait table:** one row per plant instance and date.
- **Growth figures.**
- **Alignment table.**
- **Ground-truth check.**
- **Check figures** for every step.

The first (italic) row of each trait table is the **deviation / error (±)** of a single plant instance's value:

- When the run includes its own ground-truth check, it uses those errors.
- Otherwise it uses the errors measured on 3 hand-labelled plants of 21.06.

The alignment table gives the deviation / error of the shift in two directions:

- **x/y:** how much the 1.5 m row sections disagree about the sideways shift.
- **z:** how much the soil heights of the two dates still differ after the shift.

---

## How it works (short)

| Step | Method |
|---|---|
| Soil / plant | Cloth simulation filter on a subsample, then an improved terrain model (DTM). Points higher than a set height above the ground are plant; plant clusters are cleaned (`classify_plants.py`). |
| Alignment | Soil surfaces and stem bases of each date are matched to the reference date in 1.5 m sections along the row, giving a shift (x, y, z) and a confidence. |
| Stems and plant clusters | Stems are found as thin vertical structures above the ground and pooled over all aligned dates. Stems close together along the row form a plant cluster. |
| Plant instances | Points are split between neighbouring plants with a voxel graph: shortest paths from each followed stem axis. |
| Structure and traits | The plant structure script finds the stem, leaf clusters, stem skeleton and leaf midribs, and measures the traits. |
| Tracking and growth | Plant instances are matched across dates by position in the reference frame; missing or covered plant instances are marked. |
| Ground truth | Hand-labelled plants are matched to predicted plant instances; IoU per organ and trait errors are reported. |

All settings are in `pipeline/config.py`.
