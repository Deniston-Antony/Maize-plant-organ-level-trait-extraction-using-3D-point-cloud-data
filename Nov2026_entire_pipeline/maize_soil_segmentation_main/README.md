# Maize soil segmentation

This folder shares the two Python scripts used for plant/soil separation and optional evaluation.

## Files

- `soil_segmentation/classify_plants.py` reads a `.las` scan and saves classified, plants-only, and ground-only `.ply` files.
- `soil_segmentation/evaluate_classification.py` compares the classified PLY with a manually labelled `GT_manual.las` file. It is optional and does not perform soil separation.

## Setup

Tested with Python 3.12. From this repository's root in PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Put your scan here (replace the dataset name with your own):

```text
data/
  230601pc_kinematic_calibration/
    230601pc_kinematic_calibration.las
    230601pc_kinematic_calibration_GT_manual.las  # optional; evaluation only
```

The coordinate distances and thresholds in the scripts are in metres. Edit `DATASET` in both scripts to match your folder and file name. In both scripts, set `USE_SIDEBAND = True` for the small-plant row experiment or `False` to keep the full bed width. The classifier currently has `USE_SIDEBAND = False`, so change it before running the small-plant mode. Set the same value in the evaluator. Other thresholds are dataset-specific and should be checked on your scans.

## Run

```powershell
& .\.venv\Scripts\python.exe soil_segmentation\classify_plants.py
```

The plant cloud is saved in `data/<DATASET>/output/soil_segmentation/` as `*_plants_only_sideband.ply` or `*_plants_only_nosideband.ply`. The script also saves a full coloured classification and a ground-only PLY.

If you have the manually labelled LAS, run:

```powershell
& .\.venv\Scripts\python.exe soil_segmentation\evaluate_classification.py
```

The evaluator opens a 3D window and prints the classification results after you close it. Set `SHOW = False` in the evaluator if you only want the printed results.

