# Maize plant candidates from a LAS scan

This package starts with a maize-row `.las` scan and generates a coloured point cloud of **candidate plant instances**. It includes the already trained ExtraTrees stem/leaf model. You do not need the original project folders or precomputed `.npy` files.

## First-time setup on Windows

Install Python 3.12, then open PowerShell in this extracted folder:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

The saved model was made with scikit-learn 1.7.2, which is pinned in `requirements.txt`.

## Run one scan

Drag a `.las` file onto `RUN_FROM_LAS.cmd`, or run:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\run_from_las.py' --las 'C:\path\to\your_scan.las'
```

The script performs four stages:

1. Downsample the LAS to roughly 4 mm spacing.
2. Remove soil with the original CSF/height-above-ground code.
3. Apply the bundled ExtraTrees model to classify remaining points as stem, leaf or unknown.
4. Estimate stem-location hints from those predictions, then run stem-guided geometry to propose plant groups.

Open `review/<scan_name>/candidate_axes_all_points.ply` in CloudCompare. Each colour is a **candidate** plant group. Open `stem_axis_review.png` and `stem_axes_to_review.csv` to inspect the proposed stalks. The CSV starts with blank `plant_id` values.

The output coordinates are shifted close to zero so millimetre-scale spacing is retained during the geometric step. `inputs/<scan_name>/work_config.json` records `local_coordinate_origin_xyz_m`; add it to output XYZ if you need the original map coordinates.

To confirm plants, enter a positive `plant_id` for each real stalk candidate in the CSV. If two stalk candidates belong to the same biological plant, give them the same ID. Leave a false or unresolved candidate blank. Then run:

```powershell
& '.\.venv\Scripts\python.exe' -u '.\segment_confirmed.py' --dataset '<scan_name>'
```

This saves reviewed point clouds under `results/<scan_name>/`. The run name is printed at the end of the first command and appears in the `review` folder.

## What the results mean

- The output contains the **observed, downsampled plant points**, not every point from the raw LAS. Soil is removed; missing scan points are not reconstructed.
- This is for maize scans of one approximately straight crop row, like the five project dates. Other fields may need different settings.
- Automatic colours are proposals. Close stalks, tillers, touching leaves and missing stem bases can cause wrong splits or assignments. Confirm biological plant identity in the CSV before measuring traits.
- Very young seedlings have almost no visible stalk. On the June 1 prepared scan, this stem-guided method found only one axis across the row, so that date is currently **underdetected** even though the command runs. Do not use its colours as a plant count.
- The older review used saved stem-location hints from a separate experiment. This runner estimates hints from the current scan, so its result may differ from the earlier screenshot even on the same date. Accuracy on an unseen LAS has not been measured.
- A second run on the same unchanged LAS keeps an existing review CSV intact. To start over, first move or rename that scan's `review/<scan_name>` folder.

The main entry point is `run_from_las.py`. The original ExtraTrees feature, downsampling and CSF code is in `pipeline/`; `gap_segmentation.py` contains the candidate instance algorithm.

## Code provenance

The bundled ExtraTrees model and the feature/ground-removal routines come from the project's existing RFMF work. The candidate plant grouping code comes from the project's experimental stem-guided geometry work. This package connects those stages so they can run on a new LAS without copying precomputed predictions or hints from another date.
