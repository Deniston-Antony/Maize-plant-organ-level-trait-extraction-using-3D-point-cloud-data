# Stage 9: cross-date matching (clusters → plants → stems → leaves)

Matches what was found on each date to the reference date (29.06.). It runs at the end of every full pipeline run
and writes `<run>/cross_date_matching/`. Code: `stage.py` (`run_cross_date`) and one module per step.

Everything is geometry. It uses density profiles, Gaussian fits, order-preserving sequence alignment, DBSCAN
(unsupervised), image cross-correlation and the Hungarian method. No trained model is used.

## Run on its own (from the pipeline folder)

```bash
../../venv/bin/python run_matching.py                        # newest finished run
../../venv/bin/python run_matching.py --run outputs/run_...  # a given run
../../venv/bin/python run_matching.py --until clusters       # steps 1-4 only (~25 s)
../../venv/bin/python run_matching.py --retune               # choose the leaf rule again (step 7)
../../venv/bin/python view_3d.py --list                      # 3D viewer
../../venv/bin/python view_3d.py 16/03 [--overlay] [--save p.png] [--run outputs/run_...]
../../venv/bin/python -m pipeline.cross_date.report outputs/run_.../cross_date_matching   # rebuild the PDF
```

Step 7 uses your structure code's result on the 4 ground-truth plants (`outputs/_gt_structure/`). If that cache is
missing, it is made automatically (~25 s, once). The chosen leaf rule is stored per run in
`03_leaves/leaf_rule.json`.

## Status

| step | what | state |
|---|---|---|
| 1 | one frame: final shifts from `alignment.json`, height above ground | done |
| 2 | clusters per date from the stem-base density profile + one Gaussian per cluster | done |
| 3 | match clusters to 29.06.: A = sequence alignment, B = DBSCAN; confirmed / split / merged / missing / extra / flagged | done |
| 4 | fine alignment per matched cluster (image cross-correlation) | done |
| 5 | plant check inside matched clusters (Hungarian) | done |
| 6 | stem matching (centre lines) | done |
| 7 | leaf pieces → real leaves (rule tuned on the 21.06. ground truth) | done |
| 8 | leaf matching (order-preserving alignment with stem stretch, leaf IDs from 29.06.) | done |
| 9 | tables, figures, PDF (`cross_date_matching_report.pdf`) | done |

## Step 2 in short

1. **Points.** The low layer (3–15 cm above ground) within 10 cm of the row centre line, thinned to 5 mm so every date has the same density scale.
2. **Column filter.** Only points in 2 cm cells whose points span at least 3 cm in height are kept, i.e. upright stems and not low leaves.
   - It is skipped on a seedling date (01.06.); the clod test is used there instead.
3. **Density profile.** Points per 1 cm bin along the row.
4. **Clusters.**
   - **Threshold:** background (the mean density in the gaps) + 3 sd.
   - **Clusters:** the stretches above the threshold, joined across dips shorter than 3 cm.
   - **Gaussian:** a density-weighted mean and sd for each cluster.
   - **Cluster points:** the points within ±2σ.
5. **Clods.** A cluster whose plant points never rise above 5 cm is marked as a clod.

## Outputs (`<run>/cross_date_matching/01_clusters/`)

| file | content |
|---|---|
| `clusters_per_date.csv` | every cluster on every date: stretch, Gaussian centre and width, clod flag, DBSCAN group |
| `stem_gaussians.csv` | per-stem Gaussians inside each cluster (2-D mixture, number chosen by BIC); used by step 5 |
| `cluster_match.csv` | each reference cluster × date: matched clusters, method A result, method B verdict, status, overlap, centre difference |
| `fine_alignment.csv` | fine shift per matched cluster, correlation before and after, accepted or not (and why) |
| `register_comparison.csv` | the 29.06. clusters compared with the pipeline's register clusters |
| `fig1`–`fig4` | density profiles with Gaussians, the cluster matches, the fine shifts, top-view examples |

`02_plants_stems/`: `plant_check.csv`, `stem_match.csv`, `assignment.json` (geometric plant pairs used for the
leaves), figures 5–6. `03_leaves/`: `leaf_rule.json` (+ `leaf_rule_tuning.csv`), `leaves_per_plant.csv`,
`leaf_match.csv`, `leaf_tracks.csv` (per-leaf growth), `plants/` (one figure per 29.06. plant), figures 7–9.
Summary: `cross_date_matching_report.pdf` (rebuild: `python -m matching.report <output folder>`).

## Names

- A 29.06. cluster that corresponds one-to-one to a register cluster carries its name, e.g. `plant_cluster_03`.
- Any other 29.06. cluster is `ref_cluster_KK` (KK = its order along the row).
- Clusters on the other dates are `<date>_cNN`.

All settings are in `pipeline/cross_date/config.py` (steps 1-6), `leaf_pieces.py` (step 7 grid) and `leaf_match.py` (step 8 weights).
