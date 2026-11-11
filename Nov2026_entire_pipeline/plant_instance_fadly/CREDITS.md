# Credits

The code in this folder is by **fadly6800**: [fadly6800/maize-soil-segmentation](https://github.com/fadly6800/maize-soil-segmentation)

- ExtraTrees stem / leaf model and its geometric features (`pipeline/rfpipe.py`, `model/`)
- downsampling and ground removal (`pipeline/step1_downsample.py`, `pipeline/step2_remove_ground.py`)
- stem-guided plant instance segmentation (`gap_segmentation.py`, `run_from_las.py`, `run_review.py`, `segment_confirmed.py`)

It is an unchanged copy of `Plant_instance_fadly_nov2026/plant_instances_from_las` (received 10.11.2026), used by the Nov2026 pipeline through `../pipeline/instances_fadly.py`.
