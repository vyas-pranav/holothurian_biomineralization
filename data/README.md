# Data

This repository contains **code only**. The datasets analysed in the paper are available from the lead contact on request. They comprise:
- micro-CT scans and per-ossicle meshes of juvenile *Apostichopus parvimensis*;
- live and confocal imaging;
- vesicle kymographs and tracks.

The model scripts (folders 05 and 06) need no input data.

## Where the scripts look

Every script reads its inputs relative to `OSSICLE_DATA`, which defaults to this `data/` folder. It writes results relative to `OSSICLE_OUT`, which defaults to `outputs/` at the repository root.

```bash
export OSSICLE_DATA=/path/to/ossicle-data
export OSSICLE_OUT=/path/to/results
```
In PowerShell: `$env:OSSICLE_DATA = "D:\ossicle-data"`.

## Expected layout under `OSSICLE_DATA`

| Folder | Contents | Used by |
|---|---|---|
| `microct/animal_1/extracted_ossicles/` | Juvenile used for Fig. 1–2 (micro-CT, 0.82 µm voxels; segmented in Dragonfly 2022.2). Subfolders: `mesh/`, `vtk/`, `skeleton/`, `aligned/` and `Data Calculated/`, with per-ossicle tables such as `ossicle_type_list.txt`. | `01_microct_morphometrics` |
| `microct/animal_5/`, `microct/animal_1_20220825/`, `microct/animal_3_20220825/`, `microct/mid_01_20220822/`, `microct/mid_02_20220823/`, `microct/relaxed_animal_20220612/` | Meshes of further juveniles at different growth stages, pooled for the surface-area vs volume scaling (Fig. S5I) | `01_microct_morphometrics/fields/size_scaling/sa_vol.py` |
| `tables/hole_area_vol/` | Per-ossicle hole count, area and volume table (`hole_sa_vol_stat.xlsx`) | `01_microct_morphometrics/fields/size_scaling/plot_data.py` |
| `tables/angle/` | Bifurcation-angle tables written by `all_angle_plots2.py` | `01_microct_morphometrics/fields/angles/` |
| `footpad/` | Adult foot-pad ossicle DIC images and masks, plus the graphs derived from them | `02_image_morphometrics_2d/footpad`, `01_microct_morphometrics` `*_foot` scripts |
| `growth_series/` | Bright-field time-lapse frames and segmentation masks of a growing front-plate ossicle (Fig. 3G–H) | `02_image_morphometrics_2d/growth_series` |
| `vesicles/confocal_2025/` | Confocal movies of membrane-labelled vesicles, with path masks, kymographs and tracks (nocodazole series; Fig. 5, S16) | `03_vesicle_transport` |
| `vesicles/front_plate_2024/` | Front-plate vesicle dataset (Fig. S15) | `03_vesicle_transport` |

Each folder README lists the exact files a script reads and writes.
