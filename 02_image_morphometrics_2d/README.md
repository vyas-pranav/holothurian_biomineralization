# 02 · 2D image morphometrics

**Figures:**
- **Adult foot-pad ossicles:** Fig. 2I, Fig. S3B–D, and the foot-pad series in Fig. 2D–H.
- **Live front-plate growth series:** Fig. 3H, Fig. 3J, Fig. S5A.

**Environment:** `environment-analysis.yml` (Python 3.7).

**Data:** paths are relative to `$OSSICLE_DATA` (see `data/README.md`).

Images are segmented (masks drawn by hand) and skeletonized with scikit-image `medial_axis`. The skeletons become NetworkX graphs (`sknw`) that carry the local medial thickness (SI, "Growth series analysis for Figure 3" and "Foot pad ossicle data").

## `footpad/`: adult foot-pad ossicles

Foot-pad tissue from adults was bleached to leave the ossicle, which was imaged by DIC.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `bin2skel2.py`; set `ctr` = 0 or 1 near the end for `test1` or `test2` | `footpad/images/test{1,2}.png`, `footpad/masks/test{1,2}_mask.bmp` | `footpad/graphs/<name>graph_{sknw,full,simple,simple_origin}.gpickle`, `<name>_graph.png`, label and skeleton images |
| 2 | `reassign_origin.py`; set `ctr` | `footpad/graphs/<name>graph_simple.gpickle` | origin moved to the midpoint of a hand-picked edge: `<name>graph_simple_origin.gpickle` |
| — | `segment_from_image.py`; run from the folder that holds `test2.png` and `Image 1-2.bmp` | the thresholded image | hole areas and centroid distances (`data1.csv`) for the foot-pad series of the hole-area plot |

The graphs feed the `*_foot` scripts in `01_microct_morphometrics/fields/`.

## `growth_series/`: volume growth of a live front-plate ossicle

Five bright-field frames of one ossicle are used, taken at t = 0, 6.04, 27.82, 44.68 and 96.58 h. They are named `11-518-1`, `16-70-1`, `24-385-2`, `36-90-1` and `56-283-1`.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `bin2skel.py` | `growth_series/images/*.png`, `growth_series/masks/*.png` | `growth_series/graphs/<name>graph_*.gpickle`, and skeleton overlays `<name>_graph.png` (Fig. 3J, S5A) |
| 2 | `sa_vol_calc.py` | `growth_series/graphs/*graph_full.gpickle` | Each graph edge is treated as a cylinder whose diameter is the medial thickness. Volumes and lateral areas are summed and fitted linearly against time: `volume_increase` and `surface_area_increase` (`.png`/`.svg`) in `$OSSICLE_OUT/02_image_morphometrics_2d/growth_series/` (Fig. 3H) |

**Reproduced from the published graphs:**

| Quantity | This code | Paper |
|---|---|---|
| Volume growth rate | 1.49 × 10⁻⁷ mm³/h (fit standard error 0.13 × 10⁻⁷) | 1.5(±0.1) × 10⁻⁷ mm³/h |
| Surface-area growth rate | 11.36 × 10⁻⁵ mm²/h (fit standard error 0.80 × 10⁻⁵) | 11.4(±0.8) × 10⁻⁵ mm²/h |

## Note on randomness

scikit-image's `medial_axis` breaks ties at random when no `random_state` is given. Re-running `bin2skel.py` or `bin2skel2.py` can therefore change a few skeleton pixels; graph node counts changed by 1–8 in our tests.
- The original and cleaned scripts give identical results when seeded the same way.
- Regenerated graphs give the same growth rates to within the fit error: 1.47 × 10⁻⁷ mm³/h and 11.26 × 10⁻⁵ mm²/h.
