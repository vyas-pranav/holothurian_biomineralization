# 03 · Vesicle transport along ossicle branches

**Figures:** Fig. 5D–G, Fig. S15C–I, Fig. S16A–E.

- **Imaging:** membrane-labelled vesicles moving within the cytoplasmic sheath were imaged live (confocal time-lapses).
- **Tracking:** their motion along curved paths on the branches was converted into kymographs and traced into position–time tracks.
- **Analysis:** the tracks are analysed by mean-squared displacement (MSD). The single-regime fit gives the scaling exponent α and the effective transport coefficient K_α = 0.5 × 10^b, where b is the intercept (SI, "Vesicle tracking for Figure 5").

**Environment:** `environment-analysis.yml` (Python 3.7).

**Data:** `$OSSICLE_DATA/vesicles/` (see `data/README.md`).

## Workflow for the nocodazole datasets (Fig. 5E–G, Fig. S16)

1. **Fiji.** Make a maximum-intensity projection of the time-lapse.
2. **`draw_track.py`** (interactive, Tk). Paint over each streak along which vesicles move, using a 15-px brush, and save one binary mask per path.
3. **`apply_mask.py`.** Multiplies every frame by the path mask.
4. **Fiji.** Apply a median filter, then run **KymoResliceWide** along the path. This gives the `Reslice (AVRG) of …` and `Reslice (MAX) of …` kymographs.
5. **`kymograph_track_multi3.py`** (interactive, Tk). Paint one region per vesicle trace, then choose *Process All Tracks*. This writes one-pixel tracks and the region masks.
6. **`recalc_tracks_from_masks.py`.** Recomputes the tracks strictly inside the saved masks, for both kymographs, into `tracks_from_masks_{AVRG,MAX}/`. Set `kymo_root` to one `track_<k>/kymograph` folder.
7. **`pool_tracks_v2.py`.** Pools all paths of one movie into `pooled_{AVRG,MAX}/tracks`.
8. **`tracks_processing_fit_v3.py`.** `SETTING` selects the dataset; the default reproduces the original run.
   - Settings: 0.0598188 µm/px, 1.42 s/frame, fit for τ > 3 s.
   - Computes positions, speeds and time-averaged MSD, with single-regime, two-regime and diffusion-plus-drift fits.
   - Writes per-track tables and plots into `plots3/`. The folder is recreated on every run.
9. **`plot_comparative_data.py`.** Ensemble MSD (median and IQR), with box plots of α and K_α across conditions. `SETTING` selects the panel:
   - `"S16A-C_1uM_time_series"` (default): Fig. S16A–C.
   - `"fig5E-G_mixed_conditions"`: Fig. 5E–G.
10. **`plot_compare_data_sig.py`.** Track-level tests for the Fig. 5F–G statistics, comparing each control with its nocodazole condition: Mann–Whitney U, Hodges–Lehmann shift, geometric-mean ratio, Cliff's delta with bootstrap CIs, Benjamini–Hochberg q.

Steps 9–10 read the fit outputs from `<pooled>/plots2`, the folder name used when the paper figures were made. Copy or rename the `plots3` output of step 8 to `plots2`. Recomputing the statistics this way reproduces the published values; only the bootstrap confidence intervals differ, as expected for a bootstrap.

## Front-plate dataset (Fig. S15)

The Fig. S15 tracks were made from the `Reslice (AVRG) of images.tif` kymographs with `s15_front_plate_2024/kymograph_track_multi2.py`. This earlier tracer re-traces all 42 saved tracks we tested exactly.

Run `s15_front_plate_2024/tracks_processing_fit.py` with `SETTING = "S15"`, which uses 0.1 µm/px, 0.5901 s/frame and a fit for τ > 1 s.
- It reproduces Fig. S15I: 93 tracks, α from 0.044 to 1.813, mean 0.733.
- Its regenerated track, speed and MSD plots are byte-identical to the originals.
- `tracks_processing_fit_v3.py` does **not** reproduce S15I, because it fits τ > 3 s and resamples with splines.
