# Holothurian ossicle lattices: analysis and model code

This repository holds the code behind the figures of

> Vyas P., Brannon C., Formery L., Lowe C. J., Prakash M. **Cellular construction of topologically complex biomineral lattices in holothurians.** *Submitted to Cell.*

Juvenile sea cucumbers (*Apostichopus parvimensis*) build hundreds of calcite ossicles, each a finite lattice of branches that fuse into closed loops.

The repository contains four kinds of code:
- Micro-CT and imaging analyses that measure ossicle geometry and topology.
- Analyses of how ossicles grow and how vesicles move along their branches.
- Two minimal models:
  - a 1.5D advection–reaction–diffusion model of single-branch growth;
  - a 2D self-closing branching model of whole-ossicle lattices.
- Scripts for schematic panels.

The code is published as it was used for the paper, organized by task. It was only cleaned (paths, headers, documentation) and never changed in what it computes; see [How this code was prepared](#how-this-code-was-prepared).

## Repository map

| Folder | What it contains | Environment |
|---|---|---|
| [`01_microct_morphometrics`](01_microct_morphometrics) | Micro-CT meshes → skeleton graphs → curvature, topology, lengths, holes, angles, thickness, geodesics | analysis |
| [`02_image_morphometrics_2d`](02_image_morphometrics_2d) | 2D image skeletons: adult foot-pad ossicles; volume growth of a live ossicle | analysis |
| [`03_vesicle_transport`](03_vesicle_transport) | Kymograph tracking of vesicles, MSD, scaling exponents, nocodazole comparisons | analysis |
| [`04_drug_perturbations`](04_drug_perturbations) | Fraction of ossicles growing under cytoskeletal drugs | analysis |
| [`05_branch_transport_model_1p5d`](05_branch_transport_model_1p5d) | 1.5D advection–reaction–diffusion model of branch growth | models |
| [`06_branching_lattice_model_2d`](06_branching_lattice_model_2d) | 2D self-closing branching (graph) model of ossicle lattices | models |
| [`07_lattice_schematics`](07_lattice_schematics) | Ordered, disordered and hyperbolic lattice schematics | analysis |

Each folder has a README with the run order, inputs, outputs and any manual steps (Dragonfly, Fiji).

## Figure legend

| Figure | Panel content | Code |
|---|---|---|
| 1C | Segmented ossicles on the whole animal, coloured by type | rendering exported from Dragonfly; related script `01_microct_morphometrics/renders/whole_animal_coordinate.py` (panel role unconfirmed) |
| 1D | All ossicles of one juvenile, aligned and size-sorted | `01_microct_morphometrics/renders/array_mesh_plot.py` |
| 1E | Ossicle type counts and volumes | Excel chart from the per-ossicle table (`ossicle_data_list.xlsx`, part of the data) |
| 2A | Mean curvature of table-ossicle bases and whole-animal average | `01_microct_morphometrics/fields/curvature/curvature_calc.py`, `renders/whole_animal_coordinate_curvature.py` |
| 2B | Central symmetry of table ossicles | Excel chart from the per-ossicle table (`ossicle_data_list.xlsx`) |
| 2C, 2E | Node-depth maps and node-depth distributions | `01_microct_morphometrics/fields/topology/` |
| 2D | Branch length vs edge depth | `01_microct_morphometrics/fields/length/` |
| 2F | Hole area vs distance from centre | `01_microct_morphometrics/fields/holes/` |
| 2G | Bifurcation angles vs node depth | `01_microct_morphometrics/fields/angles/` |
| 2H | Medial thickness vs path length along the skeleton | maps and data: `01_microct_morphometrics/fields/thickness/`; inset: `fields/thickness/plot_thickness_scaled_small.py`; the script that drew the main plot was not archived |
| 2I | Skeletonized adult foot-pad ossicle | `02_image_morphometrics_2d/footpad/` |
| 2J | Lattice symmetry-breaking schematics | `07_lattice_schematics/` |
| 3E–F | Branch extension rates | tip lengths measured in Fiji; fit not archived |
| 3H, 3J | Volumetric growth of a front-plate ossicle and segmentation overlays | `02_image_morphometrics_2d/growth_series/bin2skel.py` → `sa_vol_calc.py` |
| 4G | Nucleus distances from ossicle centres | distances measured in Fiji; plotting script not archived |
| 5C | Fraction of ossicles growing under drugs | `04_drug_perturbations/plot_growth_fraction.py` |
| 5E–G | Ensemble MSD, scaling exponent α and K_α under nocodazole | `03_vesicle_transport/` pipeline → `tracks_processing_fit_v3.py` → `plot_comparative_data.py` (SETTING `fig5E-G_mixed_conditions`), `plot_compare_data_sig.py` |
| 6A | Simulated growing tip (3D) | `05_branch_transport_model_1p5d/para_sweep_run.py` → `3d_time_growth.py` |
| 6B | Branch profiles in the (Pe_f⁻¹, Da_d) plane | `05_branch_transport_model_1p5d/para_sweep_non_dim.py` → `plot_2D_ND.py` |
| 6D–G | Lattices grown from 2, 4, 5 and 7 seed branches | `06_branching_lattice_model_2d/sweeps/seed_edge_sweep3.py` |
| 6H, 6J | Lattices resembling other holothurians (α = 0.050, β = 0.306; β = −0.1) | single runs of the 2D model (`06_branching_lattice_model_2d/`); the exact script version is not recorded |
| 6I | Lattice at α = 0.035, β = 0.073 | `06_branching_lattice_model_2d/sweeps/cell_pair_force_sweep3.py` (a point of the S22 grid) |
| 6K | Asymmetric lattice from moving sclerocytes | `06_branching_lattice_model_2d/model/growth_sim_v18_asymm.py` |
| 6L–M | Lattices for different D and k | `06_branching_lattice_model_2d/sweeps/lambda_D_sweep3.py` → `plotting/replot_final_pruned.py` → `plotting/plot_grid_lambda_D_param.py` |
| S2C–D | Bifurcation and budding nodes | `01_microct_morphometrics/fields/angles/`, `fields/topology/` |
| S3A | Ossicles coloured by volume | `01_microct_morphometrics/renders/whole_animal_coordinate.py` (panel role unconfirmed) |
| S3B–D | Foot-pad segmentation, skeleton and edge lengths | `02_image_morphometrics_2d/footpad/`, `01_microct_morphometrics/fields/length/` |
| S3F | Volume and surface area vs number of holes; holes by type | `01_microct_morphometrics/fields/size_scaling/plot_data.py` |
| S3G | Geodesic minus radial distance | `01_microct_morphometrics/fields/geodesic/geodesic_rad_diff.py` |
| S3H | Number of hole sides vs hole depth | `01_microct_morphometrics/fields/polygonality/` |
| S4C | Calcite facet orientations from SEM | orientation analysis script not archived |
| S5A | Growth series overlays | `02_image_morphometrics_2d/growth_series/` |
| S5I | Volume vs surface area (log-log) | `01_microct_morphometrics/fields/size_scaling/sa_vol.py` |
| S10G | Microtubule intensity profiles | paths traced as in `03_vesicle_transport`; profile plot not archived |
| S10L | Bolus velocities (polar plot) | positions tracked manually in Fiji; velocity script not archived |
| S11A (inset) | Filopodia count over time | counted manually |
| S14F | Fraction of ossicles growing vs nocodazole dose | made with a variant of the Fig. 5C script; not archived |
| S15C–I | Vesicle kymographs, tracks, speeds, MSD | `03_vesicle_transport/s15_front_plate_2024/kymograph_track_multi2.py` → `tracks_processing_fit.py` (SETTING `S15`) |
| S16 | Nocodazole time course of vesicle MSD | `03_vesicle_transport/` pipeline → `plot_comparative_data.py` (default SETTING) |
| S17 | Ossicle morphospace schematic renders | `01_microct_morphometrics/renders/phase_cartoon.py` (panel role unconfirmed) |
| S18 | Terminal branch length over (Da_d, 1/Pe_f) | `05_branch_transport_model_1p5d/phase_plot_final_length_v2.py` |
| S19 | Branch geometry profiles over (Da_d, Pe_f⁻¹) | `05_branch_transport_model_1p5d/plot_2D_ND.py` |
| S20A | Growth sequence of a simulated lattice | `06_branching_lattice_model_2d/plotting/plot_growth_series.py` |
| S20B | Primary hole areas, micro-CT vs simulation | analysis script not archived |
| S21–S23 | α–β phase plots | `06_branching_lattice_model_2d/sweeps/cell_pair_force_sweep3.py` → `plotting/plot_grid_alpha_beta_param.py` |
| S24 | k–D phase plot | `06_branching_lattice_model_2d/sweeps/lambda_D_sweep3.py` → `plotting/replot_final_pruned.py` → `plotting/plot_grid_lambda_D_param.py` |
| S25 | Simulated branch lengths and hole areas vs depth | analysis script not archived |
| S26 | Angle-noise series | `06_branching_lattice_model_2d/sweeps/angle_noise_sweep3.py` → `plotting/plot_grid_single_param.py` |
| SV8 | Simulation movies | `05_branch_transport_model_1p5d/para_sweep_run.py` frames and 2D model frames → `06_branching_lattice_model_2d/plotting/video_from_frames.py` |

All other panels are microscopy images, renderings exported from Dragonfly, or schematics.

## Getting started

### 1. Create the environment

There are two conda environments, pinned to the versions the code actually ran with:

```bash
conda env create -f environment-models.yml     # folders 05, 06 (Python 3.13)
conda env create -f environment-analysis.yml   # folders 01-04, 07 (Python 3.7)
```
- **Record of the original setup.** `environment-analysis.win-64.lock.txt` lists the exact Windows environment the analyses were run in.
- **Apple silicon.** Python 3.7 has no native build for Apple-silicon Macs. Create the analysis environment with `CONDA_SUBDIR=osx-64` so it runs under Rosetta.

### 2. Point the scripts at data and an output folder

```bash
export OSSICLE_DATA=/path/to/ossicle-data   # inputs (see data/README.md)
export OSSICLE_OUT=/path/to/results         # outputs, default ./outputs
```
The models need no data. For example:
```bash
conda activate ossicle-models
python 05_branch_transport_model_1p5d/phase_plot_final_length_v2.py
python 06_branching_lattice_model_2d/model/growth_sim_v18_main.py
```

## Data availability

The repository contains code only. The micro-CT, imaging and tracking data are available from the lead contact on request. [`data/README.md`](data/README.md) describes the folder layout the scripts expect.

## How this code was prepared

- **Source.** Every file was copied from the authors' working repository or project folders. [`PROVENANCE.md`](PROVENANCE.md) lists the original location and SHA-256 of each file.
- **Edits allowed.** Changes are limited to:
  - absolute paths, replaced by `OSSICLE_DATA` / `OSSICLE_OUT`;
  - file headers and documentation;
  - IPython magics and missing imports;
  - deterministic ordering of folder listings;
  - font fallbacks and paper notation in plot labels;
  - for the models only, saving graphs with `pickle` in place of the removed networkx gpickle helpers, plus an optional random seed.
- **Verification.** Each cleaned file was checked against its original by comparing the parsed code with comments and docstrings removed; only the edits above remain. Where the code runs without data, original and cleaned versions produced bit-identical outputs from the same inputs.
- **Randomness.** The 2D branching model and the disordered-lattice schematic are stochastic, and the published runs were not seeded. Each script now has an optional `SEED`.

## Third-party code and credits

- **Coral3D.** The micro-CT skeletonization builds on the Coral3D morphometrics code of Ramírez-Portilla *et al.* (2022), *Frontiers in Marine Science* 9:955582. That code is not redistributed here; our modifications are recorded in `01_microct_morphometrics/pipeline/skeletonization/coral3d_modifications.patch`.
- **Other tools.** The analyses use NumPy, SciPy, Matplotlib, pandas, NetworkX, PyVista/VTK, vedo, libigl, pygeodesic, scikit-image, OpenCV, shapely, ITK, and Fiji (including the KymoResliceWide plugin).

## License and citation

- **License.** GPL-3.0-or-later; see [`LICENSE`](LICENSE).
- **Citation.** Please cite the article; see [`CITATION.cff`](CITATION.cff).
