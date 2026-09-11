# 01 · Micro-CT morphometrics of ossicles

**Figures:**
- **Main:** Fig. 1D, Fig. 2A, Fig. 2C–H.
- **Supplementary:** Fig. S2C–D, Fig. S3D, Fig. S3F–H, Fig. S5I.

**Inputs:** a whole juvenile was scanned by micro-CT at 0.82 µm voxels and segmented into one closed surface mesh per ossicle. Each mesh is then:
1. skeletonized into a graph of nodes and edges;
2. aligned with PCA and cut to its base;
3. measured for curvature, topological depth, branch lengths, holes, bifurcation angles, medial thickness and geodesic distances.

The methods are described in the SI, "Analysis of micro-CT datasets".

**Environment:** `environment-analysis.yml` (Python 3.7).

**Data:** all paths are relative to `$OSSICLE_DATA/microct/animal_1/extracted_ossicles/` unless noted; see `data/README.md`.

## Run order

**0. Manual segmentation (Dragonfly 2022.2).** Intensity thresholding and seeded watershed, then each ossicle is exported as a closed mesh to `mesh/*.stl`.

**1. Conversion and first skeletonization.**

| Script | Reads | Writes |
|---|---|---|
| `pipeline/stl2vtk.py` | `mesh/*.stl` | `vtk/` |
| `pipeline/skeletonization/skeletonize_code.py` | `vtk/` | `skeleton/<ossicle>/`: skeleton voxels and medial thickness (ITK thinning; see *Skeletonization* below) |
| `pipeline/skeletonization/skel2graph.py` | `skeleton/<ossicle>/` | skeleton graphs `*_G.gpickle` (NetworkX; medial thickness on nodes and edges; branches shorter than 15 voxels trimmed) |

**2. Ossicle order and centres.**

| Script | Writes |
|---|---|
| `renders/centroid_calc.py` | `centroids.npy`, `start_points.npy` (working directory; copy them to `Data Calculated/`) |
| `renders/relabel_with_coordinate.py` | `centroid_sorted_indices.npy` (same) |

**3. Alignment and base graphs.**

| Script | Reads | Writes |
|---|---|---|
| `pipeline/mesh_align.py` | meshes | PCA-aligned meshes in `aligned/stl_files/` |
| `pipeline/stl2vtk.py`, then step 1 again | `aligned/stl_files/` | `aligned/vtk/`, `aligned/skeleton/`, `*_G.gpickle` |
| `pipeline/base_isolation.py` | aligned meshes | pillars removed, base surface fitted: `aligned/base/vtk/` |
| `fields/curvature/curvature_calc.py` | base surfaces | libigl principal curvatures: `aligned/curvature/`, `avg_mean_curvature50.npy` |
| `pipeline/extract_base_graphs.py` | aligned graphs | base graphs with their origin: `*_base_G.gpickle` |

**4. Data producers.** These write to `Data Calculated/`.

| Script | Writes |
|---|---|
| `fields/length/length_by_level.py` | `edge_level_length` (edge length vs edge depth) |
| `fields/holes/hole_dist_area.py` | `hole_distance_area` (minimum-cycle-basis holes) |
| `fields/polygonality/polygonality_level_poly.py` | `level_poly_data` |
| `fields/thickness/thickness_with_dist_lumped.py`, then `thickness_with_dist_lumped_scaled.py` | `thickness_data_lumped`, then `*_scaled` |

Two more producers only write when a commented line is enabled:
- `fields/topology/level_count_plot.py` writes `level_count_data`.
- `fields/angles/all_angle_plots2.py` writes the angle tables (`$OSSICLE_DATA/tables/angle`) through a commented `np.save`.

The `*_foot` variants do the same for the adult foot-pad graphs made in `02_image_morphometrics_2d/footpad`:
- `length_by_level_foot.py`, `length_field_foot.py`
- `polygonality_level_poly_foot.py`
- `thickness_with_distance_lumped_foot.py`, `thickness_with_dist_lumped_scaled_foot.py`
- `topology_plot_foot.py`, `vector_plot_foot.py`

They read `$OSSICLE_DATA/footpad/graphs/` and write into `graphs/`. Move the results into the sub-folders the plotters read: `len_data/`, `poly_data/`, `thick_data/`, `topo_data/`.

**5. Figures.**

| Panel | Script(s) |
|---|---|
| 1D | `renders/array_mesh_plot.py` (volume-sorted spiral of all ossicles) |
| 2A left | `fields/curvature/curvature_calc.py` (mean-curvature maps on base surfaces) |
| 2A right | `renders/whole_animal_coordinate_curvature.py` (`ani_curv_map_scale.png`) |
| 2C, S2C–D | `fields/topology/topology_plot.py`, `vector_plot.py` and their `_foot` versions (node-depth maps) |
| 2E | `fields/topology/level_count_plot.py` |
| 2D, S3D | `fields/length/length_fields.py`, `length_field_foot.py` (maps); `plot_length_by_level.py` (plot) |
| 2F | `fields/holes/hole_fields.py` (maps); `dist_area_plot.py` (plot) |
| 2G, S2C | `fields/angles/angle_fields.py` (maps); `all_angle_plots2.py` (plot and statistics) |
| 2H | `fields/thickness/thickness_fields.py` (maps); `plot_thickness_scaled_small.py` (scaled inset). The script that drew the unscaled main plot was not archived. Its data come from `thickness_with_dist_lumped.py`, where path length is the shortest path along the skeleton graph. |
| S3F | `fields/size_scaling/plot_data.py` |
| S3G | `fields/geodesic/geodesic_rad_diff.py` (geodesic minus radial distance; pygeodesic); `geodesic.py` draws the geodesic distance maps |
| S3H | `fields/polygonality/polygonality_fields.py` (maps); `level_poly_plot.py` (plot) |
| S5I | `fields/size_scaling/sa_vol.py` (pooled over the juveniles listed in `data/README.md`) |

Other shipped renders whose figure panel is not confirmed:
- `renders/whole_animal_coordinate.py`
- `renders/cross_section_image.py`
- `renders/phase_cartoon.py`

Fig. 1E and Fig. 2B were drawn in Excel from the per-ossicle table `ossicle_data_list.xlsx`, which is part of the data.

## Reproduced numbers

These were checked by running the released code on the original data:
- Fig. 2G bifurcation angles: 106.5 ± 22.3° (pillared tables), 107.7 ± 24.4° (other plates), 116.5 ± 21.2° (foot pad).
- Fig. S5I: log–log slope 1.10.

## Notes

**Skeletonization.** This step builds on the Coral3D morphometrics code of Ramírez-Portilla *et al.* (2022, *Front. Mar. Sci.* 9:955582). It calls a compiled ITK thinning program (`skel_itk`, ITK 5.2.1 with the Thickness3D module).
- That code is not part of this repository, so `skeletonize_code.py` and `skel2graph.py` do not run from this repository alone.
- Our changes to it are recorded in `pipeline/skeletonization/coral3d_modifications.patch`.
- `graph2mesh.py` is an inspection utility for the skeleton graphs.

**Hand-selected inputs.** Every field script reads the hand-made selection lists in `Data Calculated/`:
- `loc_file_id*.txt`
- `selected_table.npy`, `selected_other_flat.npy`
- `centroids.npy`, `avg_mean_curvature.npy`

**One ossicle, or one ossicle type, per run.**
- `skeletonize_code.py`, `skel2graph.py`, `extract_base_graphs.py` and `thickness_with_dist_lumped.py` process one ossicle per run; change the index to run others.
- The other-plate runs of `hole_dist_area.py` and `polygonality_level_poly.py`, and the table run of `length_by_level.py`, were made by editing the ossicle-type selection. Their headers say which lines.

**Windows paths.** Some paths are joined with `\\`, so these scripts run as written on Windows.

**Radius mismatch in curvature files.** `curvature_calc.py` uses a neighbourhood radius of 1 edge length (per-point maps). The whole-animal averages are stored in files named `*50`, which matches the radius of 50 edge lengths given in the SI.

**Output locations.** Figures are saved to the working directory, except that `plot_data.py` and `plot_length_by_level.py` save to `$OSSICLE_OUT/01_microct_morphometrics/`. Large renders are saved as 10000 × 10000 px screenshots.
