# 07 · Lattice schematics

**Figure:** Fig. 2J. It compares ossicle lattices with a symmetric ordered lattice and three ways of breaking its symmetry.

| Script | Panel | Output |
|---|---|---|
| `hexagonal_lattice2graph.py` | Symmetric ordered lattice; radial asymmetry. A hexagonal lattice clipped to a disc, with the origin marker off-centre. | the lattice graph (`.gpickle`, under `$OSSICLE_OUT/07_lattice_schematics/graphs_extracted/`); the figure was saved by enabling the commented `plt.savefig("hex1.png")` line |
| `disorder_graph.py` | Polygonal disorder. Voronoi cells of a randomly jittered hexagonal point lattice; every run differs. | `disordered_lattice.png` |
| `hyperbolic_tiling.py` | Translational asymmetry. A {7,3} hyperbolic tiling whose cells shrink towards the rim. | `hyp_graph1.png` |

- **Environment:** `environment-analysis.yml`. `hyperbolic_tiling.py` needs the `hypertiling` package.
- **Inputs:** none.
- **Outputs:** PNGs are written to the current working directory.
- **Assembly:** the published panel was assembled from these drawings in a vector editor.
