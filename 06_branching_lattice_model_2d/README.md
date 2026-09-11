# 06 · 2D self-closing branching model of ossicle lattices

**Figures:** Fig. 6D–M, Fig. S20A, Fig. S21–S24, Fig. S26, Supplementary Video SV8 (lattice part).

The ossicle is a NetworkX graph whose active tips grow by a fixed step Δs per iteration. The direction of growth follows the pseudo-force F = α F_cell + η F_pers + β F_pair:
- **F_cell:** away from the sclerocytes;
- **F_pers:** persistence along the current direction;
- **F_pair:** attraction between paired tips.

Timing and topology changes follow these rules:
- **Positional field.** The sclerocytes (N_c = 4) set a field c(x) = Σ_j S/(2πD) K₀(√(k/D)|x − x_j|).
- **Bifurcation.** A tip bifurcates at 108° (plus angle noise) once the interval τ_rep = max[τ_min, τ_max ln c(x)/κ] has passed (SI Eq. 48).
- **Fusion and budding.** Tips that meet in a search box fuse, and may bud a new tip.
- **Stopping.** Tips stop when c drops below a threshold.

The algorithm and the SI Table 4 defaults are described in the SI, "2D self-closing branching model".

## Scripts

| Panel | Scripts (run order) |
|---|---|
| Fig. 6D–G, S20A | `sweeps/seed_edge_sweep3.py` (2, 4, 5 and 7 seed branches); then `plotting/plot_growth_series.py` on the 4-branch run for S20A |
| Fig. 6I, S21–S23 | `sweeps/cell_pair_force_sweep3.py` (α–β grids; Fig. 6I is the S22 point α = 0.034966, β = 0.073487); then `plotting/plot_grid_alpha_beta_param.py` |
| Fig. 6K | `model/growth_sim_v18_asymm.py` (moving sclerocytes, rotated seeds) |
| Fig. 6L–M, S24 | `sweeps/lambda_D_sweep3.py` (10 × 10 grid, k = 0.001–0.1 h⁻¹, D = 1–100 µm² h⁻¹); then `plotting/replot_final_pruned.py`, then `plotting/plot_grid_lambda_D_param.py`. Fig. 6L–M are tiles of this grid. |
| Fig. S26 | `sweeps/angle_noise_sweep3.py` (angle noise 0.1–1.0 rad); then `plotting/plot_grid_single_param.py` |
| SV8 | frames from the runs above, assembled with `plotting/video_from_frames.py` |
| Reference | `model/growth_sim_v18_main.py`: standalone implementation with the SI Table 4 defaults |

`sweeps/ossicle_class3.py` holds the model class that all sweep scripts import. It is the April-2025 version of the model and has the same growth and timing rules as `growth_sim_v18_main.py`. Fig. 6H (α = 0.050, β = 0.306) and Fig. 6J (β = −0.1) are single model runs; the script version used for them is not recorded.

## Code symbols and SI notation

| Code | SI symbol | Meaning | Default (Table 4) |
|---|---|---|---|
| `alpha`, `beta`, `eta` | α, β, η | pseudo-force weights | 0.0075, 0.001, 1 |
| `step_length` | Δs | growth per step | 0.5 µm |
| `angle`, `angle_noise`, `time_noise` | θ, σ_θ, σ_t | bifurcation angle and noise | 108°, 0.01 rad |
| `morphogen_release_rate` (`Q`) | S | release rate | 100 pg h⁻¹ |
| `morphogen_diffusion_coefficient` (`D`) | D | diffusion coefficient | 10 µm² h⁻¹ |
| `morphogen_decay_rate` (`lambda`, `lambda0`) | k | decay rate | 0.01 h⁻¹ |
| `morphogen_rep_max` / `morphogen_rep_min` / `morphogen_rep_scale` | τ_max, τ_min, κ | replication-interval rule | |
| `box_size`, `pair_thresh_length` | | fusion search box, tip pairing distance | ~10 Δs, 10 µm |
| `seed_size`, `initial_branch_count` | | seed | 0.1 µm, 2–7 |
| `cell_translation_step` | | sclerocyte motion per step | |

## Environment

- **Models env.** `environment-models.yml` (Python 3.13). The published runs used Python 3.7 and networkx 2.6. In our tests the model gives bit-identical graphs in both environments for the same random seed.
- **Graph files.** Graphs are saved with `pickle`, because networkx 3 removed the gpickle helpers.
- **Grid plotters.** The `plot_grid_*.py` scripts use Pillow < 10 APIs, so run them in `environment-analysis.yml`.

## Randomness and verification

- **Randomness.** Bifurcation angles, replication times and seed directions have small random perturbations (`np.random`). The published runs were not seeded. Every runnable script has a `SEED` setting; its default, `None`, keeps the original behaviour.
- **Checks passed.**
  - Original and cleaned code give bit-identical final graphs for the same seed: `growth_sim_v18_main`, `growth_sim_v18_asymm`, and a 4-branch `seed_edge_sweep3` run. This holds on Python 3.7 and on Python 3.13.
  - Reduced sweep and plotting runs produce byte-identical outputs (61 of 61 files).
