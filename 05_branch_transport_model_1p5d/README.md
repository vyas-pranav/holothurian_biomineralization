# 05 · 1.5D advection–reaction–diffusion model of single-branch growth

**Figures:** Fig. 6A, Fig. 6B, Fig. S18, Fig. S19, Supplementary Video SV8 (advection–reaction–diffusion clips).

A growing ossicle branch is modelled on one axial coordinate $x \in [0, \ell(t)]$, with two coupled fields:
- $c_f(x,t)$: mobile precursor, carried along the branch by effective diffusion $D_f$ and motor-driven advection $v_m$;
- $c_d(x,t)$: deposited mineral, accumulating at rate $k_d$, which sets the local branch width $w(x,t)$.

Mass enters at the cell body at rate $\dot m_0$. A tip growth zone of length $l_g$ collects precursor at rate $\epsilon$ and extends the branch at rate $k_t$ (SI Eqs. 3–8). The equations are solved with explicit finite differences: upwind advection, central diffusion and forward-Euler time stepping.

## Scripts and run order

Scripts write to `$OSSICLE_OUT/05_branch_transport_model_1p5d/` (default `outputs/…`) and need no input data.

| Step | Script | Produces | Figure |
|---|---|---|---|
| 1 | `para_sweep_non_dim.py` | Non-dimensional sweep over a 10 × 10 log grid of $(Pe_f^{-1}, Da_d) \in [0.01, 10]^2$, 50,000 s per run; pickled snapshots per run | data for 6B, S19 |
| 2 | `plot_2D_ND.py` | Final branch profiles drawn at their $(Pe_f^{-1}, Da_d)$ positions (`geometry_profiles_Da_d_invPe_loglog_small.eps`). Set `my_root_folder` to the sweep folder from step 1. | 6B, S19 |
| 3 | `para_sweep_run.py` | Dimensional sweep ($D_f$, $k_d$ ∈ linspace(0.01, 0.1, 5), $v_m$ ∈ linspace(0.01, 0.5, 5), $\epsilon = 10\,v_m$). Saves a PNG frame and a snapshot every 100 s. | data for 6A, SV8 |
| 4 | `3d_time_growth.py` | 3D render of one run from step 3 at several time points (the published panel used $D_2 = 0.055$, $k_4 = 0.055$, $v_m = 0.255$) | 6A |
| 5 | `../06_branching_lattice_model_2d/plotting/video_from_frames.py` | Movie from the PNG frames of a run from step 3 | SV8 |
| — | `phase_plot_final_length_v2.py` | Terminal length $L$ from the steady-state transcendental condition (SI Eqs. 19–21), solved with `scipy.optimize.root_scalar(method="brentq")` on a 200 × 200 grid | S18 |

## Runtime and storage

Steps 1 and 3 are long: 100 and 125 runs of 5 × 10⁶ time steps each.

`para_sweep_non_dim.py` saves pickles only (`save_plot=False`). `para_sweep_run.py` saves a PNG frame for every 100 s of every run, as in the original sweep.

## Code symbols and SI notation

| Code | SI | Default (SI Table 3) |
|---|---|---|
| `D2` | $D_f$, effective diffusion of transported precursor | 0.01 µm² s⁻¹ |
| `k4` | $k_d$, deposition rate | 0.1 s⁻¹ |
| `k5` | $k_t$, tip extension rate constant | 100 s⁻¹ |
| `epsilon` | $\epsilon$, influx into the growth zone | 100 µm s⁻¹ |
| `vm` | $v_m$, advection along microtubules | 0.5 µm s⁻¹ |
| `m_dot_0` | $\dot m_0$, injected mass rate per branch | 0.01 pg s⁻¹ |
| `w0`, `rho`, `l_g`, `csat` | $w_0$, $\rho$, $l_g$, $c_{sat}$ | 1 µm, 2.71 pg µm⁻³, 0.5 µm, 0 |
| `c2`, `c4`, `cg` (dimensional) / `cf_bar`, `cd_bar`, `cg_bar` (non-dimensional) | $c_f$, $c_d$, $c_g$ | |
| `Pe_inv`, `Pe2_inv` | $Pe_f^{-1} = D_f/(v_m w_0)$ | |
| `Da_d`, `Da4` | $Da_d = k_d w_0/v_m$ | |
| `Pe2`, `Da4` in `phase_plot_final_length_v2.py` | $Pe_f$ (not inverted), $Da_d$ | |

## Notes

- **Pickles.** Snapshots are pickled `ReactionDiffusionSimulator` objects. Load them from a script that defines the same class, as `plot_2D_ND.py` and `3d_time_growth.py` do, and run the sweep and the plot in the same environment.
- **Changes from the originals.** Paths, headers, font fallback, the paper notation in the S18 axis labels, a missing `import traceback`, sorted folder listing in the Fig. 6B/S19 plotter, and `save_pkl=True` in the non-dimensional sweep. See `PROVENANCE.md` for the original files.
