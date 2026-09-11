#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Compare vesicle MSD statistics across conditions (ensemble MSD, alpha and K_alpha).

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5E–G (SETTING "fig5E-G_mixed_conditions"); Fig. S16A–C (default)
What it does  : Reads track_analysis.xlsx/.csv and data/*.csv of each pooled folder (from
                <pooled> itself or <pooled>/plots2) and draws, per condition: box + jitter
                plots of alpha, K_alpha, D and v, the ensemble MSD (median and IQR of the
                tracks interpolated on a common log-tau grid), speed ECDFs, two-regime plots,
                and writes summary_metrics_by_condition.csv.
Inputs        : DATA_ROOT/vesicles/confocal_2025/<...>/pooled_AVRG/plots2/ (fit outputs; the
                plots3 folder written by tracks_processing_fit_v3.py is not searched)
Outputs       : OUT_ROOT/03_vesicle_transport/Comparison plots/<save_name>/*.png, *.csv
Environment   : environment-analysis.yml (Python 3.7)
Run           : python plot_comparative_data.py   (set SETTING)
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
# --- USER PATHS -------------------------------------------------------------
# Point OSSICLE_DATA / OSSICLE_OUT at your copies (see data/README.md).
import os
from pathlib import Path
try:
    _HERE = Path(__file__).resolve()
except NameError:  # interactive (e.g. Spyder cell) execution
    _HERE = Path.cwd().resolve() / "_"
REPO_ROOT = next((p for p in _HERE.parents if (p / "CITATION.cff").exists()), _HERE.parent)
DATA_ROOT = Path(os.environ.get("OSSICLE_DATA", REPO_ROOT / "data"))
OUT_ROOT = Path(os.environ.get("OSSICLE_OUT", REPO_ROOT / "outputs"))
# ----------------------------------------------------------------------------


# === Styling to match your pipeline ===
mult_factor = 1.5
plt.rcParams['font.family'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12 * mult_factor
plt.rcParams['axes.titlesize'] = 12 * mult_factor
plt.rcParams['axes.labelsize'] = 11 * mult_factor
plt.rcParams['xtick.labelsize'] = 10 * mult_factor
plt.rcParams['ytick.labelsize'] = 10 * mult_factor
plt.rcParams['legend.fontsize'] = 7 * mult_factor
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

# ------------------------- IO helpers -------------------------

def _resolve_plot_folder(folder_path: str) -> str:
    """
    Accept either a 'pooled_*' dataset folder OR its 'plots2' subfolder.
    Return the folder that actually contains track_analysis.* (usually plots2).
    """
    cand = []
    # if they passed plots2 already
    cand.append(folder_path)
    # if they passed the dataset root, check plots2
    cand.append(os.path.join(folder_path, "plots2"))

    for c in cand:
        if os.path.isdir(c):
            if os.path.exists(os.path.join(c, "track_analysis.xlsx")) or \
               os.path.exists(os.path.join(c, "track_analysis.csv")):
                return c
    raise FileNotFoundError(f"Could not find track_analysis.xlsx/csv under {folder_path} or {folder_path}/plots2")

def _read_track_analysis(plot_folder: str) -> pd.DataFrame:
    xlsx = os.path.join(plot_folder, "track_analysis.xlsx")
    csv  = os.path.join(plot_folder, "track_analysis.csv")
    if os.path.exists(xlsx):
        return pd.read_excel(xlsx)
    elif os.path.exists(csv):
        return pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Missing track_analysis.xlsx/csv in {plot_folder}")

def _read_optional_csv(plot_folder: str, rel_path: str) -> Optional[pd.DataFrame]:
    p = os.path.join(plot_folder, rel_path)
    return pd.read_csv(p) if os.path.exists(p) else None

def _common_parent(folder_paths: List[str]) -> str:
    try:
        return os.path.commonpath(folder_paths)
    except ValueError:
        return os.path.dirname(folder_paths[0])

# ---------------------- Plotting helpers ----------------------

def _ensure_outdir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# def _color_cycle(n):
#     # discrete, readable palette across conditions
#     cmap = plt.get_cmap('tab20')
#     return [cmap(i % 20) for i in range(n-1)] + [cmap(8)]  # last one distinct

def _color_cycle(n):
    # discrete, readable palette across conditions
    cmap = plt.get_cmap('tab10')
    # choose well separate colors from the colormap
    # return [cmap(i / (n-1)) for i in range(n)]
    return [cmap(i%10) for i in range(n)]

def _box_jitter(ax, data_by_group, labels, colors, ylabel, title="", ylog=False):
    positions = np.arange(1, len(labels)+1)
    bp = ax.boxplot(
        data_by_group, positions=positions, patch_artist=True, showfliers=False
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor('black')
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.5)

    rng = np.random.default_rng(0)
    for pos, vals, c in zip(positions, data_by_group, colors):
        if len(vals) == 0: 
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.25
        ax.plot(np.full(len(vals), pos) + jitter, vals, 'o',
                ms=3, mfc=c, mec='black', alpha=0.9)

    ax.set_xticks(positions)
    # <<< rotate and right-align so long labels don't collide >>>
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.margins(x=0.05)  # small side margins
    # -----------------------------------------------------------

    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    # give bottom a bit more room for slanted labels
    plt.subplots_adjust(bottom=0.2)
    return ax


def _ecdf(x):
    x = np.sort(np.asarray(x))
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def _interp_loggrid(tau, msd, tau_grid):
    """Interpolate in log-log for power-law-like MSD."""
    lt, lm = np.log10(tau), np.log10(msd)
    # guard: need strictly increasing lt; unique values
    lt_u, idx = np.unique(lt, return_index=True)
    lm_u = lm[idx]
    lm_interp = np.interp(np.log10(tau_grid), lt_u, lm_u, left=np.nan, right=np.nan)
    return 10**lm_interp

# ---------------------- Main comparison -----------------------

def compare_datasets(folder_paths: List[str],
                     labels: Optional[List[str]] = None,
                     save_dir: Optional[str] = None):
    """
    folder_paths: list of dataset folders (either 'pooled_*' root or its 'plots2' folder)
    labels:       condition labels in same order; if None, derive from folder tails
    save_dir:     where to save; defaults to <common_parent>/_comparisons
    """
    plot_folders = [_resolve_plot_folder(p) for p in folder_paths]
    if labels is None:
        labels = [os.path.basename(os.path.normpath(p)) for p in folder_paths]
    assert len(labels) == len(plot_folders)

    if save_dir is None:
        save_dir = os.path.join(_common_parent(plot_folders), "_comparisons")
    _ensure_outdir(save_dir)

    # Load per-dataset tables
    datasets = []
    for lab, pf in zip(labels, plot_folders):
        ta = _read_track_analysis(pf)
        # optional tables
        sp_long = _read_optional_csv(pf, os.path.join("data", "speeds_by_step_long.csv"))
        sp_ts   = _read_optional_csv(pf, os.path.join("data", "speeds_time_series.csv"))
        msd_mask= _read_optional_csv(pf, os.path.join("data", "msd_points_masked_tau_gt_TGAP.csv"))
        datasets.append({
            "label": lab,
            "plot_folder": pf,
            "track_analysis": ta,
            "speeds_by_step": sp_long,
            "speeds_ts": sp_ts,
            "msd_masked": msd_mask
        })

    colors = _color_cycle(len(datasets))

    # --------- (1a) α (single-regime) across conditions ---------
    fig, ax = plt.subplots(figsize=(6, 6))
    data_alpha = [pd.to_numeric(d["track_analysis"].get("alpha_single"), errors='coerce').dropna().values
                  for d in datasets]
    _box_jitter(ax, data_alpha, labels, colors, ylabel="α (single-regime MSD)",
                title="Scaling exponent α across conditions", ylog=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alpha_single_box_jitter.png"), dpi=600)
    plt.close(fig)

    # --------- (1b) Kα (single-regime) across conditions (log y) ---------
    fig, ax = plt.subplots(figsize=(6, 6))
    data_Kalpha = [
        pd.to_numeric(d["track_analysis"].get("Kalpha_single"), errors='coerce')
          .replace([np.inf, -np.inf], np.nan).dropna().values
        for d in datasets
    ]
    _box_jitter(
        ax, data_Kalpha, labels, colors,
        ylabel=r"$K_{\alpha}$ ($\mu$m$^2$/s$^{\alpha}$)",
        title=r"$K_{\alpha}$ (single-regime) across conditions",
        ylog=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Kalpha_single_box_jitter.png"), dpi=600)
    plt.close(fig)

    # --------- (2) D across conditions (log y) ---------
    fig, ax = plt.subplots(figsize=(6, 6))
    data_D = [pd.to_numeric(d["track_analysis"].get("D"), errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().values
              for d in datasets]
    _box_jitter(ax, data_D, labels, colors, ylabel="D (μm²/s)",
                title="Diffusion coefficient across conditions", ylog=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "D_box_jitter.png"), dpi=600)
    plt.close(fig)

    # --------- (3) v across conditions (log y) ---------
    fig, ax = plt.subplots(figsize=(6, 6))
    data_v = [pd.to_numeric(d["track_analysis"].get("v"), errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().values
              for d in datasets]
    _box_jitter(ax, data_v, labels, colors, ylabel="v (μm/s)",
                title="Directed velocity across conditions", ylog=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "v_box_jitter.png"), dpi=600)
    plt.close(fig)

    # --------- (4) D vs v scatter per track (log-log), colored by condition ---------
    fig, ax = plt.subplots(figsize=(6, 6))
    for c, d in zip(colors, datasets):
        ta = d["track_analysis"]
        D = pd.to_numeric(ta.get("D"), errors='coerce')
        v = pd.to_numeric(ta.get("v"), errors='coerce')
        mask = (D > 0) & (v > 0) & np.isfinite(D) & np.isfinite(v)
        if mask.any():
            ax.scatter(D[mask], v[mask], s=12, alpha=0.9, label=d["label"],
                       edgecolors='black', linewidths=0.3, c=[c])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("D (μm²/s)")
    ax.set_ylabel("v (μm/s)")
    ax.set_title("D vs v per track")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "D_vs_v_scatter_by_condition.png"), dpi=600)
    plt.close(fig)

    # --------- (5) ECDF of speeds (Δt=1) per condition ---------
    # Prefer speeds_by_step_long.csv (step column). Fall back to speeds_time_series.csv
    fig, ax = plt.subplots(figsize=(6, 6))
    any_plotted = False
    for c, d in zip(colors, datasets):
        df = d["speeds_by_step"]
        if df is not None and "step" in df.columns:
            x = pd.to_numeric(df.loc[df["step"] == 1, "speed_um_per_s"], errors='coerce').dropna().values
        else:
            ts = d["speeds_ts"]
            x = pd.to_numeric(ts["speed_um_per_s"], errors='coerce').dropna().values if ts is not None else np.array([])
        xs, ys = _ecdf(x)
        if xs.size:
            any_plotted = True
            ax.plot(xs, ys, label=d["label"], alpha=0.9, c=c)
    ax.set_xlabel("Instantaneous speed (μm/s) [Δt=1]")
    ax.set_ylabel("ECDF")
    ax.set_title("Speed distributions (tails expose active runs)")
    ax.grid(True, linestyle='--', alpha=0.5)
    if any_plotted:
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ecdf_speed_step1.png"), dpi=600)
    plt.close(fig)

    # --------- (6) Ensemble MSD overlays (median with IQR) ---------
    msd_sets = []
    tau_mins, tau_maxs = [], []
    for d in datasets:
        msd = d["msd_masked"]
        if msd is None or msd.empty:
            continue
        tau = pd.to_numeric(msd["tau_s"], errors='coerce')
        val = pd.to_numeric(msd["msd_um2"], errors='coerce')
        good = tau.notna() & val.notna() & (tau > 0) & (val > 0)
        if good.any():
            tau_mins.append(tau[good].min())
            tau_maxs.append(tau[good].max())
            msd_sets.append((d["label"], d["plot_folder"], msd.loc[good, ["track_id", "tau_s", "msd_um2"]].copy()))

    if len(msd_sets) >= 1 and tau_mins and tau_maxs:
        tau_lo = max(np.min(tau_mins), 1e-6)
        tau_hi = min(np.max(tau_maxs), tau_lo*1e6)
        if tau_hi > tau_lo:
            tau_grid = np.logspace(np.log10(tau_lo), np.log10(tau_hi), 60)
            fig, ax = plt.subplots(figsize=(6, 6))

            # optional collector to write coverage counts
            coverage_rows = []

            for c, (lab, pf, df_msd) in zip(colors, msd_sets):
                # Interpolate each track onto shared grid (NaN outside support)
                series = []
                for tid, sub in df_msd.groupby("track_id"):
                    t = sub["tau_s"].values
                    m = sub["msd_um2"].values
                    if len(t) >= 3:
                        interp = _interp_loggrid(t, m, tau_grid)  # NaN outside support
                        series.append(interp)

                if not series:
                    continue

                M = np.vstack(series)  # (n_tracks, len_grid)
                n_tracks = M.shape[0]

                # Find earliest τ where ALL tracks have finite data
                valid_counts = np.sum(np.isfinite(M), axis=0)
                if np.any(valid_counts == n_tracks):
                    full_start = int(np.argmax(valid_counts == n_tracks))
                else:
                    # no complete overlap; skip this condition
                    continue

                # From that point onward, keep everything; med/IQR computed from available tracks at each τ
                tau_use = tau_grid[full_start:]
                M_use   = M[:, full_start:]

                # Per-τ stats with varying n (NaNs ignored)
                med = np.nanmedian(M_use, axis=0)
                q25 = np.nanpercentile(M_use, 25, axis=0)
                q75 = np.nanpercentile(M_use, 75, axis=0)
                n_per_tau = np.sum(np.isfinite(M_use), axis=0)  # accurate sample size per τ

                ax.plot(tau_use, med, label=lab, c=c, linewidth=2)
                ax.fill_between(tau_use, q25, q75, color=c, alpha=0.3, linewidth=0)

                # (optional) stash coverage table for this condition
                coverage_rows.append(pd.DataFrame({
                    "condition": lab,
                    "tau_s": tau_use,
                    "n_tracks_contributing": n_per_tau
                }))

            #plot reference lines corresponding to free diffusion and ballistic motion corresponding to slopes 1 and 2
            # Add reference lines
            ref_time = np.logspace(np.log10(tau_lo), np.log10(tau_hi), 100)
            D_eff = 0.01  # Adjust as needed
            msd_passive = 2 * D_eff * ref_time
            plt.loglog(ref_time, msd_passive, 'k--', linewidth=2, label='Passive Diffusion (slope=1)')
            v_eff = 0.1  # Adjust as needed
            msd_active = (v_eff * ref_time)**2
            plt.loglog(ref_time, msd_active, 'k:', linewidth=2, label='Active Transport (slope=2)')

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel("Time lag τ (s)")
            ax.set_ylabel("MSD (μm²)")
            ax.set_title("Ensemble MSD (median ± IQR)")
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "ensemble_MSD_overlay.png"), dpi=600)
            plt.close(fig)

            # (optional) write coverage table
            if coverage_rows:
                pd.concat(coverage_rows, ignore_index=True).to_csv(
                    os.path.join(save_dir, "ensemble_MSD_tau_coverage.csv"), index=False
                )


    # --------- (7) α1 vs α2 scatter + τ_break hist (if two-regime) ---------
    # Scatter
    made_scatter = False
    fig, ax = plt.subplots(figsize=(6, 6))
    for c, d in zip(colors, datasets):
        ta = d["track_analysis"]
        cols = {"alpha_reg1", "alpha_reg2"}
        if cols.issubset(ta.columns):
            a1 = pd.to_numeric(ta["alpha_reg1"], errors='coerce')
            a2 = pd.to_numeric(ta["alpha_reg2"], errors='coerce')
            mask = a1.notna() & a2.notna()
            if mask.any():
                made_scatter = True
                ax.scatter(a1[mask], a2[mask], s=16, alpha=0.9, label=d["label"],
                           edgecolors='black', linewidths=0.3, c=[c])
    if made_scatter:
        # diagonal
        mn, mx = ax.get_xlim(); mn2, mx2 = ax.get_ylim()
        lo, hi = min(mn, mn2), max(mx, mx2)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, xs, linestyle='--', color='gray', linewidth=1)
        ax.set_xlabel("α₁ (shorter lags)")
        ax.set_ylabel("α₂ (longer lags)")
        ax.set_title("Two-regime scaling exponents")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "alpha_reg1_vs_reg2_scatter.png"), dpi=600)
        plt.close(fig)
    else:
        plt.close(fig)

    # τ_break hist (log x)
    made_hist = False
    fig, ax = plt.subplots(figsize=(6, 6))
    for c, d in zip(colors, datasets):
        ta = d["track_analysis"]
        if "break_tau" in ta.columns:
            bt = pd.to_numeric(ta["break_tau"], errors='coerce')
            bt = bt[(bt > 0) & np.isfinite(bt)]
            if bt.size:
                made_hist = True
                ax.hist(bt, bins=np.logspace(np.log10(bt.min()), np.log10(bt.max()), 20),
                        alpha=0.75, color=c, edgecolor='black', label=d["label"])
    if made_hist:
        ax.set_xscale('log')
        ax.set_xlabel("τ_break (s)")
        ax.set_ylabel("Count")
        ax.set_title("Break times between regimes")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "break_tau_hist_log.png"), dpi=600)
        plt.close(fig)
    else:
        plt.close(fig)

    # --------- (8) Summary table across conditions ---------
    def iqr(x): 
        return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

    rows = []
    for lab, d in zip(labels, datasets):
        ta = d["track_analysis"]
        for metric in ["alpha_single", "Kalpha_single", "D", "v", "alpha_reg1", "alpha_reg2", "break_tau"]:
            if metric not in ta.columns: 
                continue
            vals = pd.to_numeric(ta[metric], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size == 0:
                continue
            rows.append({
                "condition": lab,
                "metric": metric,
                "n_tracks": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "iqr": float(iqr(vals))
            })
    if rows:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(os.path.join(save_dir, "summary_metrics_by_condition.csv"), index=False)

    print(f"Saved comparative plots & tables to: {save_dir}")


# --------------- Example usage (edit and run) -----------------
if __name__ == "__main__":
    # Comparison sets. The original file kept the Fig. 5E-G set as a commented-out
    # alternative (with several exploratory per-day sets, removed here); choose one with
    # SETTING. The default reproduces the run that was active in the original file.
    SETTING = "S16A-C_1uM_time_series"
    SETTINGS = {
        # Fig. S16A-C: control, then 15, 25 and 60 min after adding 1 uM nocodazole (6 Sep 2025)
        "S16A-C_1uM_time_series": {
            "folder_paths": [str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 5" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 13_after_15_min" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 13_after_25_min" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 13_after_60_min" / "pooled_AVRG")],
            "labels": ["Ctrl", "15 min", "25 min", "60 min"],
            "save_name": "1um_time_series_2",
        },
        # Fig. 5E-G: three control / nocodazole pairs (1, 5 and 10 uM) and DMSO
        "fig5E-G_mixed_conditions": {
            "folder_paths": [str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 5" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 13_after_60_min" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "7 Sep 2025 membrane test" / "4th animal" / "good" / "Image 20" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "7 Sep 2025 membrane test" / "4th animal" / "good" / "Image 21_after_75min" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good" / "Image 10" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good" / "Image 15" / "pooled_AVRG"),
                             str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 12" / "pooled_AVRG")],
            "labels": ["Ctrl 1", "1 $\mu$M Noc", "Ctrl 2", "5 $\mu$M Noc", "Ctrl 3", "10 $\mu$M Noc", "DMSO"],
            "save_name": "mixed_conditions_AVRG",  # same folder as plot_compare_data_sig.py
        },
    }
    folder_paths = SETTINGS[SETTING]["folder_paths"]
    labels = SETTINGS[SETTING]["labels"]

    plot_path = str(OUT_ROOT / "03_vesicle_transport" / "Comparison plots")
    save_path = os.path.join(plot_path, SETTINGS[SETTING]["save_name"])  # change as needed
    os.makedirs(save_path, exist_ok=True)

    compare_datasets(folder_paths, labels=labels, save_dir=save_path)  # save_dir=None auto-chooses a folder
