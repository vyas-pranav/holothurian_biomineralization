#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Track-level significance tests of alpha and K_alpha, control vs nocodazole.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. 5F–G (statistics)
What it does  : For the pairs (Ctrl 1, 1 uM Noc), (Ctrl 2, 5 uM Noc) and (Ctrl 3, 10 uM Noc):
                two-sided Mann-Whitney U on alpha and on log10 K_alpha, Hodges-Lehmann shift,
                geometric-mean ratio and Cliff's delta with bootstrap 95% CIs, and
                Benjamini-Hochberg q-values across all tests.
Inputs        : DATA_ROOT/vesicles/confocal_2025/<...>/pooled_AVRG/plots2/track_analysis.xlsx/.csv
Outputs       : OUT_ROOT/03_vesicle_transport/Comparison plots/mixed_conditions_AVRG/
                _significance_alpha_kalpha/alpha_kalpha_significance.csv and .txt
Environment   : environment-analysis.yml (Python 3.7)
Run           : python plot_compare_data_sig.py
"""
import os, warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
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

# --- style (kept minimal; no plots in this script) ---
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

# ------------------------- IO helpers -------------------------

def _resolve_plot_folder(folder_path: str) -> str:
    """
    Accept either a 'pooled_*' dataset folder OR its 'plots2' subfolder.
    Return the folder that actually contains track_analysis.* (usually plots2).
    """
    candidates = [folder_path, os.path.join(folder_path, "plots2")]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.exists(os.path.join(c, "track_analysis.xlsx")) or
            os.path.exists(os.path.join(c, "track_analysis.csv"))
        ):
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

def _ensure_outdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

# ----------------------- Stats helpers ------------------------

def _nanfilter_pos(x: np.ndarray) -> np.ndarray:
    """Keep finite positive values (for Kalpha)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x[x > 0]

def _nanfilter_any(x: np.ndarray) -> np.ndarray:
    """Keep finite values (for alpha)."""
    x = np.asarray(x, float)
    return x[np.isfinite(x)]

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: P(X>Y) - P(X<Y), robust rank-based effect size in [-1,1].
    Efficient computation using ranks.
    """
    x = np.asarray(x); y = np.asarray(y)
    # Use Mann-Whitney U relation: delta = 2*U/(nx*ny) - 1
    U, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    nx, ny = len(x), len(y)
    return (2.0 * U / (nx * ny)) - 1.0

def _bootstrap_ci(func, x, y, B=5000, alpha=0.05, random_state=0):
    """
    Generic bootstrap CI for a statistic based on two samples (resampling within each sample).
    func(x_resamp, y_resamp) -> float
    """
    rng = np.random.default_rng(random_state)
    nx, ny = len(x), len(y)
    boot = []
    for _ in range(B):
        xb = rng.choice(x, size=nx, replace=True)
        yb = rng.choice(y, size=ny, replace=True)
        boot.append(func(xb, yb))
    lo = np.percentile(boot, 100*alpha/2)
    hi = np.percentile(boot, 100*(1-alpha/2))
    return lo, hi

def _hodges_lehmann_diff(x: np.ndarray, y: np.ndarray) -> float:
    """
    HL estimator for median difference: median of all pairwise (y - x).
    (Positive => drug > ctrl; negative => drug < ctrl)
    """
    x = np.asarray(x); y = np.asarray(y)
    diffs = (y.reshape(-1,1) - x.reshape(1,-1)).ravel()
    return np.median(diffs)

def _hl_diff_ci(x, y, B=5000, alpha=0.05, random_state=0):
    return _bootstrap_ci(lambda xb,yb: _hodges_lehmann_diff(xb,yb), x, y, B=B, alpha=alpha, random_state=random_state)

def _gmr(x_pos: np.ndarray, y_pos: np.ndarray) -> float:
    """
    Geometric Mean Ratio (drug / ctrl) using log10: 10^(median(log10(y)) - median(log10(x))).
    Median-of-logs is robust (pairs well with Mann–Whitney).
    """
    lx = np.log10(x_pos); ly = np.log10(y_pos)
    return 10.0 ** (np.median(ly) - np.median(lx))

def _gmr_ci(x_pos, y_pos, B=5000, alpha=0.05, random_state=0):
    def stat(xb, yb):
        return _gmr(xb, yb)
    lo, hi = _bootstrap_ci(stat, x_pos, y_pos, B=B, alpha=alpha, random_state=random_state)
    return lo, hi

# ----------------------- Core comparison ----------------------

def _extract_metric_arrays(df_ctrl: pd.DataFrame, df_drug: pd.DataFrame, metric: str, positive: bool):
    """
    Return two 1D arrays for the given metric from two track_analysis DataFrames.
    """
    x = pd.to_numeric(df_ctrl.get(metric), errors='coerce').to_numpy()
    y = pd.to_numeric(df_drug.get(metric), errors='coerce').to_numpy()
    xf = _nanfilter_pos(x) if positive else _nanfilter_any(x)
    yf = _nanfilter_pos(y) if positive else _nanfilter_any(y)
    return xf, yf

def test_pair_alpha(ctrl_df: pd.DataFrame, drug_df: pd.DataFrame, ctrl_label: str, drug_label: str) -> Dict:
    """
    Track-level test for alpha_single: Mann–Whitney U + HL diff + Cliff's delta (with CIs).
    Negative HL diff = decrease in alpha in drug vs ctrl.
    """
    x, y = _extract_metric_arrays(ctrl_df, drug_df, "alpha_single", positive=False)
    result = {
        "pair": f"{ctrl_label} vs {drug_label}",
        "metric": "alpha_single",
        "n_ctrl_tracks": int(len(x)),
        "n_drug_tracks": int(len(y)),
    }
    if len(x) < 2 or len(y) < 2:
        result.update({"status": "insufficient tracks", "p_raw": np.nan})
        return result

    # Mann–Whitney U (two-sided)
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    # Effect sizes
    hl = _hodges_lehmann_diff(x, y)           # (drug - ctrl)
    hl_lo, hl_hi = _hl_diff_ci(x, y, B=5000)
    cd = _cliffs_delta(x, y)
    cd_lo, cd_hi = _bootstrap_ci(lambda xb,yb: _cliffs_delta(xb,yb), x, y, B=4000)

    result.update({
        "status": "ok",
        "p_raw": float(p),
        "U": float(U),
        "HL_diff_drug_minus_ctrl": float(hl),
        "HL_diff_CI95_lo": float(hl_lo),
        "HL_diff_CI95_hi": float(hl_hi),
        "Cliffs_delta": float(cd),
        "Cliffs_delta_CI95_lo": float(cd_lo),
        "Cliffs_delta_CI95_hi": float(cd_hi),
        # “Decrease?” flag (strictly negative HL & CI not crossing 0)
        "decrease_flag": bool((hl < 0) and (hl_hi < 0))
    })
    return result

def test_pair_kalpha(ctrl_df: pd.DataFrame, drug_df: pd.DataFrame, ctrl_label: str, drug_label: str) -> Dict:
    """
    Track-level test for Kalpha_single (>0): Mann–Whitney on log values, report robust GMR and CI, Cliff's delta.
    GMR < 1 means a decrease in Kalpha in drug vs ctrl.
    """
    x, y = _extract_metric_arrays(ctrl_df, drug_df, "Kalpha_single", positive=True)
    result = {
        "pair": f"{ctrl_label} vs {drug_label}",
        "metric": "Kalpha_single",
        "n_ctrl_tracks": int(len(x)),
        "n_drug_tracks": int(len(y)),
    }
    if len(x) < 2 or len(y) < 2:
        result.update({"status": "insufficient tracks", "p_raw": np.nan})
        return result

    # Mann–Whitney on logs (still two-sided)
    lx, ly = np.log10(x), np.log10(y)
    U, p = stats.mannwhitneyu(lx, ly, alternative="two-sided")

    gmr = _gmr(x, y)
    gmr_lo, gmr_hi = _gmr_ci(x, y, B=5000)
    cd = _cliffs_delta(lx, ly)  # effect on log scale is equivalent ordering

    # CI for Cliff’s delta on log-scale (same order)
    cd_lo, cd_hi = _bootstrap_ci(lambda xb,yb: _cliffs_delta(xb,yb), lx, ly, B=4000)

    result.update({
        "status": "ok",
        "p_raw": float(p),
        "U": float(U),
        "GMR_drug_over_ctrl": float(gmr),
        "GMR_CI95_lo": float(gmr_lo),
        "GMR_CI95_hi": float(gmr_hi),
        "percent_change": float((gmr - 1.0) * 100.0),
        "Cliffs_delta": float(cd),
        "Cliffs_delta_CI95_lo": float(cd_lo),
        "Cliffs_delta_CI95_hi": float(cd_hi),
        # “Decrease?” flag (GMR<1 and CI entirely <1)
        "decrease_flag": bool((gmr < 1.0) and (gmr_hi < 1.0))
    })
    return result

def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """
    Simple BH-FDR adjuster. Returns q-values in the original order.
    """
    p = np.array([np.nan if (v is None or not np.isfinite(v)) else v for v in pvals], float)
    n = np.sum(np.isfinite(p))
    if n == 0:
        return [np.nan]*len(pvals)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan, dtype=float)
    cummin = np.inf
    rank = 0
    for idx in order:
        if not np.isfinite(p[idx]): 
            continue
        rank += 1
        val = p[idx] * n / rank
        if val < cummin:
            cummin = val
        q[idx] = min(cummin, 1.0)
    return q.tolist()

# ------------------- Orchestration function -------------------

def run_alpha_kalpha_tests(
    folder_paths: List[str],
    labels: List[str],
    pairs: Optional[List[Tuple[int,int]]] = None,
    save_dir: Optional[str] = None
) -> str:
    """
    Compare α and Kα decreases for specified control–drug pairs (by index into folder_paths/labels).
    pairs: list of (ctrl_idx, drug_idx). Default: [(0,1),(2,3),(4,5)]
    Returns the path to the results CSV.
    """
    assert len(folder_paths) == len(labels), "folder_paths and labels must align."

    plot_folders = [_resolve_plot_folder(p) for p in folder_paths]
    track_tables = [_read_track_analysis(pf) for pf in plot_folders]

    if pairs is None:
        # default: (1–2), (3–4), (5–6)
        pairs = [(0,1), (2,3), (4,5)]

    if save_dir is None:
        save_dir = os.path.join(os.path.commonpath(plot_folders), "_significance_alpha_kalpha")
    _ensure_outdir(save_dir)

    results = []
    # Run tests per pair for alpha and Kalpha
    for (i,j) in pairs:
        df_ctrl, df_drug = track_tables[i], track_tables[j]
        lab_ctrl, lab_drug = labels[i], labels[j]

        ra = test_pair_alpha(df_ctrl, df_drug, lab_ctrl, lab_drug)
        rk = test_pair_kalpha(df_ctrl, df_drug, lab_ctrl, lab_drug)
        results.extend([ra, rk])

    # Make DataFrame & adjust p-values across all tests
    df = pd.DataFrame(results)
    if "p_raw" in df.columns:
        qvals = benjamini_hochberg(df["p_raw"].tolist())
        df["q_FDR_BH"] = qvals

    # Add helpful interpretive columns
    def _star(p):
        if not np.isfinite(p): return ""
        return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))

    df["signif_raw"] = [ _star(p) for p in df.get("p_raw", np.nan) ]
    df["signif_q"]   = [ _star(q) for q in df.get("q_FDR_BH", np.nan) ]

    out_csv = os.path.join(save_dir, "alpha_kalpha_significance.csv")
    df.to_csv(out_csv, index=False)

    # Also write a concise text report
    lines = ["Alpha & Kalpha significance (track-level) — decreases in Noc vs Control\n"]
    for (i,j) in pairs:
        lab_ctrl, lab_drug = labels[i], labels[j]
        lines.append(f"Pair: {lab_ctrl}  vs  {lab_drug}")
        for met in ["alpha_single", "Kalpha_single"]:
            row = df[(df["pair"] == f"{lab_ctrl} vs {lab_drug}") & (df["metric"] == met)]
            if row.empty:
                lines.append(f"  {met}: no data")
                continue
            r = row.iloc[0].to_dict()
            if met == "alpha_single":
                lines.append(
                    f"  α: HL(drug-ctrl)={r.get('HL_diff_drug_minus_ctrl', np.nan):.3g} "
                    f"[{r.get('HL_diff_CI95_lo', np.nan):.3g}, {r.get('HL_diff_CI95_hi', np.nan):.3g}], "
                    f"U={r.get('U', np.nan):.3g}, p={r.get('p_raw', np.nan):.3g}, q={r.get('q_FDR_BH', np.nan):.3g} "
                    f"({r.get('signif_q','')}); decrease={r.get('decrease_flag', False)}"
                )
            else:
                gmr = r.get('GMR_drug_over_ctrl', np.nan)
                lines.append(
                    f"  Kα: GMR={gmr:.3g} ({(gmr-1)*100:.1f}%), "
                    f"CI95=[{r.get('GMR_CI95_lo', np.nan):.3g}, {r.get('GMR_CI95_hi', np.nan):.3g}], "
                    f"U={r.get('U', np.nan):.3g}, p={r.get('p_raw', np.nan):.3g}, q={r.get('q_FDR_BH', np.nan):.3g} "
                    f"({r.get('signif_q','')}); decrease={r.get('decrease_flag', False)}"
                )
        lines.append("")

    out_txt = os.path.join(save_dir, "alpha_kalpha_significance.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_txt}")
    return out_csv

# ----------------------- Example usage ------------------------

if __name__ == "__main__":
    # <<< EDIT: your folders / labels here >>>
    folder_paths = [
        str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 5" / "pooled_AVRG"),
        str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 13_after_60_min" / "pooled_AVRG"),
        str(DATA_ROOT / "vesicles" / "confocal_2025" / "7 Sep 2025 membrane test" / "4th animal" / "good" / "Image 20" / "pooled_AVRG"),
        str(DATA_ROOT / "vesicles" / "confocal_2025" / "7 Sep 2025 membrane test" / "4th animal" / "good" / "Image 21_after_75min" / "pooled_AVRG"),
        str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good" / "Image 10" / "pooled_AVRG"),
        str(DATA_ROOT / "vesicles" / "confocal_2025" / "5 Sep 2025 membrane test" / "session2" / "good" / "Image 15" / "pooled_AVRG"),
        str(DATA_ROOT / "vesicles" / "confocal_2025" / "6 Sep 2025 membrane test" / "good" / "Image 12" / "pooled_AVRG")
    ]
    labels = ["Ctrl 1", "1 μM Noc", "Ctrl 2", "5 μM Noc", "Ctrl 3", "10 μM Noc", "DMSO"]

    # Default control–drug pairs: (0,1), (2,3), (4,5)
    save_dir = str(OUT_ROOT / "03_vesicle_transport" / "Comparison plots" / "mixed_conditions_AVRG" / "_significance_alpha_kalpha")
    os.makedirs(save_dir, exist_ok=True)

    run_alpha_kalpha_tests(folder_paths, labels, pairs=[(0,1),(2,3),(4,5)], save_dir=save_dir)
