# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2025 Pranav Vyas
"""Terminal branch length of the steady 1.5D model over the (Pe_f^-1, Da_d) plane.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Fig. S18
What it does  : For each (Pe_f, Da_d) pair on a 200 x 200 log grid
                (Pe_f in [1e-3, 1e3], Da_d in [0.1, 10]) solves the transcendental
                condition Da_d Gamma(L) = Omega(L) (SI Eqs. 19-21) for the terminal
                length L. Both sides are multiplied by exp(-r1 L) to avoid
                overflow (the form written out after SI Eq. 19). The grid is
                solved with scipy root_scalar(brentq), refined with fsolve, and
                then re-solved for 50 sweeps with brentq, each point bracketed
                around the median of its already-solved neighbours.
Inputs        : none
Outputs       : OUT_ROOT/05_branch_transport_model_1p5d/terminal_length_phase_space/
                L_*.npy grids and phase_space_iter_*.png maps (colour limit 0-200)
Environment   : environment-models.yml (Python 3.13)
Run           : python phase_plot_final_length_v2.py

In this file `Pe2` is Pe_f (not its inverse) and `Da4` is Da_d; the maps are
plotted against 1/Pe_f.

Symbols (code -> SI): D2 -> D_f, k4 -> k_d, k5 -> k_t, epsilon -> eps,
vm -> v_m, m_dot_0 -> mdot_0, c2/c4/cg -> c_f/c_d/c_g (dimensional classes),
cf_bar/cd_bar/cg_bar -> non-dimensional c_f/c_d/c_g, Pe2_inv/Pe_inv -> Pe_f^-1,
Da4/Da_d -> Da_d, Da5/Da_t -> Da_t, Da_epsilon/Da_eps -> Da_eps (SI Eqs. 3-8).
"""
import numpy as np
from typing import Optional, Tuple
from scipy.optimize import root_scalar
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

# --- USER PATHS -------------------------------------------------------------
# Point OSSICLE_DATA / OSSICLE_OUT at your copies (see data/README.md).
from pathlib import Path
try:
    _HERE = Path(__file__).resolve()
except NameError:  # interactive (e.g. Spyder cell) execution
    _HERE = Path.cwd().resolve() / "_"
REPO_ROOT = next((p for p in _HERE.parents if (p / "CITATION.cff").exists()), _HERE.parent)
DATA_ROOT = Path(os.environ.get("OSSICLE_DATA", REPO_ROOT / "data"))
OUT_ROOT = Path(os.environ.get("OSSICLE_OUT", REPO_ROOT / "outputs"))
# ----------------------------------------------------------------------------

#set fontstyle as helvetica
plt.rcParams['font.family'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 14


# ------------------------------------------------------------------


def terminal_length(Pe2: float,
                    Da4: float,
                    bracket: Optional[Tuple[float, float]] = None
                   ) -> float:
    """
    Solve Da4 * Γ(L) = Ω(L) for L > 0 (SI Eq. 19), with both sides
    multiplied by exp(-r1 L) (Gamma_prime, Omega_prime) to avoid overflow.
    """
    if Pe2 <= 0 or Da4 <= 0:
        raise ValueError("Pe2 and Da4 must be positive.")

    Δ  = np.sqrt(Pe2**2 + 4*Pe2*Da4)
    r1 = 0.5 * (Pe2 + Δ)          # > 0
    r2 = 0.5 * (Pe2 - Δ)          # < 0

    
    def Gamma_prime(L):
        return (1.0 - np.exp(-r1 * L)) / r1 - (1.0 - np.exp(-r2 * L)) / r2

    def Omega_prime(L):
        return (- (r1 * np.exp(-r1 * L) - r2 * np.exp(-r2 * L)) / Pe2
            + np.exp(-r1 * L) - np.exp(-r2 * L))

    def F(L):
        return Da4 * Gamma_prime(L) - Omega_prime(L)



    # ------------- automatic bracketing ---------------------------
    if bracket is None:
        L_lo, L_hi = 1e-12, 1.0
        f_lo, f_hi = F(L_lo), F(L_hi)
        for _ in range(60):          # expand until sign change
            if f_lo * f_hi < 0.0:
                break
            L_hi *= 2.0
            f_hi  = F(L_hi)
        else:
            raise RuntimeError("Could not bracket the root; "
                               "try a custom 'bracket'.")

        bracket = (L_lo, L_hi)

    sol = root_scalar(F, bracket=bracket, method="brentq")
    if not sol.converged:
        raise RuntimeError("Root finding failed.")
    return sol.root

def compute_phase_space(Pe2_range=(0.1, 50), Da4_range=(0.02, 10), 
                        n_Pe=100, n_Da=100):
    """
    Generate a logarithmic phase space of final lengths L* over Pe2 and Da4 grid.

    Parameters
    ----------
    Pe2_range : tuple
        (min_Pe2, max_Pe2), log scale.
    Da4_range : tuple
        (min_Da4, max_Da4), log scale.
    n_Pe : int
        Number of Pe2 grid points.
    n_Da : int
        Number of Da4 grid points.

    Returns
    -------
    Pe2_vals : 1D array
        Pe2 values used.
    Da4_vals : 1D array
        Da4 values used.
    L_grid : 2D array
        Computed final lengths L*, shape (n_Da, n_Pe).
    """
    Pe2_vals = np.logspace(np.log10(Pe2_range[0]), np.log10(Pe2_range[1]), n_Pe)
    Da4_vals = np.logspace(np.log10(Da4_range[0]), np.log10(Da4_range[1]), n_Da)

    L_grid = np.full((n_Da, n_Pe), np.nan)

    for i, Da4 in enumerate(Da4_vals):
        for j, Pe2 in enumerate(Pe2_vals):
            try:
                L_star = terminal_length(Pe2, Da4)
                L_grid[i, j] = L_star
            except Exception as e:
                # On any error (non-convergence, etc), keep NaN
                L_grid[i, j] = np.nan

    return Pe2_vals, Da4_vals, L_grid


#function to fill NaN values in the grid using neighbor medians
def refine_phase_space(Pe2_vals, Da4_vals, L_grid, max_iters=10):
    """
    Iteratively retry solving NaN grid points using neighbor medians as bracket guess.
    """
    n_Da, n_Pe = L_grid.shape
    L_refined = L_grid.copy()

    for iter_num in range(max_iters):
        changed = False

        #print number of NaN values in the grid
        n_nan = np.sum(np.isnan(L_refined))
        print(f"Iteration {iter_num + 1}: {n_nan} NaN values to refine...")

        for i in range(n_Da):
            for j in range(n_Pe):
                if np.isnan(L_refined[i, j]):
                    neighbors = []

                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n_Da and 0 <= nj < n_Pe:
                                val = L_refined[ni, nj]
                                if not np.isnan(val):
                                    neighbors.append(val)

                    if neighbors:
                        # Use median instead of mean
                        guess_val = np.median(neighbors)

                        # Create bracket around guess
                        bracket = (max(guess_val *0.5 , 1e-8), guess_val * 2.0)

                        Pe2 = Pe2_vals[j]
                        Da4 = Da4_vals[i]

                        try:
                            L_star = terminal_length(Pe2, Da4, bracket=bracket)
                            L_refined[i, j] = L_star
                            changed = True
                        except Exception:
                            pass

        if not changed:
            print(f"Converged after {iter_num + 1} refinement iterations.")
            break

    return L_refined


#create a function that replaces NaN values with the mean of the neighbors
def fill_nan_with_neighbors(L_grid):
    """
    Fill NaN values in the grid with the mean of neighboring values.
    """
    n_Da, n_Pe = L_grid.shape
    L_filled = L_grid.copy()

    for i in range(n_Da):
        for j in range(n_Pe):
            if np.isnan(L_filled[i, j]):
                neighbors = []

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n_Da and 0 <= nj < n_Pe:
                            val = L_filled[ni, nj]
                            if not np.isnan(val):
                                neighbors.append(val)

                if neighbors:
                    L_filled[i, j] = np.mean(neighbors)

    return L_filled

# function to refine the phase space using fsolve
# This function will use fsolve to find the terminal length L* for NaN values in
def refine_phase_space_with_fsolve(Pe2_vals, Da4_vals, L_grid, max_iters=5):
    n_Da, n_Pe = L_grid.shape
    L_refined = L_grid.copy()

    for iter_num in range(max_iters):
        changed = False

        for i in range(n_Da):
            for j in range(n_Pe):
                if np.isnan(L_refined[i, j]):
                    neighbors = []

                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n_Da and 0 <= nj < n_Pe:
                                val = L_refined[ni, nj]
                                if not np.isnan(val):
                                    neighbors.append(val)

                    if neighbors:
                        guess_val = np.median(neighbors)

                        Pe2 = Pe2_vals[j]
                        Da4 = Da4_vals[i]

                        Δ  = np.sqrt(Pe2**2 + 4 * Pe2 * Da4)
                        r1 = 0.5 * (Pe2 + Δ)
                        r2 = 0.5 * (Pe2 - Δ)

                        def Gamma_prime(L):
                            return (1.0 - np.exp(-r1 * L)) / r1 - (1.0 - np.exp(-r2 * L)) / r2

                        def Omega_prime(L):
                            return (- (r1 * np.exp(-r1 * L) - r2 * np.exp(-r2 * L)) / Pe2
                                    + np.exp(-r1 * L) - np.exp(-r2 * L))

                        def F(L):
                            return Da4 * Gamma_prime(L) - Omega_prime(L)

                        try:
                            sol = fsolve(F, guess_val, xtol=1e-10)
                            if sol[0] > 0 and np.isfinite(sol[0]):
                                L_refined[i, j] = sol[0]
                                changed = True
                        except Exception:
                            pass

        if not changed:
            print(f"Converged after {iter_num + 1} refinement iterations.")
            break

    return L_refined


#function to re-evaluate the grid using fsolve
def reevaluate_grid_with_fsolve(Pe2_vals, Da4_vals, L_grid, save_path=None,
                                max_iters=10, neighborhood_size=3, 
                                tol=1e-6):
    """
    Re-evaluate all values in L_grid using fsolve,
    using neighborhood median as initial guess.
    Stops when max change is below tol or after max_iters.

    Parameters
    ----------
    Pe2_vals : array
        Array of Pe2 values (1D).
    Da4_vals : array
        Array of Da4 values (1D).
    L_grid : 2D array
        Grid of L values to be re-evaluated (ideally no NaNs).
    max_iters : int
        Maximum number of sweep iterations.
    neighborhood_size : int
        Size of neighborhood window (must be odd).
    tol : float
        Convergence threshold (max change).

    Returns
    -------
    L_refined : 2D array
        Updated grid with refined L values.
    """

    if neighborhood_size % 2 == 0:
        raise ValueError("Neighborhood size must be odd.")

    n_Da, n_Pe = L_grid.shape
    L_refined = L_grid.copy()
    half_size = neighborhood_size // 2

    sigma = 1.0  # Standard deviation for Gaussian weighting
    def weighted_neighbor_guess(i, j):
        guess_vals = []
        weights = []

        for di in range(-half_size, half_size + 1):
            for dj in range(-half_size, half_size + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < n_Da and 0 <= nj < n_Pe:
                    val = L_refined[ni, nj]
                    if not np.isnan(val):
                        d2 = di**2 + dj**2
                        w = np.exp(-d2 / (2 * sigma**2))
                        guess_vals.append(val * w)
                        weights.append(w)

        if weights:
            return np.sum(guess_vals) / np.sum(weights)
        else:
            return np.nan  # fallback if no valid neighbors
        

    for iter_num in range(max_iters):
        print(f"Re-evaluation sweep iteration {iter_num + 1}")
        L_previous = L_refined.copy()

        for i in range(n_Da):
            for j in range(n_Pe):
                neighbors = []

                # nxn neighborhood
                for di in range(-half_size, half_size + 1):
                    for dj in range(-half_size, half_size + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n_Da and 0 <= nj < n_Pe:
                            val = L_refined[ni, nj]
                            if not np.isnan(val) and val > 0:
                                neighbors.append(val)

                if neighbors:
                    guess_val = np.median(neighbors)
                    # guess_val = np.mean(neighbors)
                    # guess_val = weighted_neighbor_guess(i, j)


                    Pe2 = Pe2_vals[j]
                    Da4 = Da4_vals[i]

                    Δ = np.sqrt(Pe2**2 + 4 * Pe2 * Da4)
                    r1 = 0.5 * (Pe2 + Δ)
                    r2 = 0.5 * (Pe2 - Δ)

                    def Gamma_prime(L):
                        return (1.0 - np.exp(-r1 * L)) / r1 - (1.0 - np.exp(-r2 * L)) / r2

                    def Omega_prime(L):
                        return (- (r1 * np.exp(-r1 * L) - r2 * np.exp(-r2 * L)) / Pe2
                                + np.exp(-r1 * L) - np.exp(-r2 * L))

                    def F(L):
                        return Da4 * Gamma_prime(L) - Omega_prime(L)

                    try:
                        sol = fsolve(F, guess_val, xtol=1e-10)
                        # Check if solution is valid
                        # If fsolve returns a valid solution, update the grid
                        if sol[0] > 0 and np.isfinite(sol[0]):
                            L_refined[i, j] = sol[0]

                        

                    except Exception:
                        # Leave previous value if fsolve fails
                        pass

        #save the grid as an array on disk
        filename = f"L_refined_iter_{iter_num + 1}.npy"
        file_path = os.path.join(save_path, filename) if save_path else filename
        np.save(file_path, L_refined)

        #plot the current state of the grid and save the image
        plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_refined, L_clim=(0, 200))
        plot_name = f"phase_space_iter_{iter_num + 1}.png"
        plot_path = os.path.join(save_path, plot_name) if save_path else plot_name
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)

        # Check convergence
        delta = np.abs(np.sum(L_refined) - np.sum(L_previous))
        print(f"  Max change after iteration: {delta:.2e}")
        # if delta < tol:
        #     print(f"Converged after {iter_num + 1} iterations (delta < {tol})")
        #     break


    return L_refined


#write a function that takes in converged refined grid and re-evaluates it with bracket method for all values

def refine_grid_with_brent(Pe2_vals, Da4_vals, L_grid, bracket_factor=0.5, tol=1e-8):
    """
    Re-evaluate all values in L_grid using robust bracket (Brent) method.

    Parameters
    ----------
    Pe2_vals : array
        Array of Pe2 values (1D).
    Da4_vals : array
        Array of Da4 values (1D).
    L_grid : 2D array
        Grid of L values to be re-evaluated.
    bracket_factor : float
        Factor to create bracket around current L (e.g., 0.5 means [0.5*L, 2*L]).
    tol : float
        Tolerance for root finding.

    Returns
    -------
    L_brent : 2D array
        Updated grid with values re-evaluated using Brent.
    """
    n_Da, n_Pe = L_grid.shape
    L_brent = L_grid.copy()

    for i in range(n_Da):
        for j in range(n_Pe):
            L_guess = L_grid[i, j]

            if not np.isfinite(L_guess) or L_guess <= 0:
                continue  # skip invalid cells

            Pe2 = Pe2_vals[j]
            Da4 = Da4_vals[i]

            Δ = np.sqrt(Pe2**2 + 4 * Pe2 * Da4)
            r1 = 0.5 * (Pe2 + Δ)
            r2 = 0.5 * (Pe2 - Δ)

            def Gamma_prime(L):
                return (1.0 - np.exp(-r1 * L)) / r1 - (1.0 - np.exp(-r2 * L)) / r2

            def Omega_prime(L):
                return (- (r1 * np.exp(-r1 * L) - r2 * np.exp(-r2 * L)) / Pe2
                        + np.exp(-r1 * L) - np.exp(-r2 * L))

            def F(L):
                return Da4 * Gamma_prime(L) - Omega_prime(L)

            # Construct bracket around L_guess
            bracket_lo = max(L_guess * bracket_factor, 1e-12)
            bracket_hi = L_guess * (2.0 - bracket_factor)

            try:
                sol = root_scalar(F, bracket=(bracket_lo, bracket_hi), method='brentq', xtol=tol)
                if sol.converged and sol.root > 0:
                    L_brent[i, j] = sol.root
            except Exception:
                # If fails, leave original value
                pass

    return L_brent



def reevaluate_grid_with_brent(Pe2_vals, Da4_vals, L_grid, save_path=None,
                               max_iters=10, neighborhood_size=3, 
                               tol=1e-6, bracket_factor=0.5):
    """
    Re-evaluate all values in L_grid using Brent (bracket) method,
    using neighborhood median as bracket center guess.
    Stops when max change is below tol or after max_iters.

    Parameters
    ----------
    Pe2_vals : array
        Array of Pe2 values (1D).
    Da4_vals : array
        Array of Da4 values (1D).
    L_grid : 2D array
        Grid of L values to be re-evaluated (ideally no NaNs).
    save_path : str or None
        Directory to save intermediate files (optional).
    max_iters : int
        Maximum number of sweep iterations.
    neighborhood_size : int
        Size of neighborhood window (must be odd).
    tol : float
        Convergence threshold (max change).
    bracket_factor : float
        Factor to create bracket range around median guess.

    Returns
    -------
    L_refined : 2D array
        Updated grid with refined L values.
    """

    if neighborhood_size % 2 == 0:
        raise ValueError("Neighborhood size must be odd.")

    n_Da, n_Pe = L_grid.shape
    L_refined = L_grid.copy()
    half_size = neighborhood_size // 2

    for iter_num in range(max_iters):
        print(f"Re-evaluation sweep iteration {iter_num + 1}")
        L_previous = L_refined.copy()

        for i in range(n_Da):
            for j in range(n_Pe):
                neighbors = []

                # nxn neighborhood
                for di in range(-half_size, half_size + 1):
                    for dj in range(-half_size, half_size + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n_Da and 0 <= nj < n_Pe:
                            val = L_refined[ni, nj]
                            if not np.isnan(val) and val > 0:
                                neighbors.append(val)

                if neighbors:
                    guess_val = np.median(neighbors)

                    Pe2 = Pe2_vals[j]
                    Da4 = Da4_vals[i]

                    Δ = np.sqrt(Pe2**2 + 4 * Pe2 * Da4)
                    r1 = 0.5 * (Pe2 + Δ)
                    r2 = 0.5 * (Pe2 - Δ)

                    def Gamma_prime(L):
                        return (1.0 - np.exp(-r1 * L)) / r1 - (1.0 - np.exp(-r2 * L)) / r2

                    def Omega_prime(L):
                        return (- (r1 * np.exp(-r1 * L) - r2 * np.exp(-r2 * L)) / Pe2
                                + np.exp(-r1 * L) - np.exp(-r2 * L))

                    def F(L):
                        return Da4 * Gamma_prime(L) - Omega_prime(L)

                    # Build bracket
                    lo = max(guess_val * bracket_factor, 1e-12)
                    hi = guess_val * (2.0 - bracket_factor)

                    try:
                        sol = root_scalar(F, bracket=(lo, hi), method='brentq', xtol=1e-10)
                        if sol.converged and sol.root > 0:
                            L_refined[i, j] = sol.root
                    except Exception:
                        # Leave previous value if solver fails
                        pass

        # Save grid as array
        filename = f"L_refined_iter_{iter_num + 1}.npy"
        file_path = os.path.join(save_path, filename) if save_path else filename
        np.save(file_path, L_refined)

        # Plot and save
        plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_refined, L_clim=(0, 200))
        plot_name = f"phase_space_iter_{iter_num + 1}.png"
        plot_path = os.path.join(save_path, plot_name) if save_path else plot_name
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Check convergence
        delta = np.abs(np.sum(L_refined) - np.sum(L_previous))
        print(f"  Total change after iteration: {delta:.2e}")
        # If needed, enable convergence stopping
        # if delta < tol:
        #     print(f"Converged after {iter_num + 1} iterations (delta < {tol})")
        #     break

    return L_refined




def plot_phase_space(Pe2_vals, Da4_vals, L_grid):
    """
    Plot phase space as a color mesh.
    """
    plt.figure(figsize=(8, 6))
    Pe2_mesh, Da4_mesh = np.meshgrid(Pe2_vals, Da4_vals)

    #also mask values above a certain threshold
    # Mask values above a threshold for better visualization
    L_grid_plotted = L_grid.copy()



    c = plt.pcolormesh(Pe2_mesh, Da4_mesh, L_grid_plotted, 
                       shading='auto', cmap='plasma')
    #set the color limits to avoid NaN issues
    c.set_clim(0, 100)  # Adjust based on expected L* range
    plt.colorbar(c, label='Terminal Length L*')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pe$_f$')
    plt.ylabel('Da$_d$')
    plt.title('Phase Space of Terminal Length L*')
    plt.tight_layout()
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    log_dy = np.log10(Da4_mesh.max()) - np.log10(Da4_mesh.min())
    log_dx = np.log10(Pe2_mesh.max()) - np.log10(Pe2_mesh.min())
    aspect_ratio = log_dx / log_dy
    plt.gca().set_aspect(aspect_ratio, adjustable='box')

    # plt.show()


def plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_grid, L_clim=(0, 100)):
    """
    Plot phase space as a color mesh with Da4 vs 1/Pe2.
    """
    plt.figure(figsize=(8, 6))

    # Inverse Pe2
    inv_Pe2_vals = 1.0 / Pe2_vals
    # Prevent division by zero (if Pe2_vals includes 0 — but yours doesn't)

    # Build mesh
    inv_Pe2_mesh, Da4_mesh = np.meshgrid(inv_Pe2_vals, Da4_vals)

    L_grid_plotted = L_grid.copy()

    c = plt.pcolormesh(inv_Pe2_mesh, Da4_mesh, L_grid_plotted, 
                       shading='auto', cmap='plasma')

    c.set_clim(*L_clim)
    plt.colorbar(c, label='Terminal Length L')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1 / Pe$_f$')
    plt.ylabel('Da$_d$')
    plt.title('Phase Space: Da$_d$ vs 1 / Pe$_f$')
    plt.tight_layout()

    # Compute aspect ratio for square pixels
    log_dy = np.log10(Da4_mesh.max()) - np.log10(Da4_mesh.min())
    log_dx = np.log10(inv_Pe2_mesh.max()) - np.log10(inv_Pe2_mesh.min())
    aspect_ratio = log_dx / log_dy
    plt.gca().set_aspect(aspect_ratio, adjustable='box')

    # plt.show()



# ------------------------------------------------------------------

# quick smoke-test: comment these lines out in production
if __name__ == "__main__":
    save_path = str(OUT_ROOT / "05_branch_transport_model_1p5d" / "terminal_length_phase_space")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # for Pe2, Da4 in [(5, 0.2)]:
    #     L = terminal_length(Pe2, Da4)
    #     print(f"Pe2={Pe2:6g}  Da4={Da4:6g}  ->  L*={L:.6g}")

    N = 200 # Number of grid points in each dimension
    Pe2_vals, Da4_vals, L_grid = compute_phase_space(
        Pe2_range=(0.001, 1000), Da4_range=(0.1, 10), n_Pe=N, n_Da=N)

    # plot_phase_space(Pe2_vals, Da4_vals, L_grid)
    plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_grid, L_clim=(0, 200))
    #save the initial grid
    filename = "L_initial.npy"
    file_path = os.path.join(save_path, filename) if save_path else filename
    np.save(file_path, L_grid)
    plot_name = "initial_phase_space.png"
    plot_path = os.path.join(save_path, plot_name) if save_path else plot_name
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)

    # # Fill NaN values in the grid
    # L_grid_ref = refine_phase_space(Pe2_vals, Da4_vals, L_grid, max_iters=20)

    # # plot_phase_space(Pe2_vals, Da4_vals, L_grid_ref)
    # plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_grid_ref, L_clim=(0, 200))

    # Refine the phase space using fsolve
    L_grid_ref_fsolve = refine_phase_space_with_fsolve(Pe2_vals, Da4_vals, L_grid, max_iters=5)
    # plot_phase_space(Pe2_vals, Da4_vals, L_grid_ref_fsolve)
    plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_grid_ref_fsolve, L_clim=(0, 200))
    #save the refined grid
    filename = "L_refined_fsolve.npy"
    file_path = os.path.join(save_path, filename) if save_path else filename
    np.save(file_path, L_grid_ref_fsolve)
    plot_name = "refined_phase_space_fsolve.png"
    plot_path = os.path.join(save_path, plot_name) if save_path else plot_name
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)

    # # Optionally, re-evaluate the grid using fsolve
    # L_grid_reeval_fsolve = reevaluate_grid_with_fsolve(Pe2_vals, Da4_vals, L_grid, save_path, max_iters=20, neighborhood_size=3, tol=1e-6)


    # #calculate the final refined grid using Brent's method
    # L_grid_brent = refine_grid_with_brent(Pe2_vals, Da4_vals, L_grid_reeval_fsolve, bracket_factor=0.5, tol=1e-10)
    # plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_grid_brent, L_clim=(0, 200))
    # #save the final refined grid
    # filename = "L_final_brent.npy"
    # file_path = os.path.join(save_path, filename) if save_path else filename
    # np.save(file_path, L_grid_brent)
    # plot_name = "final_phase_space_brent.png"
    # plot_path = os.path.join(save_path, plot_name) if save_path else plot_name
    # plt.savefig(plot_path, bbox_inches='tight', dpi=300)


    #Refine the phase space using Brent's method with neighborhood median
    L_grid_refined_brent = reevaluate_grid_with_brent(Pe2_vals, Da4_vals, L_grid, save_path=save_path, max_iters=50, neighborhood_size=3, tol=1e-6, bracket_factor=0.5)

    
    # # Optionally, fill NaN values with neighbors' mean
    # L_grid_filled = fill_nan_with_neighbors(L_grid_ref)
    # plot_phase_space_inverse_pe(Pe2_vals, Da4_vals, L_grid_filled, L_clim=(0, 200))

    # print(L_grid)
    # print(L_grid_ref)

    # plt.show()
