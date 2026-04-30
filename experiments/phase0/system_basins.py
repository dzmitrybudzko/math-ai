"""
Phase 0, Experiment 0.2: System basins of attraction for the restricted
four-body problem, reproducing Figs. 7-8 and Table 2 from
Budzko, Cordero, Torregrosa (CMMP 2015).

System (18): equilibrium positions in the Newtonian circular restricted
four-body problem based on Lagrange triangular solutions.

Three primaries at equilateral triangle vertices:
  P0 = (0, 0),  P1 = (1, 0),  P2 = (1/2, sqrt(3)/2)

Self-contained. Run locally or in Google Colab / Kaggle (CPU only).
Requirements: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

SQRT3 = np.sqrt(3.0)

# =================================================================
# System (18) and its Jacobian
# =================================================================

def system_F(xy, mu1, mu2):
    """
    F(x,y) = [F1, F2] from equation (18).

    Parameters
    ----------
    xy : ndarray, shape (..., 2)
    mu1, mu2 : float

    Returns
    -------
    F : ndarray, shape (..., 2)
    """
    x = xy[..., 0]
    y = xy[..., 1]

    r0_sq = x**2 + y**2
    r1_sq = (x - 1.0)**2 + y**2
    r2_sq = 1.0 - x + x**2 - SQRT3 * y + y**2  # = (x-1/2)^2 + (y-sqrt3/2)^2

    r0_3 = np.power(r0_sq, 1.5)
    r1_3 = np.power(r1_sq, 1.5)
    r2_3 = np.power(r2_sq, 1.5)

    term_A = SQRT3 * x - y
    term_B = SQRT3 * (x - 1.0) + y

    F1 = term_A * (1.0 - 1.0 / r0_3) + mu1 * term_B * (1.0 - 1.0 / r1_3)
    F2 = 2.0 * y * (1.0 - 1.0 / r0_3) + mu2 * term_B * (1.0 - 1.0 / r2_3)

    return np.stack([F1, F2], axis=-1)


def system_J(xy, mu1, mu2):
    """
    Jacobian J(x,y) = dF/d(x,y), shape (..., 2, 2).

    Computed via analytical partial derivatives.
    """
    x = xy[..., 0]
    y = xy[..., 1]

    r0_sq = x**2 + y**2
    r1_sq = (x - 1.0)**2 + y**2
    r2_sq = 1.0 - x + x**2 - SQRT3 * y + y**2

    r0_3 = np.power(r0_sq, 1.5)
    r1_3 = np.power(r1_sq, 1.5)
    r2_3 = np.power(r2_sq, 1.5)
    r0_5 = np.power(r0_sq, 2.5)
    r1_5 = np.power(r1_sq, 2.5)
    r2_5 = np.power(r2_sq, 2.5)

    term_A = SQRT3 * x - y
    term_B = SQRT3 * (x - 1.0) + y

    # Partial derivatives of 1/r^3 terms
    # d/dx (1/r0^3) = -3x / r0^5
    # d/dy (1/r0^3) = -3y / r0^5
    # d/dx (1/r1^3) = -3(x-1) / r1^5
    # d/dy (1/r1^3) = -3y / r1^5
    # d/dx (1/r2^3) = -3(2x - 1) / (2 * r2^5)  [since dr2_sq/dx = 2x-1]
    # d/dy (1/r2^3) = -3(2y - sqrt3) / (2 * r2^5)

    # F1 = term_A * (1 - 1/r0^3) + mu1 * term_B * (1 - 1/r1^3)
    # dF1/dx = sqrt3 * (1 - 1/r0^3) + term_A * (3x/r0^5)
    #        + mu1 * sqrt3 * (1 - 1/r1^3) + mu1 * term_B * (3(x-1)/r1^5)
    J11 = (SQRT3 * (1.0 - 1.0 / r0_3)
           + term_A * 3.0 * x / r0_5
           + mu1 * SQRT3 * (1.0 - 1.0 / r1_3)
           + mu1 * term_B * 3.0 * (x - 1.0) / r1_5)

    # dF1/dy = -1 * (1 - 1/r0^3) + term_A * (3y/r0^5)
    #        + mu1 * 1 * (1 - 1/r1^3) + mu1 * term_B * (3y/r1^5)
    J12 = (-(1.0 - 1.0 / r0_3)
           + term_A * 3.0 * y / r0_5
           + mu1 * (1.0 - 1.0 / r1_3)
           + mu1 * term_B * 3.0 * y / r1_5)

    # F2 = 2y * (1 - 1/r0^3) + mu2 * term_B * (1 - 1/r2^3)
    # dF2/dx = 2y * (3x/r0^5) + mu2 * sqrt3 * (1 - 1/r2^3)
    #        + mu2 * term_B * 3(2x-1) / (2*r2^5)
    J21 = (2.0 * y * 3.0 * x / r0_5
           + mu2 * SQRT3 * (1.0 - 1.0 / r2_3)
           + mu2 * term_B * 3.0 * (2.0 * x - 1.0) / (2.0 * r2_5))

    # dF2/dy = 2 * (1 - 1/r0^3) + 2y * (3y/r0^5)
    #        + mu2 * 1 * (1 - 1/r2^3)
    #        + mu2 * term_B * 3(2y - sqrt3) / (2*r2^5)
    J22 = (2.0 * (1.0 - 1.0 / r0_3)
           + 2.0 * y * 3.0 * y / r0_5
           + mu2 * (1.0 - 1.0 / r2_3)
           + mu2 * term_B * 3.0 * (2.0 * y - SQRT3) / (2.0 * r2_5))

    J = np.stack([np.stack([J11, J12], axis=-1),
                  np.stack([J21, J22], axis=-1)], axis=-2)
    return J


# =================================================================
# Find all roots numerically (for basin classification)
# =================================================================

def find_roots(mu1, mu2, grid_n=200, tol=1e-12, max_iter=500):
    """Find roots of system (18) by running Newton from a coarse grid."""
    xs = np.linspace(-3, 3, grid_n)
    ys = np.linspace(-3, 3, grid_n)
    X, Y = np.meshgrid(xs, ys)
    xy = np.stack([X.ravel(), Y.ravel()], axis=-1)

    for _ in range(max_iter):
        F = system_F(xy, mu1, mu2)
        J = system_J(xy, mu1, mu2)
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        Jinv_F0 = (J[..., 1, 1] * F[..., 0] - J[..., 0, 1] * F[..., 1]) / det
        Jinv_F1 = (-J[..., 1, 0] * F[..., 0] + J[..., 0, 0] * F[..., 1]) / det
        step = np.stack([Jinv_F0, Jinv_F1], axis=-1)
        step = np.clip(step, -10, 10)
        xy = xy - step

    F_final = system_F(xy, mu1, mu2)
    norms = np.linalg.norm(F_final, axis=-1)
    converged = norms < tol

    if not converged.any():
        print("WARNING: no roots found!")
        return np.array([])

    candidates = xy[converged]
    roots = [candidates[0]]
    for pt in candidates[1:]:
        dists = [np.linalg.norm(pt - r) for r in roots]
        if min(dists) > 0.01:
            roots.append(pt)

    roots = np.array(roots)
    print(f"  Found {len(roots)} roots for mu1={mu1}, mu2={mu2}:")
    for i, r in enumerate(roots):
        res = np.linalg.norm(system_F(r, mu1, mu2))
        print(f"    root {i}: ({r[0]:+.6f}, {r[1]:+.6f})  |F|={res:.2e}")

    return roots


# =================================================================
# Iterative method steps for 2D systems
# =================================================================

def solve_2x2(J, F):
    """Solve J @ d = F for d, vectorized. J: (...,2,2), F: (...,2)."""
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    det = np.where(np.abs(det) < 1e-30, np.sign(det + 1e-30) * 1e-30, det)
    d0 = (J[..., 1, 1] * F[..., 0] - J[..., 0, 1] * F[..., 1]) / det
    d1 = (-J[..., 1, 0] * F[..., 0] + J[..., 0, 0] * F[..., 1]) / det
    return np.stack([d0, d1], axis=-1)


def step_newton_sys(xy, mu1, mu2):
    F = system_F(xy, mu1, mu2)
    J = system_J(xy, mu1, mu2)
    d = solve_2x2(J, F)
    return xy - d


def step_ek_sys(xy, mu1, mu2):
    F = system_F(xy, mu1, mu2)
    J = system_J(xy, mu1, mu2)
    d = solve_2x2(J, F)  # Newton direction = J^{-1} F
    f_norm_sq = np.sum(F**2, axis=-1, keepdims=True)
    diff = F - d
    diff_norm_sq = np.sum(diff**2, axis=-1, keepdims=True)
    beta = f_norm_sq / (f_norm_sq + diff_norm_sq + 1e-300)
    return xy - beta * d


def step_traub_sys(xy, mu1, mu2):
    F = system_F(xy, mu1, mu2)
    J = system_J(xy, mu1, mu2)
    d = solve_2x2(J, F)
    y = xy - d
    Fy = system_F(y, mu1, mu2)
    dy = solve_2x2(J, Fy)
    return xy - d - dy


def step_jarratt_sys(xy, mu1, mu2):
    F = system_F(xy, mu1, mu2)
    J = system_J(xy, mu1, mu2)
    d = solve_2x2(J, F)
    w = xy - (2.0 / 3.0) * d
    Jw = system_J(w, mu1, mu2)

    # Jarratt for systems: x_{k+1} = x_k - (1/2)(3J(w)-J(x))^{-1}(3J(w)+J(x)) J(x)^{-1} F(x)
    # = x_k - (1/2)(3Jw - Jx)^{-1} (3Jw + Jx) d
    A = 3.0 * Jw - J  # (..., 2, 2)
    B_d = 3.0 * np.einsum("...ij,...j->...i", Jw, d) + np.einsum("...ij,...j->...i", J, d)
    # (3Jw+Jx) @ d
    correction = solve_2x2(A, B_d)
    return xy - 0.5 * correction


def step_pm_sys(xy, mu1, mu2, alpha):
    """
    PM method for systems — scalar-norm generalization of formula (4).

    Predictor:  y = x - alpha * J(x)^{-1} F(x)
    Corrector:  x_{k+1} = y - u * J(x)^{-1} F(y)

    where u = ||F(x)||^2 / (b ||F(x)||^2 + c ||F(y)||^2)
    mirrors the scalar formula u = f(x)^2 / (b f(x)^2 + c f(y)^2).

    b = (1+alpha^2)/(2 alpha^2),  c = (1+alpha)/(2 alpha^2 (alpha-1))
    """
    b_coeff = (1.0 + alpha**2) / (2.0 * alpha**2)
    c_coeff = (1.0 + alpha) / (2.0 * alpha**2 * (alpha - 1.0))

    F_x = system_F(xy, mu1, mu2)
    J_x = system_J(xy, mu1, mu2)
    d = solve_2x2(J_x, F_x)

    # Predictor
    y = xy - alpha * d

    F_y = system_F(y, mu1, mu2)

    # Scalar damping factor generalized via norms
    fx_sq = np.sum(F_x**2, axis=-1, keepdims=True)
    fy_sq = np.sum(F_y**2, axis=-1, keepdims=True)
    u = fx_sq / (b_coeff * fx_sq + c_coeff * fy_sq + 1e-300)

    # Corrector
    Jinv_Fy = solve_2x2(J_x, F_y)
    return y - u * Jinv_Fy


# =================================================================
# Basin computation for 2D systems
# =================================================================

def compute_basins_sys(step_fn, grid_xy, roots, max_iter=80, tol=1e-3):
    """
    Compute basins of attraction for a 2D system.

    Parameters
    ----------
    step_fn : callable, xy_array -> xy_array_next
    grid_xy : ndarray, shape (H, W, 2)
    roots : ndarray, shape (n_roots, 2)
    """
    shape = grid_xy.shape[:2]
    xy = grid_xy.reshape(-1, 2).copy()
    n = len(xy)

    root_id = np.full(n, -1, dtype=np.int32)
    iters = np.full(n, max_iter, dtype=np.int32)
    live = np.ones(n, dtype=bool)

    for i in range(max_iter):
        idx = np.where(live)[0]
        if len(idx) == 0:
            break

        with np.errstate(all="ignore"):
            xy_new = step_fn(xy[idx])

        bad = ~np.all(np.isfinite(xy_new), axis=-1)
        xy_new[bad] = 1e20
        xy[idx] = xy_new

        for j, r in enumerate(roots):
            dist = np.linalg.norm(xy[idx] - r, axis=-1)
            hit = dist < tol
            hit_idx = idx[hit]
            root_id[hit_idx] = j
            iters[hit_idx] = i + 1
            live[hit_idx] = False

        div = np.linalg.norm(xy[idx], axis=-1) > 1e6
        live[idx[div]] = False

    return root_id.reshape(shape), iters.reshape(shape)


# =================================================================
# Visualization
# =================================================================

COLORS_8 = np.array([
    [1.0, 0.3, 0.0],    # red-orange
    [0.0, 0.7, 0.0],    # green
    [0.2, 0.4, 1.0],    # blue
    [1.0, 0.85, 0.0],   # yellow
    [0.7, 0.0, 0.7],    # purple
    [0.0, 0.8, 0.8],    # cyan
    [1.0, 0.5, 0.7],    # pink
    [0.6, 0.4, 0.2],    # brown
    [0.5, 1.0, 0.3],    # lime
    [0.9, 0.4, 0.4],    # salmon
])


def basin_image_sys(root_id, iters, max_iter=80):
    h, w = root_id.shape
    img = np.zeros((h, w, 3))

    n_roots = root_id.max() + 1
    for j in range(n_roots):
        mask = root_id == j
        if not mask.any():
            continue
        bright = np.clip(1.0 - 0.7 * iters[mask] / max_iter, 0.15, 1.0)
        color = COLORS_8[j % len(COLORS_8)]
        for c in range(3):
            img[mask, c] = color[c] * bright

    return img


def plot_basins_sys(results, extent, roots, suptitle, filename=None):
    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).ravel()

    for ax, (title, rid, it, mi) in zip(axes, results):
        img = basin_image_sys(rid, it, mi)
        ax.imshow(img, extent=extent, origin="lower", aspect="equal")
        if roots is not None and len(roots) > 0:
            ax.plot(roots[:, 0], roots[:, 1], "wo", ms=4, mew=0.8,
                    markerfacecolor="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
    plt.close(fig)


# =================================================================
# Convergence table (Table 2)
# =================================================================

def run_method_sys(step_fn, x0, mu1, mu2, max_iter=200, tol=1e-12):
    """Run an iterative method from x0 (real 2D), return (iters, converged)."""
    xy = np.array(x0, dtype=np.float64)

    for i in range(1, max_iter + 1):
        try:
            with np.errstate(all="ignore"):
                xy_new = step_fn(xy[None, :])[0]
            if not np.all(np.isfinite(xy_new)) or np.linalg.norm(xy_new) > 1e10:
                return i, False, np.nan
            xy = xy_new
            res = np.linalg.norm(system_F(xy, mu1, mu2))
            if res < tol:
                return i, True, res
        except Exception:
            return i, False, np.nan

    res = np.linalg.norm(system_F(xy, mu1, mu2))
    return max_iter, res < tol, res


def print_convergence_table(mu1, mu2, roots):
    """Reproduce Table 2."""
    print(f"\n{'=' * 70}")
    print(f"TABLE 2: System (18), mu1={mu1}, mu2={mu2}")
    print(f"{'=' * 70}")

    test_points = [
        ("(-0.2, -0.7)", [-0.2, -0.7]),
        ("(3, 0.21)",    [3.0, 0.21]),
        ("(3, -0.01)",   [3.0, -0.01]),
    ]

    alpha_pm = 0.1

    methods = [
        ("Newton",  lambda xy: step_newton_sys(xy, mu1, mu2)),
        ("E-K",     lambda xy: step_ek_sys(xy, mu1, mu2)),
        ("PM(0.1)", lambda xy: step_pm_sys(xy, mu1, mu2, alpha_pm)),
        ("Traub",   lambda xy: step_traub_sys(xy, mu1, mu2)),
        ("Jarratt", lambda xy: step_jarratt_sys(xy, mu1, mu2)),
    ]

    header = f"{'Method':<10}"
    for label, _ in test_points:
        header += f" | {label:>14}"
    print(header)
    print("-" * len(header))

    for method_name, step_fn in methods:
        row = f"{method_name:<10}"
        for _, x0 in test_points:
            it, conv, res = run_method_sys(step_fn, x0, mu1, mu2)
            if conv:
                row += f" | {it:>14}"
            else:
                row += f" | {'--':>14}"
        print(row)
    print()


# =================================================================
# Main
# =================================================================

def make_grid_2d(x_range, y_range, n=400):
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X, Y], axis=-1)


def run_all(grid_size=400, output_dir="output/phase0"):
    mu1, mu2 = 0.25, 0.35
    max_iter = 80

    print("Finding roots of system (18)...")
    roots = find_roots(mu1, mu2)
    if len(roots) == 0:
        print("ERROR: no roots found. Aborting.")
        return

    extent = [-2, 3, -3, 3]
    grid = make_grid_2d((-2, 3), (-3, 3), grid_size)

    # -- Fig 7: classical methods --
    print(f"\n{'=' * 50}")
    print("  Fig 7: Classical methods -- system (18)")
    print(f"{'=' * 50}")

    classical = []
    for label, sfn in [
        ("(a) Newton",          lambda xy: step_newton_sys(xy, mu1, mu2)),
        ("(b) Ermakov-Kalitkin", lambda xy: step_ek_sys(xy, mu1, mu2)),
        ("(c) Traub",           lambda xy: step_traub_sys(xy, mu1, mu2)),
        ("(d) Jarratt",         lambda xy: step_jarratt_sys(xy, mu1, mu2)),
    ]:
        print(f"  {label:.<32}", end=" ", flush=True)
        t0 = time.time()
        rid, it = compute_basins_sys(sfn, grid, roots, max_iter)
        conv = np.mean(rid >= 0) * 100
        dt = time.time() - t0
        print(f"{dt:6.1f}s  converged {conv:5.1f}%")
        classical.append((label, rid, it, max_iter))

    plot_basins_sys(classical, extent, roots,
                    f"Fig. 7: Classical methods -- system (18), mu1={mu1}, mu2={mu2}",
                    f"{output_dir}/fig7_classical_system.png")

    # -- Fig 8: PM family --
    print(f"\n{'=' * 50}")
    print("  Fig 8: PM family -- system (18)")
    print(f"{'=' * 50}")

    pm_results = []
    for alpha in [-0.01, 0.1]:
        label = f"a = {alpha}"
        sfn = (lambda xy, a=alpha: step_pm_sys(xy, mu1, mu2, a))
        print(f"  PM {label:.<28}", end=" ", flush=True)
        t0 = time.time()
        rid, it = compute_basins_sys(sfn, grid, roots, max_iter)
        conv = np.mean(rid >= 0) * 100
        dt = time.time() - t0
        print(f"{dt:6.1f}s  converged {conv:5.1f}%")
        pm_results.append((f"({chr(ord('a') + len(pm_results))}) PM {label}", rid, it, max_iter))

    plot_basins_sys(pm_results, extent, roots,
                    f"Fig. 8: PM family -- system (18), mu1={mu1}, mu2={mu2}",
                    f"{output_dir}/fig8_pm_system.png")

    # -- Table 2 --
    print_convergence_table(mu1, mu2, roots)

    # -- Also test mu1=0.1, mu2=0.2 (second set in Table 2) --
    mu1b, mu2b = 0.1, 0.2
    print(f"\nFinding roots for mu1={mu1b}, mu2={mu2b}...")
    roots_b = find_roots(mu1b, mu2b)

    test_points_b = [
        ("(0.4, 0.8)", [0.4, 0.8]),
        ("(1, 1)",     [1.0, 1.0]),
        ("(0.2, 3)",   [0.2, 3.0]),
    ]

    print(f"\n{'=' * 70}")
    print(f"TABLE 2 (cont): System (18), mu1={mu1b}, mu2={mu2b}")
    print(f"{'=' * 70}")

    alpha_pm = 0.1
    methods_b = [
        ("Newton",  lambda xy: step_newton_sys(xy, mu1b, mu2b)),
        ("E-K",     lambda xy: step_ek_sys(xy, mu1b, mu2b)),
        ("PM(0.1)", lambda xy: step_pm_sys(xy, mu1b, mu2b, alpha_pm)),
        ("Traub",   lambda xy: step_traub_sys(xy, mu1b, mu2b)),
        ("Jarratt", lambda xy: step_jarratt_sys(xy, mu1b, mu2b)),
    ]

    header = f"{'Method':<10}"
    for label, _ in test_points_b:
        header += f" | {label:>14}"
    print(header)
    print("-" * len(header))

    for method_name, step_fn in methods_b:
        row = f"{method_name:<10}"
        for _, x0 in test_points_b:
            it, conv, res = run_method_sys(step_fn, x0, mu1b, mu2b)
            if conv:
                row += f" | {it:>14}"
            else:
                row += f" | {'--':>14}"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    run_all(grid_size=400)
