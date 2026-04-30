"""
Phase 0: Reproduce scalar basin-of-attraction plots and convergence table
from Budzko, Cordero, Torregrosa (CMMP 2015), Figures 1-6 and Table 1.

Self-contained. Run in Google Colab / Kaggle (free tier) or locally.
Requirements: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ═══════════════════════════════════════════════════════════
# Test functions (complex-valued, vectorized over arrays)
# ═══════════════════════════════════════════════════════════

def f1(z):
    return np.arctan(z)

def f1p(z):
    return 1.0 / (1.0 + z**2)

ROOTS_F1 = [0.0 + 0j]


def f2(z):
    return np.arctan(z) - 2.0 * z / (1.0 + z**2)

def f2p(z):
    return 1.0 / (1.0 + z**2) - 2.0 * (1.0 - z**2) / (1.0 + z**2)**2

ROOTS_F2 = [0.0 + 0j, 1.391745200270735 + 0j, -1.391745200270735 + 0j]


def f3(z):
    return 2.0 * z**2 / (z**2 + 1.0)

def f3p(z):
    return 4.0 * z / (z**2 + 1.0)**2

ROOTS_F3 = [0.0 + 0j]

PROBLEMS = [
    ("f1", f1, f1p, ROOTS_F1),
    ("f2", f2, f2p, ROOTS_F2),
    ("f3", f3, f3p, ROOTS_F3),
]

# ═══════════════════════════════════════════════════════════
# Iterative methods — single step, vectorized
# ═══════════════════════════════════════════════════════════

EPS = 1e-300  # guard against 0/0


def step_newton(z, f, fp):
    return z - f(z) / fp(z)


def step_ek(z, f, fp):
    fz = f(z)
    d = fz / fp(z)
    beta = np.abs(fz)**2 / (np.abs(fz)**2 + np.abs(fz - d)**2 + EPS)
    return z - beta * d


def step_traub(z, f, fp):
    fz = f(z)
    fpz = fp(z)
    y = z - fz / fpz
    return z - (fz + f(y)) / fpz


def step_jarratt(z, f, fp):
    fz = f(z)
    fpz = fp(z)
    d = fz / fpz
    w = z - (2.0 / 3.0) * d
    fpw = fp(w)
    return z - 0.5 * (3.0 * fpw + fpz) / (3.0 * fpw - fpz) * d


def step_pm(z, f, fp, alpha):
    b = (1.0 + alpha**2) / (2.0 * alpha**2)
    c = (1.0 + alpha) / (2.0 * alpha**2 * (alpha - 1.0))
    fz = f(z)
    fpz = fp(z)
    y = z - alpha * fz / fpz
    fy = f(y)
    u = fz**2 / (b * fz**2 + c * fy**2 + EPS)
    return y - u * fy / fpz


# ═══════════════════════════════════════════════════════════
# Basin computation engine
# ═══════════════════════════════════════════════════════════

def compute_basins(step_fn, grid, roots, max_iter=80, tol=1e-6):
    """
    For each grid point, iterate step_fn and classify convergence.

    Returns
    -------
    root_id : ndarray, shape=grid.shape, int
        Index of root reached (0..n-1), or -1 if not converged.
    iters : ndarray, shape=grid.shape, int
        Iteration count at convergence (max_iter if not converged).
    """
    shape = grid.shape
    z = grid.ravel().astype(np.complex128).copy()
    n = len(z)

    root_id = np.full(n, -1, dtype=np.int32)
    iters = np.full(n, max_iter, dtype=np.int32)
    live = np.ones(n, dtype=bool)

    roots_arr = np.array(roots, dtype=np.complex128)

    for i in range(max_iter):
        idx = np.where(live)[0]
        if len(idx) == 0:
            break

        with np.errstate(all="ignore"):
            z_new = step_fn(z[idx])

        bad = ~np.isfinite(z_new)
        z_new[bad] = np.inf
        z[idx] = z_new

        for j, r in enumerate(roots_arr):
            hit = np.abs(z[idx] - r) < tol
            hit_idx = idx[hit]
            root_id[hit_idx] = j
            iters[hit_idx] = i + 1
            live[hit_idx] = False

        div = np.abs(z[idx]) > 1e10
        live[idx[div]] = False

    return root_id.reshape(shape), iters.reshape(shape)


# ═══════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════

BASE_COLORS = np.array([
    [1.0, 0.55, 0.0],   # orange
    [0.0, 0.75, 0.0],   # green
    [0.25, 0.40, 1.0],  # blue
])


def basin_image(root_id, iters, max_iter=80):
    h, w = root_id.shape
    img = np.zeros((h, w, 3))

    n_roots = root_id.max() + 1
    for j in range(max(n_roots, 0)):
        mask = root_id == j
        if not mask.any():
            continue
        bright = np.clip(1.0 - 0.8 * iters[mask] / max_iter, 0.12, 1.0)
        color = BASE_COLORS[j % len(BASE_COLORS)]
        for c in range(3):
            img[mask, c] = color[c] * bright

    return img


def plot_row(results, extent, suptitle, filename=None):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (title, rid, it, mi) in zip(axes, results):
        ax.imshow(basin_image(rid, it, mi),
                  extent=extent, origin="lower", aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Re(z)", fontsize=8)
        ax.set_ylabel("Im(z)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Convergence table (Table 1 — real-valued, high precision)
# ═══════════════════════════════════════════════════════════

def run_method_real(step_fn, x0, f, max_iter=5000, tol=1e-14):
    """Run a method from real x0, return (iters, acoc, residual, converged)."""
    x = float(x0)
    history = [x]

    for i in range(1, max_iter + 1):
        try:
            with np.errstate(all="ignore"):
                x_new = float(np.real(step_fn(np.complex128(x))))
            if not np.isfinite(x_new) or abs(x_new) > 1e15:
                return i, np.nan, np.nan, False
            x = x_new
            history.append(x)
            if abs(f(x)) < tol:
                break
        except (ZeroDivisionError, OverflowError):
            return i, np.nan, np.nan, False

    converged = abs(f(x)) < tol
    iters_used = len(history) - 1

    acoc = np.nan
    if len(history) >= 4:
        e = np.array(history)
        diffs = np.abs(np.diff(e))
        diffs = diffs[diffs > 1e-300]
        if len(diffs) >= 3:
            d1, d2, d3 = diffs[-3], diffs[-2], diffs[-1]
            if d2 > 1e-300 and d1 > 1e-300:
                num = np.log(d3 / d2 + 1e-300)
                den = np.log(d2 / d1 + 1e-300)
                if abs(den) > 1e-15:
                    acoc = num / den

    residual = abs(f(x))
    return iters_used, acoc, residual, converged


def print_convergence_table():
    """Reproduce Table 1 from the paper (float64 precision)."""

    table_config = [
        ("f1", f1, f1p, ROOTS_F1, [1.1, 3.2, 7.2]),
        ("f2", f2, f2p, ROOTS_F2, [2.8, 5.8, 24.0]),
        ("f3", f3, f3p, ROOTS_F3, [0.3, 1.6, 4.8]),
    ]

    alpha_pm = 0.1

    methods = [
        ("Newton", lambda f, fp: (lambda z: step_newton(z, f, fp))),
        ("E-K",    lambda f, fp: (lambda z: step_ek(z, f, fp))),
        ("PM",     lambda f, fp: (lambda z: step_pm(z, f, fp, alpha_pm))),
        ("Traub",  lambda f, fp: (lambda z: step_traub(z, f, fp))),
        ("Jarratt",lambda f, fp: (lambda z: step_jarratt(z, f, fp))),
    ]

    print("\n" + "=" * 78)
    print("TABLE 1: Convergence results (float64, tol=1e-14)")
    print("Compare with Table 1 in Budzko, Cordero, Torregrosa (CMMP 2015)")
    print("=" * 78)

    for func_name, f, fp, roots, x0_list in table_config:
        print(f"\n{'-' * 78}")
        print(f"  {func_name}(x)")
        print(f"{'-' * 78}")

        header = f"{'Method':<10}"
        for x0 in x0_list:
            header += f" | x0={x0:<6}  ACOC"
        print(header)
        print("-" * len(header))

        for method_name, make_step in methods:
            step_fn = make_step(f, fp)
            row = f"{method_name:<10}"
            for x0 in x0_list:
                it, acoc, res, conv = run_method_real(step_fn, x0, f)
                if conv:
                    acoc_str = f"{acoc:.1f}" if np.isfinite(acoc) else "--"
                    row += f" |  {it:>5}   {acoc_str:>4}"
                else:
                    row += f" |    --     --"
            print(row)

    print()


# ═══════════════════════════════════════════════════════════
# Main — generate all figures and table
# ═══════════════════════════════════════════════════════════

def make_grid(re_lim, im_lim, n=400):
    re = np.linspace(re_lim[0], re_lim[1], n)
    im = np.linspace(im_lim[0], im_lim[1], n)
    Re, Im = np.meshgrid(re, im)
    return Re + 1j * Im


def run_all(grid_size=400, output_dir="output/phase0"):
    extent = [-6, 6, -6, 6]
    grid = make_grid((-6, 6), (-6, 6), grid_size)
    max_iter = 80
    tol = 1e-6
    alpha_values = [-0.45, -0.01, 0.1, 0.45]

    # Paper figure mapping:
    # Figs 1,3,5 = classical methods on f1,f2,f3
    # Figs 2,4,6 = PM family on f1,f2,f3
    fig_map_classical = {"f1": "fig1", "f2": "fig3", "f3": "fig5"}
    fig_map_pm = {"f1": "fig2", "f2": "fig4", "f3": "fig6"}

    t_total = time.time()

    for func_name, f, fp, roots in PROBLEMS:
        print(f"\n{'=' * 50}")
        print(f"  {func_name}(z)  --  roots: {len(roots)}")
        print(f"{'=' * 50}")

        # ── Classical methods ──
        classical_results = []
        for label, make_step in [
            ("(a) Newton",         lambda z: step_newton(z, f, fp)),
            ("(b) Ermakov-Kalitkin", lambda z: step_ek(z, f, fp)),
            ("(c) Traub",          lambda z: step_traub(z, f, fp)),
            ("(d) Jarratt",        lambda z: step_jarratt(z, f, fp)),
        ]:
            print(f"  {label:.<30}", end=" ", flush=True)
            t0 = time.time()
            rid, it = compute_basins(make_step, grid, roots, max_iter, tol)
            conv = np.mean(rid >= 0) * 100
            dt = time.time() - t0
            print(f"{dt:5.1f}s  converged {conv:5.1f}%")
            classical_results.append((label, rid, it, max_iter))

        fname = fig_map_classical[func_name]
        plot_row(classical_results, extent,
                 f"Fig. {fname[3]}: Classical methods — {func_name}(z)",
                 f"{output_dir}/{fname}_classical_{func_name}.png")

        # ── PM family ──
        pm_results = []
        for alpha in alpha_values:
            label = f"a = {alpha}"
            sfn = (lambda z, a=alpha: step_pm(z, f, fp, a))
            print(f"  PM {label:.<26}", end=" ", flush=True)
            t0 = time.time()
            rid, it = compute_basins(sfn, grid, roots, max_iter, tol)
            conv = np.mean(rid >= 0) * 100
            dt = time.time() - t0
            print(f"{dt:5.1f}s  converged {conv:5.1f}%")
            letter = chr(ord("a") + alpha_values.index(alpha))
            pm_results.append((f"({letter}) {label}", rid, it, max_iter))

        fname = fig_map_pm[func_name]
        plot_row(pm_results, extent,
                 f"Fig. {fname[3]}: PM family — {func_name}(z)",
                 f"{output_dir}/{fname}_pm_{func_name}.png")

    print(f"\nBasin plots done in {time.time() - t_total:.1f}s total.")

    # ── Convergence table ──
    print_convergence_table()


if __name__ == "__main__":
    run_all(grid_size=400)
