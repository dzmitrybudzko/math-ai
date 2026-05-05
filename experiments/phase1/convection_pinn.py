"""
Phase 1, Experiment 1.1: PINN for 1D convection equation.

PDE:  u_t + beta * u_x = 0,  x in [0, 2pi], t in [0, 1]
IC:   u(x, 0) = sin(x)
BC:   periodic
Exact: u(x, t) = sin(x - beta * t)

Canonical failure case from Krishnapriyan et al. (2021).
Beta=30 is moderate failure, beta=50 is severe.

This module provides:
  - PINN model (MLP 4x50 tanh)
  - Training loop (Adam, L-BFGS, Adam+L-BFGS)
  - L2 relative error against exact solution
  - Single-run and batch-run interfaces for basin mapping
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path


# =================================================================
# PINN Model
# =================================================================

def set_precision(use_fp64=True):
    """Set default dtype. FP64 recommended per FP64 paper (NeurIPS 2025)."""
    dtype = torch.float64 if use_fp64 else torch.float32
    torch.set_default_dtype(dtype)
    return dtype


class PINN(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = [2, 50, 50, 50, 50, 1]
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


# =================================================================
# PDE residual and losses
# =================================================================

def pde_residual(model, x, t, beta):
    """Compute u_t + beta * u_x (should be 0)."""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    grads = torch.autograd.grad(u, [t, x], grad_outputs=torch.ones_like(u),
                                create_graph=True)
    return grads[0] + beta * grads[1]


def compute_loss(model, x_pde, t_pde, x_ic, u_ic, beta,
                 w_pde=1.0, w_ic=1.0):
    """Total PINN loss = w_pde * L_pde + w_ic * L_ic."""
    r = pde_residual(model, x_pde, t_pde, beta)
    loss_pde = torch.mean(r ** 2)

    u_pred_ic = model(x_ic, torch.zeros_like(x_ic))
    loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)

    return w_pde * loss_pde + w_ic * loss_ic


# =================================================================
# Training data
# =================================================================

def generate_training_data(n_pde=300, n_ic=50, device="cpu"):
    """Generate collocation points for PDE and IC."""
    x_pde = torch.rand(n_pde, 1, device=device) * 2 * np.pi
    t_pde = torch.rand(n_pde, 1, device=device)

    x_ic = torch.linspace(0, 2 * np.pi, n_ic, device=device).unsqueeze(1)
    u_ic = torch.sin(x_ic)

    return x_pde, t_pde, x_ic, u_ic


def generate_test_grid(n_x=100, n_t=100, device="cpu"):
    """Generate uniform test grid for error evaluation."""
    x = torch.linspace(0, 2 * np.pi, n_x, device=device)
    t = torch.linspace(0, 1, n_t, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    return X.reshape(-1, 1), T.reshape(-1, 1)


# =================================================================
# Error metrics
# =================================================================

def l2_relative_error(model, beta, n_x=100, n_t=100, device="cpu"):
    """L2 relative error against exact solution."""
    x, t = generate_test_grid(n_x, n_t, device)
    with torch.no_grad():
        u_pred = model(x, t)
    u_exact = torch.sin(x - beta * t)
    err = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    return err.item()


# =================================================================
# Training routines
# =================================================================

def train_adam(model, x_pde, t_pde, x_ic, u_ic, beta,
              n_steps=5000, lr=1e-3):
    """Adam training phase."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = compute_loss(model, x_pde, t_pde, x_ic, u_ic, beta)
        loss.backward()
        optimizer.step()
    return loss.item()


def train_lbfgs(model, x_pde, t_pde, x_ic, u_ic, beta,
                n_steps=100, lr=1.0):
    """L-BFGS training phase."""
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,
                                  max_iter=20, history_size=50,
                                  line_search_fn="strong_wolfe")
    final_loss = None
    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            loss = compute_loss(model, x_pde, t_pde, x_ic, u_ic, beta)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        final_loss = loss.item()
        if not np.isfinite(final_loss):
            break
    return final_loss


# =================================================================
# Single PINN run with outcome classification
# =================================================================

# Outcome codes
CORRECT = 0
TRIVIAL = 1
DIVERGED = 2


def train_single(seed, beta, method="adam", n_adam=5000, n_lbfgs=100,
                 lr_adam=1e-3, device="cpu", layers=None):
    """
    Train a single PINN from a given seed.

    Returns
    -------
    outcome : int (CORRECT=0, TRIVIAL=1, DIVERGED=2)
    l2_err : float
    final_loss : float
    """
    torch.manual_seed(seed)
    model = PINN(layers=layers).to(device)

    x_pde, t_pde, x_ic, u_ic = generate_training_data(device=device)

    try:
        if method == "adam":
            final_loss = train_adam(model, x_pde, t_pde, x_ic, u_ic, beta,
                                   n_steps=n_adam, lr=lr_adam)
        elif method == "lbfgs":
            final_loss = train_lbfgs(model, x_pde, t_pde, x_ic, u_ic, beta,
                                     n_steps=n_lbfgs)
        elif method == "adam+lbfgs":
            train_adam(model, x_pde, t_pde, x_ic, u_ic, beta,
                       n_steps=n_adam, lr=lr_adam)
            final_loss = train_lbfgs(model, x_pde, t_pde, x_ic, u_ic, beta,
                                     n_steps=n_lbfgs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if not np.isfinite(final_loss):
            return DIVERGED, np.nan, final_loss

        l2_err = l2_relative_error(model, beta, device=device)

    except RuntimeError:
        return DIVERGED, np.nan, np.nan

    if l2_err < 0.01:
        return CORRECT, l2_err, final_loss
    elif l2_err < 10.0:
        return TRIVIAL, l2_err, final_loss
    else:
        return DIVERGED, l2_err, final_loss


# =================================================================
# Basin map: sweep over seeds
# =================================================================

def compute_basin_map_seeds(beta, method="adam", n_seeds=400,
                            n_adam=2000, n_lbfgs=50, device="cpu"):
    """
    Run n_seeds PINN trainings, return outcome array.

    This is a 1D basin map (seed axis only).
    For 2D maps (seed x learning_rate), use compute_basin_map_2d.
    """
    outcomes = np.full(n_seeds, -1, dtype=int)
    l2_errors = np.full(n_seeds, np.nan)

    for i in range(n_seeds):
        t0 = time.time()
        outcome, l2_err, loss = train_single(
            seed=i, beta=beta, method=method,
            n_adam=n_adam, n_lbfgs=n_lbfgs, device=device
        )
        dt = time.time() - t0
        outcomes[i] = outcome
        l2_errors[i] = l2_err if np.isfinite(l2_err) else 1.0

        if (i + 1) % 20 == 0 or i == 0:
            n_ok = np.sum(outcomes[:i+1] == CORRECT)
            print(f"  seed {i+1:4d}/{n_seeds}  "
                  f"{dt:.1f}s  "
                  f"correct={n_ok}/{i+1}  "
                  f"L2={l2_err:.4f}  "
                  f"loss={loss:.2e}",
                  flush=True)

    return outcomes, l2_errors


def compute_basin_map_2d(beta, method="adam",
                         n_seeds=50, lr_values=None,
                         n_adam=2000, device="cpu"):
    """
    2D basin map: seed (y-axis) x learning_rate (x-axis).

    Returns
    -------
    outcomes : ndarray, shape (n_seeds, n_lr)
    l2_errors : ndarray, shape (n_seeds, n_lr)
    lr_values : ndarray
    """
    if lr_values is None:
        lr_values = np.logspace(-5, -1, 50)

    n_lr = len(lr_values)
    outcomes = np.full((n_seeds, n_lr), -1, dtype=int)
    l2_errors = np.full((n_seeds, n_lr), np.nan)

    total = n_seeds * n_lr
    count = 0

    for j, lr in enumerate(lr_values):
        for i in range(n_seeds):
            outcome, l2_err, loss = train_single(
                seed=i, beta=beta, method=method,
                n_adam=n_adam, lr_adam=lr, device=device
            )
            outcomes[i, j] = outcome
            l2_errors[i, j] = l2_err if np.isfinite(l2_err) else 1.0
            count += 1

            if count % 100 == 0:
                done_mask = outcomes >= 0
                n_ok = np.sum(outcomes[done_mask] == CORRECT)
                print(f"  {count:5d}/{total}  "
                      f"correct={n_ok}/{count}  "
                      f"lr={lr:.1e}  seed={i}",
                      flush=True)

    return outcomes, l2_errors, lr_values


# =================================================================
# Visualization
# =================================================================

OUTCOME_COLORS = {
    CORRECT:  [0.0, 0.8, 0.0],
    TRIVIAL:  [1.0, 0.6, 0.0],
    DIVERGED: [0.2, 0.2, 0.2],
    -1:       [0.9, 0.9, 0.9],
}

OUTCOME_LABELS = {
    CORRECT:  "Correct (<1% L2)",
    TRIVIAL:  "Trivial (>1% L2)",
    DIVERGED: "Diverged",
    -1:       "Not computed",
}


def plot_basin_1d(outcomes, title="", filename=None):
    """Plot 1D basin map (seed vs outcome)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 1.5))
    img = np.array([OUTCOME_COLORS[o] for o in outcomes])
    ax.imshow(img[None, :, :], aspect="auto", extent=[0, len(outcomes), 0, 1])
    ax.set_xlabel("Seed")
    ax.set_yticks([])
    ax.set_title(title)

    present = sorted(set(outcomes))
    legend = [Patch(facecolor=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o])
              for o in present]
    ax.legend(handles=legend, loc="upper right", fontsize=7)

    fig.tight_layout()
    if filename:
        from pathlib import Path
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
    plt.close(fig)


def plot_basin_2d(outcomes, lr_values, title="", filename=None):
    """Plot 2D basin map (seed x learning_rate)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    h, w = outcomes.shape
    img = np.zeros((h, w, 3))
    for code, color in OUTCOME_COLORS.items():
        mask = outcomes == code
        img[mask] = color

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img, aspect="auto", origin="lower",
              extent=[np.log10(lr_values[0]), np.log10(lr_values[-1]),
                      0, h])
    ax.set_xlabel("log10(learning rate)")
    ax.set_ylabel("Seed")
    ax.set_title(title)

    present = sorted(set(outcomes.ravel()))
    legend = [Patch(facecolor=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o])
              for o in present]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    fig.tight_layout()
    if filename:
        from pathlib import Path
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
    plt.close(fig)


# =================================================================
# Basin entropy (Daza et al. 2016)
# =================================================================

def basin_entropy(outcomes, box_size=5):
    """
    Compute basin entropy S_b for a 2D outcome grid.
    Only counts boxes where all cells have been computed (>= 0).
    """
    h, w = outcomes.shape
    entropies = []
    for i in range(0, h - box_size + 1):
        for j in range(0, w - box_size + 1):
            box = outcomes[i:i+box_size, j:j+box_size].ravel()
            if np.any(box < 0):
                continue
            unique, counts = np.unique(box, return_counts=True)
            probs = counts / counts.sum()
            entropies.append(-np.sum(probs * np.log(probs)))
    if not entropies:
        return np.nan
    return np.mean(entropies)


# =================================================================
# Save / load results
# =================================================================

def save_results(filename, outcomes, l2_errors, lr_values, beta,
                 method, n_adam, S_b=None, extra=None):
    """Save basin map results to .npz file."""
    from pathlib import Path
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    data = dict(
        outcomes=outcomes, l2_errors=l2_errors, lr_values=lr_values,
        beta=beta, method=method, n_adam=n_adam,
    )
    if S_b is not None:
        data["basin_entropy"] = S_b
    if extra:
        data.update(extra)
    np.savez(filename, **data)
    print(f"  Saved: {filename}")


def load_results(filename):
    """Load basin map results from .npz file."""
    return dict(np.load(filename, allow_pickle=True))


def summarize_outcomes(outcomes):
    """Print outcome counts for a completed basin map."""
    done = outcomes[outcomes >= 0]
    total = len(done)
    n_c = np.sum(done == CORRECT)
    n_t = np.sum(done == TRIVIAL)
    n_d = np.sum(done == DIVERGED)
    not_done = np.sum(outcomes < 0)
    print(f"  Correct: {n_c}/{total} ({100*n_c/total:.1f}%)")
    print(f"  Trivial: {n_t}/{total} ({100*n_t/total:.1f}%)")
    print(f"  Diverged: {n_d}/{total} ({100*n_d/total:.1f}%)")
    if not_done > 0:
        print(f"  Not computed: {not_done}")
    return n_c, n_t, n_d


# =================================================================
# Main — quick sanity check
# =================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: PINN for 1D convection equation")
    print("=" * 60)

    beta = 30
    print(f"\nbeta = {beta}")

    # --- Single run sanity check ---
    print("\n--- Single run (seed=0, Adam 3000 steps) ---")
    t0 = time.time()
    outcome, l2_err, loss = train_single(
        seed=0, beta=beta, method="adam", n_adam=3000
    )
    dt = time.time() - t0
    labels = {CORRECT: "CORRECT", TRIVIAL: "TRIVIAL", DIVERGED: "DIVERGED"}
    print(f"  Outcome: {labels[outcome]}")
    print(f"  L2 relative error: {l2_err:.6f}")
    print(f"  Final loss: {loss:.4e}")
    print(f"  Time: {dt:.1f}s")

    # --- 1D basin map (100 seeds, Adam) ---
    print(f"\n--- 1D basin map: 100 seeds, Adam 2000 steps, beta={beta} ---")
    t0 = time.time()
    outcomes, l2_errors = compute_basin_map_seeds(
        beta=beta, method="adam", n_seeds=100, n_adam=2000
    )
    dt = time.time() - t0

    n_correct = np.sum(outcomes == CORRECT)
    n_trivial = np.sum(outcomes == TRIVIAL)
    n_diverged = np.sum(outcomes == DIVERGED)
    print(f"\n  Results: correct={n_correct}, trivial={n_trivial}, "
          f"diverged={n_diverged}")
    print(f"  Failure rate: {100 - n_correct}%")
    print(f"  Total time: {dt:.1f}s")

    plot_basin_1d(outcomes,
                  f"1D Basin Map: Adam, beta={beta}, 2000 steps",
                  f"output/phase1/basin_1d_adam_beta{beta}.png")

    print("\nDone.")
