"""
Phase 1B: k-NN Basin Entropy in full parameter space.

Generalizes Daza et al. 2016 basin entropy from 2D grids to point clouds
in R^d. For each initialization theta_0, find k nearest neighbors in
parameter space and compute local outcome entropy. Average = basin entropy.

This is a POST-PROCESSING script — requires Phase 1A MC results + saved
initial weight vectors.

Key insight: basin entropy measures local mixing of outcomes, which can be
estimated from point clouds without requiring a grid structure.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


def load_mc_results(jsonl_path):
    """Load MC results from JSONL file."""
    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return sorted(results, key=lambda r: r["seed"])


def collect_initial_weights(seeds, beta, method, n_adam, lr, device, fp64, layers=None):
    """
    Re-create initial weight vectors for each seed.
    Since PyTorch seeding is deterministic, we can reconstruct theta_0.
    """
    import torch
    from convection_pinn import PINN, set_precision

    if fp64:
        set_precision(True)

    weight_vectors = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = PINN(layers=layers)
        params = torch.cat([p.data.flatten() for p in model.parameters()])
        weight_vectors.append(params.numpy())

    return np.array(weight_vectors)


def knn_basin_entropy(weight_vectors, outcomes, k=10):
    """
    Compute k-NN basin entropy in full parameter space.

    Parameters
    ----------
    weight_vectors : ndarray, shape (N, d) — initial weight vectors
    outcomes : ndarray, shape (N,) — outcome codes (0, 1, 2)
    k : int — number of nearest neighbors

    Returns
    -------
    S_b : float — mean local entropy
    local_entropies : ndarray, shape (N,) — per-point entropy
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(weight_vectors)
    _, indices = nn.kneighbors(weight_vectors)

    # indices[:, 0] is the point itself — skip it
    neighbor_indices = indices[:, 1:]

    local_entropies = np.zeros(len(outcomes))
    outcome_classes = np.unique(outcomes)

    for i in range(len(outcomes)):
        neighbor_outcomes = outcomes[neighbor_indices[i]]
        counts = np.array([np.sum(neighbor_outcomes == c) for c in outcome_classes])
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        local_entropies[i] = -np.sum(probs * np.log(probs))

    S_b = np.mean(local_entropies)
    return S_b, local_entropies


def multiscale_entropy(weight_vectors, outcomes, k_values=None):
    """
    Compute basin entropy at multiple scales (k values).
    The scaling S_b(k) on log-log scale gives information about
    the uncertainty exponent.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100]

    k_values = [k for k in k_values if k < len(outcomes)]

    results = {}
    for k in k_values:
        S_b, local_H = knn_basin_entropy(weight_vectors, outcomes, k=k)
        results[k] = {
            "S_b": S_b,
            "std": float(np.std(local_H)),
            "max": float(np.max(local_H)),
            "frac_mixed": float(np.mean(local_H > 0)),
        }
        print(f"  k={k:4d}: S_b={S_b:.4f}, std={np.std(local_H):.4f}, "
              f"frac_mixed={np.mean(local_H > 0):.3f}")

    return results


def analyze_mc_results(jsonl_path, beta, method="adam", n_adam=5000, lr=1e-3,
                       fp64=True, device="cpu", k_values=None):
    """
    Full Phase 1B analysis on MC results from Phase 1A.

    Loads results, reconstructs initial weights, computes k-NN entropy.
    """
    print(f"Loading results from {jsonl_path}...")
    results = load_mc_results(jsonl_path)
    seeds = [r["seed"] for r in results]
    outcomes = np.array([r["outcome"] for r in results])

    print(f"Loaded {len(results)} runs. Reconstructing initial weights...")
    weight_vectors = collect_initial_weights(
        seeds, beta, method, n_adam, lr, device, fp64
    )
    print(f"Weight vectors shape: {weight_vectors.shape} "
          f"(N={weight_vectors.shape[0]}, d={weight_vectors.shape[1]})")

    print(f"\nMultiscale k-NN basin entropy:")
    entropy_results = multiscale_entropy(weight_vectors, outcomes, k_values)

    return {
        "weight_vectors": weight_vectors,
        "outcomes": outcomes,
        "entropy_results": entropy_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="k-NN Basin Entropy (Phase 1B)")
    parser.add_argument("--jsonl", type=str, required=True,
                        help="Path to MC results JSONL from Phase 1A")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--method", default="adam")
    parser.add_argument("--n_adam", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fp64", action="store_true", default=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    args = parser.parse_args()

    analyze_mc_results(
        jsonl_path=args.jsonl, beta=args.beta, method=args.method,
        n_adam=args.n_adam, lr=args.lr, fp64=args.fp64,
        device=args.device, k_values=args.k_values
    )
