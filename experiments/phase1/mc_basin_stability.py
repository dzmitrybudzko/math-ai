"""
Phase 1A: Monte Carlo Basin Stability for PINN training.

Dimension-independent metric (Menck et al., Nature Physics 2013):
run N random initializations, classify outcomes, compute statistics.
Accuracy depends only on N, not on parameter dimension.

Designed for Colab resilience:
- Saves each run individually to a JSONL file
- On restart, skips already-completed runs
- Summary computed from saved results at any time

Usage (local quick test):
    python mc_basin_stability.py --beta 1 --n_runs 50 --device cpu

Usage (Colab production):
    python mc_basin_stability.py --beta 30 --n_runs 1000 --device cuda --fp64
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from convection_pinn import (
    set_precision, train_single, CORRECT, TRIVIAL, DIVERGED
)


def get_results_path(out_dir, beta, method, n_adam, lr, fp64):
    """Canonical path for results JSONL file."""
    prec = "fp64" if fp64 else "fp32"
    return Path(out_dir) / f"mc_{method}_beta{int(beta)}_adam{n_adam}_lr{lr:.0e}_{prec}.jsonl"


def load_completed(path):
    """Load completed run indices from JSONL file."""
    completed = {}
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                record = json.loads(line)
                completed[record["seed"]] = record
    return completed


def append_result(path, record):
    """Append a single result to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def compute_basin_stability(results):
    """
    Compute basin stability metrics from results dict.

    Returns dict with:
    - n_total, n_correct, n_trivial, n_diverged
    - p_correct, p_trivial, p_diverged (fractions)
    - ci_correct_low, ci_correct_high (95% Wilson interval)
    - mean_l2, median_l2
    - failure_rate = 1 - p_correct
    """
    outcomes = [r["outcome"] for r in results.values()]
    l2_errors = [r["l2_err"] for r in results.values() if np.isfinite(r["l2_err"])]

    n = len(outcomes)
    n_c = sum(1 for o in outcomes if o == CORRECT)
    n_t = sum(1 for o in outcomes if o == TRIVIAL)
    n_d = sum(1 for o in outcomes if o == DIVERGED)

    p_c = n_c / n if n > 0 else 0.0

    # Wilson score interval (95%)
    z = 1.96
    if n > 0:
        denom = 1 + z**2 / n
        center = (p_c + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p_c * (1 - p_c) / n + z**2 / (4 * n**2)) / denom
        ci_low = max(0, center - margin)
        ci_high = min(1, center + margin)
    else:
        ci_low = ci_high = 0.0

    return {
        "n_total": n,
        "n_correct": n_c,
        "n_trivial": n_t,
        "n_diverged": n_d,
        "p_correct": p_c,
        "p_trivial": n_t / n if n > 0 else 0.0,
        "p_diverged": n_d / n if n > 0 else 0.0,
        "ci_correct_95": (ci_low, ci_high),
        "failure_rate": 1.0 - p_c,
        "mean_l2": float(np.mean(l2_errors)) if l2_errors else np.nan,
        "median_l2": float(np.median(l2_errors)) if l2_errors else np.nan,
    }


def run_mc_stability(beta, method="adam", n_runs=1000, n_adam=5000,
                     lr=1e-3, device="cpu", fp64=True, out_dir="results_phase1a"):
    """
    Run Monte Carlo basin stability experiment.

    Saves each run to JSONL; skips already-completed seeds on restart.
    """
    if fp64:
        set_precision(True)

    results_path = get_results_path(out_dir, beta, method, n_adam, lr, fp64)
    completed = load_completed(results_path)

    print(f"{'='*60}")
    print(f"MC Basin Stability: beta={beta}, method={method}")
    print(f"N={n_runs}, n_adam={n_adam}, lr={lr:.0e}, {'FP64' if fp64 else 'FP32'}")
    print(f"Device: {device}")
    print(f"Results: {results_path}")
    print(f"Already completed: {len(completed)}/{n_runs}")
    print(f"{'='*60}")

    seeds = list(range(n_runs))
    remaining = [s for s in seeds if s not in completed]

    if not remaining:
        print("All runs already completed!")
    else:
        t_start = time.time()
        for idx, seed in enumerate(remaining):
            t0 = time.time()
            outcome, l2_err, final_loss = train_single(
                seed=seed, beta=beta, method=method,
                n_adam=n_adam, lr_adam=lr, device=device
            )
            dt = time.time() - t0

            record = {
                "seed": seed,
                "outcome": int(outcome),
                "l2_err": float(l2_err) if np.isfinite(l2_err) else None,
                "final_loss": float(final_loss) if np.isfinite(final_loss) else None,
                "time_s": round(dt, 2),
            }
            append_result(results_path, record)
            completed[seed] = record

            if (idx + 1) % 50 == 0 or idx == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(remaining) - idx - 1) / rate if rate > 0 else 0
                stats = compute_basin_stability(completed)
                print(f"  [{idx+1}/{len(remaining)}] "
                      f"seed={seed} -> {['CORRECT','TRIVIAL','DIVERGED'][outcome]} "
                      f"({dt:.1f}s) | "
                      f"P(correct)={stats['p_correct']:.3f} "
                      f"[{stats['ci_correct_95'][0]:.3f}, {stats['ci_correct_95'][1]:.3f}] | "
                      f"ETA: {eta/60:.0f}min")

    # Final summary
    stats = compute_basin_stability(completed)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: beta={beta}, method={method}")
    print(f"{'='*60}")
    print(f"  Total runs:     {stats['n_total']}")
    print(f"  Correct:        {stats['n_correct']} ({100*stats['p_correct']:.1f}%)")
    print(f"  Trivial:        {stats['n_trivial']} ({100*stats['p_trivial']:.1f}%)")
    print(f"  Diverged:       {stats['n_diverged']} ({100*stats['p_diverged']:.1f}%)")
    print(f"  P(correct):     {stats['p_correct']:.4f}  95% CI: [{stats['ci_correct_95'][0]:.4f}, {stats['ci_correct_95'][1]:.4f}]")
    print(f"  Failure rate:   {stats['failure_rate']:.4f}")
    print(f"  Mean L2 error:  {stats['mean_l2']:.4f}")
    print(f"  Median L2 err:  {stats['median_l2']:.4f}")

    return stats


def run_multi_beta(betas, **kwargs):
    """Run MC stability for multiple beta values. Print comparison table."""
    all_stats = {}
    for beta in betas:
        stats = run_mc_stability(beta=beta, **kwargs)
        all_stats[beta] = stats
        print()

    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'beta':>6} | {'P(correct)':>10} | {'95% CI':>16} | {'failure%':>8} | {'med L2':>8}")
    print("-" * 60)
    for beta in betas:
        s = all_stats[beta]
        ci = s["ci_correct_95"]
        print(f"{beta:6.0f} | {s['p_correct']:10.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {100*s['failure_rate']:7.1f}% | {s['median_l2']:.4f}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC Basin Stability for PINN")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--betas", type=float, nargs="+", default=None,
                        help="Run multiple betas (overrides --beta)")
    parser.add_argument("--method", default="adam", choices=["adam", "lbfgs", "adam+lbfgs"])
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--n_adam", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fp64", action="store_true", default=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--out_dir", default="results_phase1a")
    args = parser.parse_args()

    fp64 = not args.fp32

    if args.betas:
        run_multi_beta(
            betas=args.betas, method=args.method, n_runs=args.n_runs,
            n_adam=args.n_adam, lr=args.lr, device=args.device,
            fp64=fp64, out_dir=args.out_dir
        )
    else:
        run_mc_stability(
            beta=args.beta, method=args.method, n_runs=args.n_runs,
            n_adam=args.n_adam, lr=args.lr, device=args.device,
            fp64=fp64, out_dir=args.out_dir
        )
