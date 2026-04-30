"""
Quick beta sweep: find the "sweet spot" where PINN has partial convergence.
10 seeds per beta value, 2000 Adam steps. ~3 min per beta on CPU.
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
from convection_pinn import train_single, CORRECT, TRIVIAL, DIVERGED

N_SEEDS = 10
N_ADAM = 2000
BETAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

print(f"Beta sweep: {len(BETAS)} values x {N_SEEDS} seeds x {N_ADAM} Adam steps")
print(f"{'beta':>6} | {'correct':>7} | {'trivial':>7} | {'diverged':>8} | {'avg L2':>8} | {'time':>6}")
print("-" * 60)

for beta in BETAS:
    t0 = time.time()
    outcomes = []
    l2s = []
    for s in range(N_SEEDS):
        o, l2, _ = train_single(seed=s, beta=beta, method="adam", n_adam=N_ADAM)
        outcomes.append(o)
        l2s.append(l2 if np.isfinite(l2) else 1.0)
    dt = time.time() - t0

    n_c = sum(1 for o in outcomes if o == CORRECT)
    n_t = sum(1 for o in outcomes if o == TRIVIAL)
    n_d = sum(1 for o in outcomes if o == DIVERGED)
    avg_l2 = np.mean(l2s)

    print(f"{beta:6.1f} | {n_c:>7} | {n_t:>7} | {n_d:>8} | {avg_l2:8.4f} | {dt:5.0f}s",
          flush=True)

print("\nDone.")
