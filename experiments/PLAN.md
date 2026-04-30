# Numerical Experiments Plan

## Overview

Three phases, ordered by dependency and risk. Each phase produces standalone results; later phases build on earlier infrastructure.

---

## Phase 0: Reproduce Paper Results in JAX (Validation Baseline)

**Goal:** Verify our JAX infrastructure by reproducing Figs. 1–8 and Table 1 from Budzko et al. 2015 before touching PINNs.

### Experiment 0.1 — Scalar Basins of Attraction

Reproduce phase planes for f₁(x) = arctan(x), f₂(x) = arctan(x) − 2x/(1+x²), f₃(x) = (x²−1)/(x²+1) + 1.

**Methods to implement:**
| Method | Formula | Order |
|--------|---------|-------|
| Newton | x_{k+1} = x_k − f(x_k)/f'(x_k) | 2 |
| Ermakov-Kalitkin | Newton with β_k = \|f\|²/(\|f\|² + \|f − f'⁻¹f\|²) | 2 |
| Traub | y = x − f/f'; x_{k+1} = x − (f(x)+f(y))/f'(x) | 3 |
| Jarratt | z = x − (2/3)(f/f'); x_{k+1} = x − (1/2)(3f'(z)+f'(x))⁻¹(3f'(z)−f'(x))f/f' | 4 |
| PM(α) | Two-step predictor-corrector, eq. (4) | 3 |

**Grid:** 400×400 in complex plane, Re(z) ∈ [−6, 6], Im(z) ∈ [−6, 6].
**Coloring:** by root reached (orange/green/blue), brightness by iteration count, black = no convergence in 80 steps.
**α values:** −0.45, −0.01, 0.1, 0.45.

**Deliverable:** 24 phase plane plots matching Figs. 1–8. Visual diff against paper figures.

### Experiment 0.2 — System Basins (Restricted Four-Body Problem)

Reproduce Figs. 7–8: basins for system (18) with μ₁ = 0.25, μ₂ = 0.35 (8 solutions).

**Grid:** 400×400 in (x, y) plane.
**Deliverable:** 6 phase plane plots (Newton, E-K, Traub, Jarratt, PM α=−0.01, PM α=0.1).

### Experiment 0.3 — Numerical Convergence Table

Reproduce Table 1 (scalar) and Table 2 (system): iteration counts, ACOC, residual norms.
**Precision:** Use float64 (not 10000-digit Mathematica arithmetic — sufficient for PINN context).

**Deliverable:** Tables matching paper results within float64 precision.

**Estimated compute:** < 1 hour on CPU (Colab free tier sufficient).

---

## Phase 1: Basin Analysis for PINN Optimizers (Diagnostic)

**Goal:** Build basin-of-convergence maps for PINN training, quantify fractal structure via basin entropy.

### Experiment 1.1 — PINN on 1D Convection Equation (Canonical Failure Case)

**PDE:** uₜ + β·uₓ = 0, x ∈ [0, 2π], t ∈ [0, 1], u(x,0) = sin(x), periodic BC.
**β values:** 30 (moderate failure), 50 (severe failure) — per Krishnapriyan et al. 2021.
**Architecture:** MLP 4×50, tanh, fixed across all runs.

**2D cross-sections for basin maps:**
| Section ID | Axis X | Axis Y | Everything else |
|------------|--------|--------|-----------------|
| S1 | random seed (0–159999) | — | Fixed LR, single row = one run per seed |
| S2 | learning rate (log scale, 1e-5 to 1e-1) | random seed (0–399) | 400×400 grid |
| S3 | weight[0,0] of layer 1 | weight[0,1] of layer 1 | Rest from fixed seed |
| S4 | PC1 | PC2 | PCA of 400 random inits, project onto top-2 components |

**Outcome classification for each run:**
- **Correct:** L2 relative error < 1% against analytical solution
- **Trivial:** loss converges but L2 error > 10% (wrong/trivial solution)
- **Divergent:** loss > initial loss or NaN

**Training budget per run:** 5000 Adam steps (enough to classify basin, not full convergence).

**Optimizers to compare:**
| Optimizer | Config |
|-----------|--------|
| Adam | lr=1e-3, default betas |
| L-BFGS | max_iter=100, strong Wolfe |
| Adam(10k) + L-BFGS | Standard two-phase |
| Adam(10k) + NysNewton-CG | Per Rathore 2024 |
| Adam(10k) + SSBFGS | Per Urbán 2024 |

**Deliverable:** 5 optimizers × 4 cross-sections × 2 β values = 40 basin maps.

### Experiment 1.2 — PINN on Allen-Cahn Equation (Stiff PDE)

**PDE:** uₜ = ε²·uₓₓ + u − u³, x ∈ [−1, 1], t ∈ [0, 1], ε = 0.01.
**Same protocol** as Experiment 1.1 but with this PDE.

**Deliverable:** 40 more basin maps. Cross-PDE comparison of basin structure.

### Experiment 1.3 — Basin Entropy Computation

For every basin map from 1.1 and 1.2:

**Basin entropy S_b (Daza et al. 2016):**
1. Partition the map into boxes of size ε × ε
2. In each box, count distinct colors (outcomes) → local entropy
3. S_b = average over all boxes
4. Repeat for ε = {2, 4, 8, 16, 32, 64} pixels

**Uncertainty exponent α_u:**
1. Count fraction f(ε) of "uncertain" boxes (boxes with >1 color)
2. Plot log(f) vs log(ε)
3. Slope = α_u − 1, so α_u = slope + 1
4. α_u = 1 → smooth boundaries; α_u < 1 → fractal

**Deliverable:** Table of (optimizer, PDE, β, S_b, α_u). Bar charts comparing optimizers.

### Experiment 1.4 — Hypothesis Testing

**H1:** Correlation of α_u with empirical failure rate.
- failure_rate = fraction of grid points classified as "trivial" or "divergent"
- Compute Pearson/Spearman correlation of α_u vs failure_rate across all configs
- Expected: negative correlation (more fractal → higher failure rate)

**H2:** Second-order methods produce less fractal basins.
- Compare α_u: {Adam} vs {L-BFGS, NysNewton-CG, SSBFGS}
- Wilcoxon rank-sum test

**H3:** Existing fixes smooth basins but not completely.
- Add runs with: (a) adaptive loss weighting (Wang et al. 2021), (b) curriculum training (Krishnapriyan 2021)
- Compare α_u with/without fixes
- 2×2 extra sets of basin maps (2 fixes × 2 PDEs)

**Deliverable:** Statistical tests with p-values, correlation plots.

**Estimated compute for Phase 1:**
- 160,000 short PINN runs per basin map × 5000 steps each
- ~80 basin maps total
- Each run ~2-5 seconds on GPU → ~100-400 GPU-hours
- Strategy: 5000-step runs on Colab Pro, parallelized via jax.vmap over batch of initializations
- Optimization: vectorize weight init, run 400 PINNs in parallel via vmap

---

## Phase 2: PM-Damped L-BFGS for PINNs (Method)

**Goal:** Implement Ermakov-Kalitkin damping in PINN training, demonstrate basin smoothing.

### Experiment 2.1 — PM-L-BFGS Implementation

**Variant 1 (zero overhead):**
```
Standard L-BFGS step gives direction d_k (≈ H⁻¹g_k)
g_k = ∇L(θ_k)
β_k = ||g_k||² / (||g_k||² + ||g_k − d_k||²)
θ_{k+1} = θ_k − β_k · d_k
```

**Variant 2 (full PM two-step):**
```
Predictor: y_k = θ_k − α · d_k
Corrector: compute ∇L(y_k), apply formula (5) adapted for optimization
θ_{k+1} = corrector result
```

Note on formula (5) adaptation: in the original paper, the corrector involves divided differences [y,x;f] and the expression bI + cα²(I − [f'(x)]⁻¹[y,x;f])². For PINN training, we approximate:
- f(x) → ∇L(θ)
- [f'(x)]⁻¹f(x) → L-BFGS direction d_k
- [y,x;f] → (∇L(y) − ∇L(x))/(y − x), approximated via finite differences or Hessian-vector product

**Practical simplification for Variant 2:** Use the scalar damping formula from the paper's eq. (4) adapted as:
```
u_k = ||g_k||² / (b·||g_k||² + c·||g(y_k)||²)
θ_{k+1} = y_k − u_k · [f'(x_k)]⁻¹ · g(y_k)
         ≈ y_k − u_k · d_k^corrector
```
where d_k^corrector is L-BFGS direction computed at y_k (extra L-BFGS step).

**Deliverable:** JAX implementation, unit tests verifying β_k ∈ (0, 1] and convergence on toy problem.

### Experiment 2.2 — PM-L-BFGS on Benchmark PDEs

Same PDEs as Phase 1 (convection β=30,50 and Allen-Cahn).

**Pipeline:**
1. Adam phase: 10,000 steps, lr=1e-3
2. Second phase: one of {L-BFGS, PM-L-BFGS(α), SSBFGS, NysNewton-CG}

**Metrics per run:**
| Metric | Description |
|--------|-------------|
| L2 relative error | Against analytical/reference solution |
| Iteration count | Steps to reach L2 < 1e-4 (or timeout) |
| Failure rate | Over 100 random seeds |
| Wall-clock time | End-to-end training |
| Final loss | Converged loss value |

**α sweep:** −0.45, −0.1, −0.01, 0.01, 0.1, 0.45 (6 values, per paper's finding that small |α| is best).

**Deliverable:** Comparison tables, convergence curves (loss vs iteration, L2 error vs iteration).

### Experiment 2.3 — Basin Maps with PM Damping

Build basin maps (same protocol as Phase 1) for:
- Adam + L-BFGS (baseline, already computed in Phase 1)
- Adam + PM-L-BFGS (α = −0.01)
- Adam + PM-L-BFGS (α = 0.1)
- Adam + PM-L-BFGS (α = 0.45)

**Deliverable:** Side-by-side basin maps showing smoothing effect. Quantitative: ΔS_b and Δα_u before/after PM damping.

### Experiment 2.4 — Ablation Study

**A. α sensitivity:**
- Fix PDE (convection β=30), architecture, Adam phase
- Vary α over {−0.45, −0.3, −0.1, −0.01, 0.01, 0.1, 0.3, 0.45}
- Plot: L2 error vs α, failure rate vs α, S_b vs α
- Expected: U-shaped curve, minimum near α ≈ 0 (matching paper's Theorem 1: small |α| minimizes error coefficient)

**B. Combination with existing fixes:**
- PM-L-BFGS alone
- PM-L-BFGS + adaptive loss weighting (Wang et al. 2021)
- PM-L-BFGS + curriculum training (Krishnapriyan et al. 2021)
- PM-L-BFGS + both
- Metrics: L2 error, failure rate over 100 seeds

**C. Approximation quality of d_k:**
- Variant 1: L-BFGS direction (zero cost)
- Variant 2: NysNewton-CG direction (extra Hessian-vector products)
- Variant 3: Diagonal Hessian via Hutchinson estimator
- Compare: β_k distribution, final L2 error, compute overhead

**Deliverable:** Ablation plots and tables.

### Experiment 2.5 — Scalability Test

**Optional (if time permits):** 2D Navier-Stokes lid-driven cavity.
- Larger network (6×100), more parameters (~50k)
- Show: PM-damping overhead is <5% of total training time
- Compare failure rates: L-BFGS vs PM-L-BFGS over 50 random seeds

**Deliverable:** Timing table, failure rate comparison.

---

## Implementation Order & Dependencies

```
Phase 0 ──────────────────────────────────────┐
  0.1 Scalar basins (JAX infra)              │
  0.2 System basins                           │
  0.3 Convergence tables                      │
                                               ▼
Phase 1 ──────────────────────────────────────┐
  1.1 Convection PINN basins  ◄── reuses basin plotting from 0.1
  1.2 Allen-Cahn PINN basins  ◄── same infra
  1.3 Basin entropy            ◄── post-processing on 1.1/1.2 maps
  1.4 Hypothesis testing       ◄── uses 1.3 results
                                               ▼
Phase 2 ──────────────────────────────────────┐
  2.1 PM-L-BFGS implementation
  2.2 Benchmark comparison     ◄── needs 2.1 + Phase 1 baselines
  2.3 Basin maps with PM       ◄── needs 2.1 + Phase 1 infra
  2.4 Ablation study           ◄── needs 2.2 results
  2.5 Scalability (optional)   ◄── needs 2.1
```

## File Structure

```
experiments/
├── PLAN.md                          # This file
├── phase0_reproduce/
│   ├── iterative_methods.py         # Newton, E-K, Traub, Jarratt, PM implementations
│   ├── basin_plotter.py             # Complex-plane basin visualization
│   ├── scalar_basins.py             # Reproduce Figs 1-6
│   ├── system_basins.py             # Reproduce Figs 7-8, four-body problem
│   └── convergence_tables.py        # Reproduce Tables 1-2
├── phase1_basin_analysis/
│   ├── pinn_solver.py               # PINN training loop (JAX)
│   ├── convection_pinn.py           # 1D convection equation PINN
│   ├── allen_cahn_pinn.py           # Allen-Cahn equation PINN
│   ├── basin_mapper.py              # Generate basin maps for PINN training
│   ├── basin_entropy.py             # S_b and α_u computation (Daza et al. 2016)
│   └── hypothesis_tests.py          # Statistical tests for H1-H3
├── phase2_pm_damping/
│   ├── pm_lbfgs.py                  # PM-damped L-BFGS optimizer
│   ├── benchmark_comparison.py      # Compare PM-L-BFGS vs baselines
│   ├── basin_smoothing.py           # Basin maps before/after PM damping
│   ├── ablation_alpha.py            # α sensitivity study
│   ├── ablation_combinations.py     # PM + existing fixes
│   └── ablation_approximation.py    # d_k approximation quality
├── shared/
│   ├── pde_definitions.py           # PDE residuals, exact solutions, BCs
│   ├── network.py                   # MLP architecture (JAX/Flax)
│   ├── metrics.py                   # L2 error, failure classification
│   └── visualization.py            # Plotting utilities
└── notebooks/
    ├── phase0_results.ipynb
    ├── phase1_results.ipynb
    └── phase2_results.ipynb
```

## Compute Budget Estimate

| Phase | GPU-hours | Colab Pro months | Cost |
|-------|-----------|------------------|------|
| Phase 0 | <1 | 0 (free tier) | $0 |
| Phase 1 | 100-400 | 1-2 | $10-20 |
| Phase 2 | 50-200 | 1 | $10 |
| **Total** | **~150-600** | **2-3** | **$20-30** |

Key optimization: jax.vmap to run hundreds of PINNs in parallel on single GPU. Each individual PINN is tiny (4×50 MLP), so GPU memory allows batching 100-400 simultaneously.

## Success Criteria

| Criterion | Minimum | Target |
|-----------|---------|--------|
| Phase 0 reproduces paper | Visual match of basins | + matching ACOC values |
| Basin entropy distinguishes optimizers | S_b differs by >0.1 between Adam and L-BFGS | + α_u correlation with failure rate (r > 0.5) |
| PM-L-BFGS reduces failure rate | >10% relative reduction vs L-BFGS | >30% reduction |
| PM-L-BFGS smooths basins | Visible smoothing in maps | + S_b reduction >0.2, α_u increase >0.1 |
| Overhead of PM damping | <20% wall-clock increase | <5% |
