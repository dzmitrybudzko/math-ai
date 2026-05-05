# Research Plan: Basin Analysis & PM-Damping for PINNs
# Target: Journal of Computational Physics (JCP)
# Updated: 2026-05-05

## Strategy

One paper with two coupled contributions:
1. **Diagnostic:** Dimension-independent basin analysis for PINN training (basin stability + basin entropy)
2. **Method:** Ermakov-Kalitkin (PM) damped L-BFGS for PINNs

Key insight from May 2025 analysis: 2D basin maps alone are insufficient (reviewers will question representativeness). The paper must lead with dimension-independent statistical metrics, supported by 2D visualizations as illustrations.

## Critical References to Cite/Position Against

| Paper | What they do | Our differentiation |
|-------|-------------|---------------------|
| Ly & Gong (Nature Comms, Apr 2025) | Multifractal loss landscape theory for generic DL | We: computational diagnostic for PINNs specifically |
| Ly & Gong (arXiv:2510.05606, 2025) | Riddled basin geometry, uncertainty exponent → 0 for DL | We: first quantitative basin entropy metric for PINNs; they are theoretical |
| Sohl-Dickstein (arXiv:2402.06184, 2024) | Fractal trainability boundary in hyperparameter space | We: initialization space basins, not hyperparameter space |
| Urbán et al. (JCP, Dec 2024) | SSBFGS: BFGS beats Adam by 6 orders of magnitude | Must compare against — this is current SOTA |
| FP64 paper (NeurIPS 2025) | PINN failures = L-BFGS premature stop in FP32 | Our PM-damping is complementary fix; run all in FP64 |
| Rathore et al. (ICML 2024) | NysNewton-CG + loss landscape via Hessian spectrum | We: global basin structure (complementary to their local curvature) |
| Ensemble PINNs (Proc. Roy. Soc. A, 2025) | 50 random inits discover multiple solutions | Closest to our Monte Carlo approach but no entropy metric |
| Menck et al. (Nature Physics, 2013) | Basin stability is dimension-independent | Theoretical foundation for our approach |
| Daza et al. (Sci. Rep. 2016) | Basin entropy for dynamical systems | We apply their metric to ML for the first time |

---

## Phase 0: Reproduce Paper Results [COMPLETED]

**Status: DONE** (commits 4105b7b, e150807)

- Scalar basins (Figs 1-6): ✓ output/phase0/fig1-6_*.png
- System basins (Figs 7-8): ✓ output/phase0/fig7-8_*.png
- JAX infrastructure validated

---

## Phase 1: Dimension-Independent Basin Analysis (Diagnostic)

**Goal:** Establish basin stability and basin entropy as quantitative PINN reliability metrics that work in full parameter space (not just 2D slices).

### Phase 1A — Monte Carlo Basin Stability [PRIORITY]

**Concept:** Run N random PINN initializations, classify outcomes, compute statistics. Dimension-independent by Menck et al. 2013 — accuracy depends only on N, not on parameter dimension.

**Setup:**
- PDE: 1D convection u_t + β·u_x = 0, β ∈ {1, 5, 10, 30, 50}
- Architecture: MLP 4×50, tanh (same as Krishnapriyan et al. 2021)
- Initialization: Xavier/Glorot normal (standard, not customized)
- Training: Adam 5000 steps, lr=1e-3
- N = 1000 runs per (beta, optimizer) pair
- **Precision: FP64** (per FP64 paper findings)

**Optimizers:**
| Optimizer | Config | Notes |
|-----------|--------|-------|
| Adam | lr=1e-3, 5000 steps | Baseline |
| Adam + L-BFGS | 5000 Adam + L-BFGS to convergence | Standard two-phase |
| Adam + PM-L-BFGS | 5000 Adam + PM-damped L-BFGS | Our method (Phase 2) |
| Adam + SSBFGS | 5000 Adam + Self-Scaled BFGS | Urbán et al. 2024 baseline |

**Outcome classification:**
- CORRECT: L2 relative error < 1%
- TRIVIAL: L2 error ∈ [1%, 100%] — converged to wrong/trivial solution
- DIVERGED: L2 > 100% or NaN or loss increased

**Metrics per (beta, optimizer):**
- Basin stability: P(CORRECT), P(TRIVIAL), P(DIVERGED)
- 95% confidence intervals (binomial)
- Mean/median L2 error
- Failure rate = 1 - P(CORRECT)

**Compute:** 1000 runs × 5 betas × 4 optimizers = 20,000 runs. At ~7s/run on A100: ~39 GPU-hours.

### Phase 1B — k-NN Basin Entropy in Full Parameter Space

**Concept:** For each trained initialization θ₀ᵢ ∈ R^d, look at k nearest neighbors in parameter space and compute local outcome entropy. Average = basin entropy estimate in full dimension.

**Algorithm:**
```
Input: {(θ₀ᵢ, outcomeᵢ)}_{i=1}^N, parameter k
For each i:
    Find k nearest neighbors of θ₀ᵢ (Euclidean norm in R^d)
    Count outcome distribution among k neighbors: n_correct, n_trivial, n_diverged
    H_i = -Σ (n_c/k) · log(n_c/k) for each class c with n_c > 0
S_b = (1/N) · Σ H_i
```

**Parameters:**
- k ∈ {5, 10, 20, 50, 100} — multi-scale analysis
- Same runs as Phase 1A (no extra computation, just post-processing)
- Compare S_b(k) across optimizers and betas

**Scaling analysis:** Plot S_b vs k on log-log scale → estimate uncertainty exponent α.

**Deliverable:** Table of S_b per (optimizer, beta, k). Comparison across optimizers.

### Phase 1C — 2D Basin Maps (Visualization)

**Purpose:** Illustration supporting the dimension-independent metrics. Framing: "analogous to loss landscape visualization (Li et al. 2018 NeurIPS)" — accepted diagnostic, not exhaustive description.

**Three types of 2D visualization:**

1. **Seed × Learning Rate grid** (30×30 or 50×50):
   - X: learning rate, logspace(1e-5, 1e-1, 30)
   - Y: random seed (0–29)
   - Color by outcome
   - Already implemented in convection_pinn.py

2. **PCA projection:**
   - Collect N=1000 initial weight vectors from Phase 1A
   - PCA → top 2 components
   - Scatter plot colored by outcome
   - Shows whether basins have spatial structure in parameter space

3. **Random 2D subspace** (à la Li et al. 2018):
   - Center: mean of all θ₀ vectors
   - Two filter-normalized random directions
   - Evaluate on 30×30 grid in that plane
   - Repeat for 5 random planes → report consistency of S_b

**Cross-section consistency test:**
- Compute S_b for 5-10 independent random 2D subspaces
- Report mean ± std
- If std is small → 2D slices are representative of full-D structure

### Phase 1D — Allen-Cahn Equation (Second PDE)

Same protocol as 1A-1C but for Allen-Cahn: u_t = ε²·u_xx + u - u³, ε=0.01.
Demonstrates generality across PDE types.

### Phase 1E — Hypothesis Testing

**H1:** Basin entropy S_b correlates with failure rate across (optimizer, beta) pairs.
- Spearman correlation of S_b vs failure_rate
- Expected: positive (higher entropy = less predictable = higher failure)

**H2:** Second-order methods produce lower S_b (smoother basins) than first-order.
- Compare: {Adam} vs {L-BFGS, SSBFGS, PM-L-BFGS}
- Mann-Whitney U test

**H3:** PM-damping reduces S_b compared to standard L-BFGS.
- Paired comparison: same runs, same inits, different optimizer
- This is the key validation of Contribution 2

---

## Phase 2: PM-Damped L-BFGS Implementation (Method)

### Phase 2A — Core Implementation

**Variant 1 (zero overhead) — PRIMARY:**
```
d_k = L-BFGS_direction(θ_k)     # already computed
g_k = ∇L(θ_k)                    # already computed
β_k = ||g_k||² / (||g_k||² + ||g_k - d_k||²)
θ_{k+1} = θ_k - β_k · d_k
```

**Variant 2 (full PM two-step):**
```
d_k = L-BFGS_direction(θ_k)
y_k = θ_k - α · d_k                          # predictor
g_y = ∇L(y_k)                                 # extra gradient
d_y = L-BFGS_direction_at(y_k)                # extra L-BFGS step
u_k = ||g_k||² / (b·||g_k||² + c·||g_y||²)   # corrector damping
θ_{k+1} = y_k - u_k · d_y
```

Where b, c are from the PM scheme formula (depend on α).

**Implementation language:** PyTorch (matching existing convection_pinn.py infrastructure).

**Tests:**
- β_k ∈ (0, 1] always
- Convergence on toy quadratic: f(x) = x^T A x, A = diag(1, 100)
- Convergence matches or exceeds standard L-BFGS on simple PINN (β=1)

### Phase 2B — Benchmark Comparison

**Setup:** Same PDEs as Phase 1 (convection β=30,50; Allen-Cahn).

**Pipeline:** Adam 10,000 steps → second-order phase.

**Methods:**
| Method | Description |
|--------|-------------|
| L-BFGS | Standard, strong Wolfe line search |
| PM-L-BFGS (α=-0.01) | Variant 1, small α |
| PM-L-BFGS (α=0.1) | Variant 1, moderate α |
| PM-L-BFGS-2step (α=-0.01) | Variant 2, full predictor-corrector |
| SSBFGS | Urbán et al. 2024 |
| NysNewton-CG | Rathore et al. 2024 (if feasible) |

**Metrics:**
- L2 relative error (final)
- Steps to L2 < 1e-4 (or timeout at 50k total)
- Failure rate over 100 seeds
- Wall-clock time
- Overhead vs standard L-BFGS (should be <5% for Variant 1)

### Phase 2C — Basin Smoothing Demonstration

The KEY figure of the paper: basin maps before and after PM damping.

- Same initializations, same Phase 1C protocol
- Side-by-side: Adam+L-BFGS vs Adam+PM-L-BFGS
- Quantitative: ΔS_b, change in basin stability P(CORRECT)
- Both in 2D maps AND dimension-independent k-NN entropy

### Phase 2D — Ablation

**α sweep:** -0.45, -0.1, -0.01, 0.01, 0.1, 0.45
- Plot: failure rate vs α, S_b vs α, L2 error vs α
- Expected: optimal near α≈0 (per Theorem 1 of 2015 paper)

**Combinations:**
- PM-L-BFGS alone
- PM-L-BFGS + adaptive loss weighting (Wang et al. 2021)
- PM-L-BFGS + curriculum training (Krishnapriyan 2021)
- Shows orthogonality of our method

### Phase 2E — 2D PDE (for JCP)

**Required for JCP submission:** at least one 2D problem.

Options (pick one):
- 2D Helmholtz equation (simpler, well-studied in PINN literature)
- 2D lid-driven cavity (Navier-Stokes, impressive but expensive)
- 2D Allen-Cahn (natural extension of 1D)

**Minimal demonstration:**
- Show PM-L-BFGS works on 2D PDE
- Basin stability comparison (1000 runs): L-BFGS vs PM-L-BFGS
- Prove scalability (overhead still <5%)

---

## Phase 3: Paper Writing & Submission

### Structure (for JCP)

1. Introduction: PINN failure modes → basin analysis as diagnostic → PM-damping as fix
2. Background: Basin stability (Menck 2013), basin entropy (Daza 2016), PM scheme (Budzko 2015)
3. Methodology:
   - 3.1 Monte Carlo basin stability for PINNs
   - 3.2 k-NN basin entropy in parameter space
   - 3.3 PM-damped L-BFGS
4. Numerical Experiments:
   - 4.1 Basin analysis results (convection, Allen-Cahn)
   - 4.2 PM-L-BFGS comparison with baselines
   - 4.3 Basin smoothing demonstration
   - 4.4 2D PDE scalability
   - 4.5 Ablation study
5. Discussion: relationship to Ly & Gong 2025, FP64 findings, limitations
6. Conclusion

### Submission plan
1. Post to arXiv (cs.LG + math.NA + cs.NA) for priority
2. Submit to JCP
3. If rejected → SISC (reframe for "methods and algorithms")
4. If rejected → AMC (emphasize iterative methods theory)

---

## Implementation Priority & Dependencies

```
Phase 0 [DONE] ─────────────────────────────────────────────┐
                                                              │
Phase 1A: Monte Carlo stability ◄── First, dimension-independent results
    │                                 (needs only train function + FP64 fix)
    │
Phase 2A: PM-L-BFGS implementation ◄── Can start in parallel with 1A
    │
Phase 1B: k-NN entropy ◄── Post-processing on 1A results (no extra runs)
    │
Phase 1C: 2D basin maps ◄── Visualization, uses same infra
    │
Phase 2B: Benchmark comparison ◄── Needs 2A implementation
    │
Phase 1D: Allen-Cahn ◄── Second PDE, same protocol
    │
Phase 2C: Basin smoothing demo ◄── KEY FIGURE, needs 2A + 1C
    │
Phase 1E: Hypothesis testing ◄── Statistical analysis on all results
    │
Phase 2D: Ablation ◄── Needs 2B results
    │
Phase 2E: 2D PDE ◄── Last experimental piece
    │
Phase 3: Paper writing ◄── When all experiments done
```

## Compute Budget (Revised)

| Phase | GPU-hours (A100) | Notes |
|-------|-----------------|-------|
| 1A: MC stability (convection, 5 betas, 4 opts) | ~39h | 20k runs × 7s |
| 1A: MC stability (Allen-Cahn) | ~39h | Same scale |
| 1B: k-NN entropy | 0 | Post-processing only |
| 1C: 2D maps (30×30, 3 betas, 4 opts, 3 types) | ~20h | ~10k runs |
| 2B: Benchmarks (100 seeds × 6 methods × 2 PDEs) | ~10h | Longer runs but fewer |
| 2C: Basin smoothing | ~10h | Subset of 1C with PM |
| 2D: Ablation | ~15h | 6 alphas × 100 seeds × 2 PDEs |
| 2E: 2D PDE | ~20h | Larger network, fewer runs |
| **Total** | **~150h** | ~6 days continuous on A100 |

At Colab Pro ($10/month): ~2-3 months with A100 allocation.

## Key Technical Decisions

1. **FP64 everywhere** — per FP64 paper (NeurIPS 2025), many "failure modes" are FP32 artifacts
2. **PyTorch** — matching existing codebase (convection_pinn.py uses PyTorch, not JAX)
3. **Checkpointing per run** — Colab disconnects are inevitable; save each run individually
4. **SSBFGS baseline** — must implement or port from Urbán et al. code (check if public)
5. **Seed × LR grid** — keep as one visualization type, but NOT the primary result

## Success Criteria (for JCP acceptance)

| Criterion | Minimum | Target |
|-----------|---------|--------|
| Basin stability clearly differentiates optimizers | P(CORRECT) differs by >15% | >30% difference |
| S_b correlates with failure rate | Spearman r > 0.4 | r > 0.7 |
| PM-L-BFGS reduces failure rate vs L-BFGS | >10% relative | >30% relative |
| PM-L-BFGS competitive with SSBFGS | Within 2× L2 error | Matches or beats |
| PM overhead | <20% wall-clock | <5% (Variant 1) |
| 2D PDE demonstration works | PM converges | PM beats L-BFGS |
| Cross-section consistency | std(S_b) < 0.2 across planes | std < 0.1 |
