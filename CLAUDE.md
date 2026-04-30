# Research Plan: Basin Analysis & Ermakov-Kalitkin Damping for PINNs

## Author Background

**Dzmitry Budzko** — PhD in Mathematics (computational methods, dynamical systems, celestial mechanics), Lead Developer at EPAM Systems (10 years, JS/TS ecosystem). Publications on the restricted 4–5 body problem (equilibria search, stability, resonances, Hamiltonian normalization) and iterative methods with extended convergence domains (jointly with Cordero & Torregrosa, Universitat Politècnica de València).

**Key publication for this project:**
Budzko D.A., Cordero A., Torregrosa J.R. "New Family of Iterative Methods Based on the Ermakov–Kalitkin Scheme for Solving Nonlinear Systems of Equations" // Computational Mathematics and Mathematical Physics, 2015, Vol. 55, No. 12, pp. 1947–1959.

---

## Project Overview

Transfer of tools from the theory of iterative methods and dynamical systems into the domain of Physics-Informed Neural Network (PINN) training. The project is a single unified study with two tightly coupled contributions: a diagnostic framework (basin analysis) and a new training method (Ermakov-Kalitkin damping), where the diagnostic directly validates the method.

---

## Contribution 1 (Diagnostic): Basin of Convergence Analysis for PINN Training

### Idea

PINN training is formulated as an iterative process for solving the nonlinear system ∇L(θ) = 0. Different weight initializations θ₀ lead to different outcomes: correct solution, trivial solution, or divergence. The structure of these basins of convergence has not been studied for PINNs.

### Novelty

1. **New diagnostic tool:** visualization of basin-of-convergence maps for PINN optimizers (Adam, L-BFGS, Adam+L-BFGS, NysNewton-CG, Self-Scaled BFGS). Analogous to phase planes from Budzko et al. 2015 (Figs. 1–8), but for PINN training.

2. **Basin entropy as a quantitative reliability metric for optimizers:** basin entropy (Daza et al. 2016) — a single number characterizing unpredictability: the higher the entropy, the more fractal the basin boundaries, the less predictable the training outcome.

3. **Uncertainty exponent α:** α = 1 for smooth boundaries, α < 1 for fractal ones. We show correlation of α with empirical PINN failure rate.

4. **Complementarity with existing approaches:** Rathore et al. (2024) analyze the loss landscape through the Hessian spectrum (local curvature). We analyze global structure — which initializations lead where. This is an orthogonal diagnostic.

---

## Contribution 2 (Method): Ermakov-Kalitkin Damping for PINN Training

### Idea

Apply the PM scheme from Budzko et al. 2015 to the L-BFGS phase of PINN training. The Ermakov-Kalitkin damping multiplier automatically controls step size, extending the convergence basin. The diagnostic from Contribution 1 is used to demonstrate the results.

### Novelty

1. **New optimizer for PINNs:** PM-damped L-BFGS, based on published third-order convergence theory (Budzko et al. 2015).

2. **Theoretical justification:** unlike heuristic approaches, the PM scheme has a proven convergence order and a formula for the error coefficient depending on parameter α. Small |α| minimizes the coefficient of e_k³.

3. **Basin smoothing:** visual and quantitative demonstration (via basin entropy from Contribution 1) that PM damping transforms fractal basin boundaries into smooth ones — improving training predictability.

4. **Orthogonality:** PM damping combines with other improvements (adaptive weighting, curriculum training, SSBFGS).

### Key Technical Challenge: Jacobian Inversion

In the classical PM scheme, [f'(x_k)]⁻¹f(x_k) is needed. For PINNs, f(x_k) = ∇L(θ_k), f'(x_k) = H(θ_k) — the Hessian of size n×n, n ~ 10⁴–10⁶. Direct inversion is infeasible.

**Solution:** we do not need the full matrix H⁻¹. We only need the vector H⁻¹g, where g = ∇L.

**Ermakov-Kalitkin damping multiplier formula:**

β_k = ||g_k||² / (||g_k||² + ||g_k - d_k||²)

where g_k = ∇L(θ_k) — gradient, d_k = H⁻¹g_k — Newton direction (approximate).

**Options for approximating d_k = H⁻¹g_k:**

| Option | Method | Overhead | Accuracy |
|--------|--------|----------|----------|
| 1 (primary) | L-BFGS direction | Zero — already computed | Rough but sufficient |
| 2 | NysNewton-CG (Rathore 2024) | Hessian-vector products | High |
| 3 | Diagonal Hessian (Hutchinson) | Low | Medium |
| 4 | Subspace projection | Controllable | Controllable |

**Recommendation:** start with Option 1 (L-BFGS direction). The modification is minimal — during the L-BFGS phase of PINN training, replace the fixed step size with β_k. Zero overhead. If results are good — that alone is publishable. Option 2 (NysNewton-CG) — as an ablation study for strengthening.

### PM Scheme Adapted for PINN Training

**Standard Adam+L-BFGS pipeline:**
1. Adam phase: N₁ steps (typically 10,000–50,000)
2. L-BFGS phase: until convergence

**PM-modified pipeline:**
1. Adam phase: N₁ steps (unchanged)
2. PM-L-BFGS phase:
   - At each L-BFGS step k, compute direction d_k (approximation of H⁻¹g_k)
   - Compute β_k = ||g_k||² / (||g_k||² + ||g_k - d_k||²)
   - Update: θ_{k+1} = θ_k - β_k · d_k
   - For α ≠ 0: two-step PM with predictor y_k = θ_k - α·d_k and corrector per formula (5) from the 2015 paper

**One-parameter family:** parameter α controls the balance. For small |α| (α = −0.01, 0.1) — maximum stability (proven in the 2015 paper).

---

## Experimental Plan

### Step 1. Benchmark Problems
- 1D convection equation (β = 30, 50) — canonical failure case from Krishnapriyan et al. 2021
- 1D reaction-diffusion (Allen-Cahn equation) — stiff PDE
- Optional: 2D Navier-Stokes lid-driven cavity — for scalability demonstration

### Step 2. Fixed Architecture
- MLP: 4 hidden layers × 50 neurons, tanh activation
- Fix everything except varied parameters

### Step 3. Building Basin Maps (Diagnostic)
- Select a 2D cross-section of the initialization space (three approaches):
  - (a) random seed + learning rate
  - (b) first two network weights (rest fixed)
  - (c) PCA projection onto 2D subspace in parameter space
- For each grid point (400×400 = 160,000 PINN runs) determine outcome: correct solution / trivial / divergence
- Color by outcome — obtain fractal-like map

### Step 4. Computing Basin Entropy and Uncertainty Exponent
- Following methodology of Daza et al. (2016): partition the map into boxes of size ε, count colors in each box, compute entropy
- Construct log-log dependence S_b(ε) to determine α
- Compare α and S_b across different optimizers

### Step 5. Comparison of Optimizers (Diagnostic Results)
- Adam → basin map + S_b
- L-BFGS → basin map + S_b
- Adam+L-BFGS → basin map + S_b
- NysNewton-CG → basin map + S_b
- SSBFGS → basin map + S_b

### Step 6. Hypothesis Testing (Diagnostic)
- H1: Fractal boundaries (α < 1) correlate with high failure rate
- H2: Second-order methods produce less fractal basins than first-order
- H3: Existing fixes (adaptive weighting, curriculum training) smooth basins, but not completely

### Step 7. PM-L-BFGS Implementation in JAX
- Modify L-BFGS step with β_k damping
- Implement full PM two-step scheme (predictor-corrector)

### Step 8. Comparison with Baselines (Method Validation)
- Adam+L-BFGS (standard)
- Adam+SSBFGS (Urbán et al. 2024/2025)
- Adam+NysNewton-CG (Rathore et al. 2024)
- Adam+PM-L-BFGS (our method)
- Metrics: L2 relative error, iteration count, failure rate, basin entropy

### Step 9. Basin Maps Before and After PM Damping
- Using the diagnostic framework from Contribution 1, show that PM damping smooths fractal boundaries

### Step 10. Ablation Study
- Dependence on α (α = −0.45, −0.01, 0.1, 0.45 — as in the 2015 paper)
- Combination of PM damping with adaptive weighting (Wang et al. 2021)
- Combination of PM damping with curriculum training (Krishnapriyan et al. 2021)

### Step 11. Scalability
- Show that PM-damping overhead is negligible

---

## Confirmed Novelty (as of April 2026)

| Result | Status | Verification |
|--------|--------|--------------|
| Basin of convergence analysis for PINN training | **Novel** | Bosman 2019 — only standard NNs, not PINNs |
| Basin entropy as PINN optimizer reliability metric | **Novel** | Daza 2016 — only dynamical systems |
| Ermakov-Kalitkin damping for neural networks | **Novel** | Not found anywhere |
| Correlation of basin fractality with PINN failure rate | **Novel** | Not investigated |
| Quasi-Newton methods for PINNs in general | Exists | SSBFGS, NysNewton-CG, energy natural gradient |
| Loss landscape analysis for PINNs | Exists | Rathore 2024, Krishnapriyan 2021, FP64 2025 |

---

## Key References

### Foundations — PINN Failure Modes
- Krishnapriyan et al. (2021) "Characterizing possible failure modes in physics-informed neural networks" — NeurIPS
- Rathore et al. (2024) "Challenges in Training PINNs: A Loss Landscape Perspective" — ICML, arXiv:2402.01868
- Wang et al. (2021) "Understanding and Mitigating Gradient Flow Pathologies in PINNs" — gradient pathology

### Foundations — Basin Analysis
- Daza et al. (2016) "Basin entropy: a new tool to analyze uncertainty in dynamical systems" — Scientific Reports
- Daza et al. (2022) "Classifying basins of attraction using the basin entropy" — arXiv:2201.08083
- Bosman (2019) "Visualising basins of attraction for the cross-entropy and the squared error neural network loss functions" — Neurocomputing

### PINN Optimizers (Baselines for Comparison)
- Urbán et al. (2024/2025) "Self-Scaled (Modified) Broyden Method for Solving PINNs" — CMAME
- Müller & Zeinhofer (2023) "Energy natural gradient descent" — Newton direction connection
- Rathore et al. (2024) NysNewton-CG optimizer — ICML

### PINN Initialization Sensitivity
- Ensemble PINNs (2025) "Learning and discovering multiple solutions using PINNs with random initialization and deep ensemble" — arXiv:2503.06320
- Quantifying Training Difficulty (2024) "effective rank and initialization techniques" — arXiv:2410.06308
- FP64 paper (2025) "Rethinking Failure Modes in PINNs" — arXiv:2505.10949

### Preconditioning & Related Approaches
- Preconditioning for PINNs (2024) — ICML, arXiv:2402.00531
- Adaptive Momentum with Cubic Damping (2025) — structural dynamics inspired

### Iterative Methods (Our Foundation)
- Ermakov V.V., Kalitkin N.N. (1981) "The optimal step and regularization for Newton's method" — USSR CMMP
- Budzko, Cordero, Torregrosa (2015) "New Family of Iterative Methods Based on the Ermakov–Kalitkin Scheme" — CMMP
- Budzko, Cordero, Torregrosa (2015) "A new family of iterative methods widening areas of convergence" — Appl. Math. Comput.
- Budzko, Cordero, Torregrosa (2014) "Modifications of Newton's method to extend the convergence domain" — SeMA Journal

---

## Target Venue

- **Primary:** Journal of Computational Physics (JCP) or Computer Methods in Applied Mechanics and Engineering (CMAME)
- **Alternative:** SIAM Journal on Scientific Computing, Neural Networks
- **Parallel:** arXiv preprint for visibility

---

## Tech Stack

- **Language:** Python (JAX preferred — jax.grad for autodiff, functional style close to mathematical notation)
- **Libraries:** JAX, Optax (optimizers), DeepXDE (PINNs, optional), Matplotlib/Plotly (visualization)
- **Compute:** Google Colab Pro ($10/month), GPU not critical for core experiments
- **Tools:** Claude Code for boilerplate and visualization
