# Leaderboard: dMRI Microstructure Inference

All results from the development of dmipy-jax and dmijl.
Methods labeled honestly by what they actually are.

---

## 1. Normalizing Flow NPE (Python/JAX) — Ball+2Stick

**Method:** Conditional neural spline flow (FlowJAX) trained via maximum likelihood
on simulated (θ, signal) pairs. Posterior samples via flow inversion.

Target: <3° median Fiber 1 orientation error (Manzano-Patron et al.)

| Date | Config | Fiber 1 | d_stick r | f1 r | Steps | Notes |
|------|--------|---------|-----------|------|-------|-------|
| Mar 19 | Baseline flow (affine, depth=6) | 10.7° | 0.95 | 0.85 | 30k | Starting point |
| Mar 19 | Spline flow + noise fix + label-switching | 6.4° | 0.978 | 0.902 | 30k | All fixes combined |
| Mar 19 | + lr=1e-4, 50k steps | 7.4° | 0.972 | 0.896 | 50k | lr too slow |
| Mar 20 | + lr=3e-4, 100k steps | **4.4°** | 0.983 | 0.924 | 100k | Big improvement |
| Mar 20 | + 200k steps | **3.2°** | 0.986 | 0.935 | 200k | Near target |

### Key findings:
- Training budget dominates: 30k→200k = 6.4°→3.2°
- Wide snr_range=(10,50) helps (regularization effect)
- Spline transformer > affine for multi-modal posteriors
- Label-switching fix in prior essential for multi-fiber models

---

## 2. Denoising Score Matching + DDPM Sampling (Python/JAX) — Ball+2Stick

**Method:** MLP trained to predict noise (ε-prediction) or velocity (v-prediction)
added to parameters at random diffusion times. Posterior samples via DDPM
discrete reverse process. NOT a score-based SDE solver — uses fixed-step
discrete denoising, not continuous-time reverse SDE.

| Date | Config | Fiber 1 | d_stick r | Notes |
|------|--------|---------|-----------|-------|
| Mar 20 | e3nn equivariant network | 60° | 0.0 | Failed — e3nn FunctionalLinear too restrictive |
| Mar 20 | MLP + Euler-Maruyama SDE sampler | 29° | 0.0 | MLP learns but SDE sampler diverges |
| Mar 21 | MLP + DDPM discrete sampler | **15.5°** | 0.977 | DDPM >> SDE >> ODE |
| Mar 21 | + gentle schedule (β_max=5) | 17.6° | 0.961 | Less noise destruction helps |
| Mar 21 | + v-prediction, 1024-wide, 100k | **14.9°** | 0.983 | Scale up |
| Mar 23 | + spherical coords (θ,φ), 30k | **12.8°** | 0.925 | Eliminates sphere constraint |
| Mar 23 | Equivariant output head, 30k | 16.1° | 0.978 | Plain MLP still better |

### Key findings:
- e3nn FunctionalLinear layers can't learn: signal conditioning can't reach 1o (vector) irreps
- DDPM discrete sampler critical — continuous SDE (Euler-Maruyama) diverges
- Spherical parameterization: 15.5°→12.8° by eliminating unit-sphere constraint
- v-prediction marginally better than ε-prediction
- Equivariant output head doesn't help (bottleneck is sampler quality, not parameterization)
- Gap vs flow (12.8° vs 3.2°) likely due to DDPM discretization vs exact flow inversion

---

## 3. Neural Forward Model Surrogate (Julia/Lux.jl)

**Method:** Plain MLP (Lux.jl) trained via supervised regression to approximate
the Ball+2Stick analytical forward model. NOT a PINN — no physics loss,
just MSE/relative-MSE on (params, signal) pairs.

Target: <1% median relative error on held-out signals

| Date | Config | Med. Error | <1% | <5% | Time | Status |
|------|--------|-----------|-----|-----|------|--------|
| Mar 23 | h256_d6_mse, 10k steps | 2.49% | 11% | 90% | 262s | — |
| Mar 23 | h256_d6_relative_mse, 10k | 2.32% | 2% | 95% | 527s | — |
| Mar 23 | h256_d6_log_cosh, 10k | 2.67% | 8% | 86% | 521s | — |
| Mar 23 | h512 configs | 93.8% | 0% | 0% | — | Diverged (relative MSE unstable) |
| Mar 23 | **h256_d6_relative_mse, 50k** | **0.96%** | **53%** | **99%** | 1088s | **SPEC PASSED** |

### Key findings:
- Input normalization to [0,1] critical (21.7%→2.3%)
- h256 stable, h512 diverges with relative MSE loss
- 50k steps crosses <1% threshold
- Surrogate is SLOWER than analytical Ball+Stick (8ms vs 2ms per batch)
- Surrogate value is for replacing expensive MC/FEM simulations, not analytical formulas
- For Ball+Stick specifically, just use the formula

---

## 4. Neural Diffusion Tensor Field Fitting (Julia/Lux.jl) — Real WAND Connectom Data

**Method:** MLP maps spatial position x → D(x) (diagonal diffusion tensor).
Signal prediction via Stejskal-Tanner equation: S(b,g) = ∫ exp(-b·gᵀD(x)g) dx
with Monte Carlo integration over voxel. Trained by minimizing signal
prediction error. **NOT a PINN** — no Bloch-Torrey PDE residual enforced.
This is neural field fitting with the Stejskal-Tanner physics baked into the
signal prediction, not learned from a PDE constraint.

Data: WAND sub-00395, Connectom CHARMED, 7 shells (b=0-6000 s/mm²), 266 volumes, 2mm.
Preprocessed with FSL topup + CUDA eddy on DGX Spark GB10.

| Date | Version | MD (m²/s) | FA | Loss | Time | Notes |
|------|---------|-----------|-----|------|------|-------|
| Mar 24 | v1 (no gradient dirs) | 6.7e-10 | 0.036 | 0.053 | 430s | FA~0 — direction-blind |
| Mar 24 | v2 (direction-aware, MSE loss) | 2.7e-9 | 0.281 | 0.005 | 252s | FA recovered, MD too high |
| Mar 25 | **v2 + log-space loss** | **7.4e-10** | **0.417** | 0.66 | ~250s | **MD + FA both correct** |

Expected WM values: MD ~ 0.7e-9 m²/s, FA ~ 0.4-0.7

### Sweep: fixing MD overestimation (8 variants, Slurm parallel)

| Variant | MD (m²/s) | FA | Key change |
|---------|-----------|-----|-----------|
| **log_signal** | **7.4e-10** | **0.417** | Log-space loss — correct MD + FA |
| **combo_best** | **7.4e-10** | **0.417** | Log + regularization + more spatial |
| baseline (MSE) | 2.75e-9 | 0.280 | MSE dominated by b=0 shell |
| D_scale_1e-10 | 2.73e-9 | 0.279 | Tighter scale didn't help |
| more_spatial (128) | 2.75e-9 | 0.280 | More MC samples — no difference |
| D_regularized | 2.60e-9 | 0.124 | Regularization hurt FA |

### Key findings:
- v1→v2: FA 0.036→0.281 by conditioning signal on gradient direction
- **Log-space loss is the critical fix**: MD 2.7e-9→7.4e-10, FA 0.28→0.42
- MSE loss dominated by b=0 shell (S~1.0); log loss weights all shells equally
- D-scale and spatial MC sample count don't matter much
- D regularization over-constrains and hurts FA

### What this is NOT:
- Not a PINN (no PDE residual in the loss)
- Not model-free in the fullest sense (uses Stejskal-Tanner equation which assumes Gaussian diffusion)
- The Bloch-Torrey PDE constraint (`pde_loss`, `train_pinn!`) is implemented but not yet active
- True PINN would add value for restricted diffusion (AxCaliber) where Stejskal-Tanner breaks down

---

## 5. Cross-validation

| Test | Result |
|------|--------|
| Microstructure.jl compartments (Cylinder, Zeppelin, Iso, Sphere) | PASS at 1e-13 |
| KomaMRI signal properties (1000 random configs) | 1000/1000 PASS |
| Julia analytical tests (88 tests) | ALL PASS |
| Julia physics invariants (264 tests) | ALL PASS |
| Julia test suite total | 477 PASS, 9 specs for future |

---

## 6. DiffEq Sampler Benchmark (Julia)

**Method:** DifferentialEquations.jl SDE/ODE solvers with dummy score function.
Measures solver overhead only — with a real score network, network evaluation dominates.

| Solver | Steps/s |
|--------|---------|
| DiffEq EM (dt=0.01, 100 steps) | 21,708 |
| DiffEq EM (dt=0.002, 500 steps) | 10,716 |
| DiffEq ODE Tsit5 (adaptive) | 8,788 |

---

## Infrastructure

- **DGX Spark**: Grace CPU (20 ARM cores, aarch64) + GB10 GPU, 120GB unified RAM
- **Slurm**: CPU + GPU partitions, fsl_sub with Slurm plugin
- **FSL**: CUDA eddy_cuda11.0 on GB10 (266 CHARMED volumes in ~2h)
- **Trackio**: HuggingFace experiment tracking
- **HF Hub**: model checkpoint push/pull
- **KomaMRI.jl**: Bloch simulation validation oracle
- **Microstructure.jl** (Ting Gong, MGH): cross-validated at machine precision
- **WAND dataset**: Welsh Advanced Neuroimaging Database, Connectom 300mT/m
