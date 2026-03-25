<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/DMI.jl-Diffusion_Microstructural_Imaging-8B5CF6?style=for-the-badge&logo=julia&logoColor=white">
    <img alt="DMI.jl" src="https://img.shields.io/badge/DMI.jl-Diffusion_Microstructural_Imaging-6D28D9?style=for-the-badge&logo=julia&logoColor=white">
  </picture>
</p>

<p align="center">
  <strong>Physics-informed neural networks and score-based inference for diffusion MRI microstructure.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/julia-%3E%3D1.10-9558B2?style=flat-square&logo=julia" alt="Julia 1.10+">
  <img src="https://img.shields.io/badge/Lux.jl-neural_nets-E91E63?style=flat-square" alt="Lux.jl">
  <img src="https://img.shields.io/badge/DifferentialEquations.jl-SDE_solvers-FF6F00?style=flat-square" alt="DiffEq">
  <img src="https://img.shields.io/badge/CUDA-GPU_ready-76B900?style=flat-square&logo=nvidia" alt="CUDA">
</p>

---

## Results on Real Data

Validated on **WAND** (Welsh Advanced Neuroimaging Database) — Siemens Connectom 300 mT/m scanner.

### AxCaliber PINN: axon radius from restricted diffusion

The Van Gelderen restricted diffusion model recovers compartment geometry
from multi-delta AxCaliber data. This is a proper PINN: the physics of
restricted diffusion inside cylinders constrains the inverse problem.
Stejskal-Tanner (Gaussian diffusion) cannot recover axon radius.

| Parameter | Recovered | Expected WM |
|:----------|:----------|:------------|
| Axon radius R | **3.15 um** | 2-5 um |
| Intra-cellular fraction | **0.46** | 0.4-0.7 |
| D intra-cellular | 4.6e-10 m^2/s | 1-2e-9 |
| Fiber direction | [0.98, 0.14, 0.14] | — |

Data: sub-00395, 4 AxCaliber acquisitions (delta = 18/30/42/55 ms, b up to 15,500 s/mm^2).
Training: 168 seconds on DGX Spark Grace CPU.

### Neural diffusion tensor field: direction-aware fitting

Recovers spatially-varying D(x) from CHARMED data (7 shells, b=0-6000 s/mm^2)
using the Stejskal-Tanner equation with direction-dependent signal prediction.
Not a PINN — no PDE residual, just physics-based signal model.

| Metric | Recovered | Expected WM |
|:-------|:----------|:------------|
| MD | **7.4e-10 m^2/s** | ~0.7e-9 |
| FA | **0.42** | 0.4-0.7 |

Key insight: log-space loss (not MSE) is critical for correct MD.

### Cross-validation

| Test | Result |
|:-----|:-------|
| Microstructure.jl compartments (Cylinder, Zeppelin, Iso, Sphere) | PASS at 1e-13 |
| KomaMRI signal properties (1000 random configurations) | 1000/1000 PASS |
| Van Gelderen restricted diffusion (112 physics tests) | ALL PASS |

---

## What DMI.jl does

| Capability | Method | Status |
|:-----------|:-------|:-------|
| **Composable signal models** | G1Ball, C1Stick, G2Zeppelin, S1Dot + multi-compartment | Tested, dmipy-compatible |
| **Orientation dispersion** | Watson distribution on Fibonacci sphere grid | Tested |
| **Multi-compartment fitting** | NLLS via Optimisers.jl | Working |
| **Axon radius estimation** | AxCaliber PINN (Van Gelderen) | Validated on real data |
| **Diffusion tensor field** | Neural field + Stejskal-Tanner | FA=0.42, MD correct |
| **Score-based posteriors** | Denoising score matching + DDPM | 12.8 deg orientation error |
| **Forward model surrogate** | Supervised MLP regression | 0.96% error, spec passed |
| **Native SDE/ODE samplers** | DifferentialEquations.jl | EM 21k samples/s |

---

## Architecture

<table>
<tr>
<th>Compartment Models</th>
<th>Composition</th>
<th>Inference</th>
</tr>
<tr>
<td>
<code>G1Ball</code> (isotropic)<br>
<code>C1Stick</code> (intra-axonal)<br>
<code>G2Zeppelin</code> (extra-cellular)<br>
<code>S1Dot</code> (stationary water)
</td>
<td>
<code>MultiCompartmentModel</code><br>
<code>ConstrainedModel</code><br>
<code>WatsonDistribution</code><br>
<code>DistributedModel</code>
</td>
<td>
<code>fit_mcm</code> (NLLS)<br>
<code>ScoreNetwork</code> (FiLM)<br>
<code>train_score!</code><br>
<code>sample_posterior</code>
</td>
</tr>
<tr>
<th>PINNs</th>
<th>Tensor Field Recovery</th>
<th>Validation</th>
</tr>
<tr>
<td>
<code>AxCaliberData</code><br>
<code>build_axcaliber_pinn</code><br>
<code>train_axcaliber_pinn!</code><br>
<code>BlochTorreyResidual</code>
</td>
<td>
<code>DiffusionFieldProblem</code><br>
<code>solve_diffusion_field_v2</code><br>
<code>extract_maps</code> (FA, MD)
</td>
<td>
KomaMRI oracle<br>
Microstructure.jl compat<br>
<code>angular_error</code> / <code>rmse</code>
</td>
</tr>
</table>

---

## Installation

```julia
using Pkg
Pkg.develop(url="https://github.com/m9h/dmijl")
```

GPU support auto-detects via `LuxCUDA`:

```julia
using DMI
dev = select_device()  # auto-detects GPU or falls back to CPU
```

---

## Quick Start

### Composable multi-compartment fitting

```julia
using DMI

# Build a Ball+Stick model (like dmipy)
mcm = MultiCompartmentModel([C1Stick(), G1Ball()])

# Add constraints
cm = ConstrainedModel(mcm)
set_fixed_parameter(cm, "G1Ball_lambda_iso", 3.0e-9)  # fix CSF diffusivity

# Fit to observed signal
acq = load_acquisition("data.bval", "data.bvec")
result = fit_mcm(cm, acq, signal; n_restarts=5)
```

### With Watson orientation dispersion (NODDI-like)

```julia
watson = WatsonDistribution(; n_grid=300)
dm = DistributedModel(C1Stick(), watson)
mcm = MultiCompartmentModel([dm, G1Ball()])
# Fits: lambda_par, mu, kappa (dispersion), lambda_iso, volume fractions
```

### AxCaliber PINN (restricted diffusion)

```julia
using DMI, Lux, Random

# Load multi-delta AxCaliber data
data = AxCaliberData(
    signals,    # 4 signal vectors (one per acquisition)
    bvalues,    # 4 b-value vectors
    bvecs,      # 4 gradient direction matrices
    deltas,     # [11e-3, 11e-3, 11e-3, 11e-3]  (small delta)
    Deltas,     # [18e-3, 30e-3, 42e-3, 55e-3]  (big delta)
)

# Build and train PINN
model = build_axcaliber_pinn(; signal_dim=264, hidden_dim=128, depth=5)
ps, st = Lux.setup(MersenneTwister(42), model)

ps, st, geom, losses = train_axcaliber_pinn!(model, ps, st, data;
    n_steps=5000, lambda_physics=1.0)

# geom.R       — axon radius (meters)
# geom.D_intra — intra-cellular diffusivity
# geom.f_intra — intra-cellular fraction
# geom.mu      — fiber orientation (unit vector)
```

### Non-parametric D(r) field

```julia
using DMI

problem = DiffusionFieldProblem(signal, bvalues, bvecs, delta, Delta, T2, voxel_size)

result = solve_diffusion_field_v2(problem;
    output_type = :diagonal,
    n_steps = 5000,
)

maps = extract_maps(result; grid_resolution=8)
# maps.FA, maps.MD
```

---

## Companion project

**[SBI4DWI](https://github.com/m9h/sbi4dwi)** (Python/JAX) — normalizing flow NPE
and score-based posteriors for the same microstructure models. Achieves 3.2 deg
median orientation error on Ball+2Stick with neural spline flows (200k steps).

The two projects share:
- Same forward models (Ball+Stick, NODDI, DTI)
- Same WAND Connectom validation data
- Cross-validated against Microstructure.jl (Ting Gong, MGH/Martinos)

---

## Key Dependencies

| Package | Role |
|:--------|:-----|
| [Lux.jl](https://github.com/LuxDL/Lux.jl) | Neural network layers (pure functional) |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl) | Automatic differentiation |
| [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) | SDE/ODE solvers for reverse diffusion |
| [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) | Bloch simulation validation oracle |
| [MCMRSimulator.jl](https://github.com/MichielCottaar/MCMRSimulator.jl) | Monte Carlo forward simulation (Cottaar/Jbabdi/Miller, FMRIB) |
| [Microstructure.jl](https://github.com/TingGong/Microstructure.jl) | Cross-validation reference (Ting Gong, MGH/Martinos) |

---

## Full Leaderboard

See [`results/LEADERBOARD.md`](results/LEADERBOARD.md) for all results across
both DMI.jl and SBI4DWI, including autoresearch sweeps.

---

## License

MIT

---

<p align="center">
  <sub>Built with Julia's <a href="https://sciml.ai/">SciML</a> ecosystem.
  Validated on <a href="https://git.cardiff.ac.uk/cubric/wand">WAND</a> Connectom data.</sub>
</p>
