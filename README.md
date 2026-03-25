<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/DMI.jl-Diffusion_Microstructural_Imaging-8B5CF6?style=for-the-badge&logo=julia&logoColor=white">
    <img alt="DMI.jl" src="https://img.shields.io/badge/DMI.jl-Diffusion_Microstructural_Imaging-6D28D9?style=for-the-badge&logo=julia&logoColor=white">
  </picture>
</p>

<p align="center">
  <strong>Score-based posterior inference for diffusion MRI microstructure — built on Julia's SciML stack.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/julia-%3E%3D1.10-9558B2?style=flat-square&logo=julia" alt="Julia 1.10+">
  <img src="https://img.shields.io/badge/Lux.jl-neural_nets-E91E63?style=flat-square" alt="Lux.jl">
  <img src="https://img.shields.io/badge/DifferentialEquations.jl-SDE_solvers-FF6F00?style=flat-square" alt="DiffEq">
  <img src="https://img.shields.io/badge/CUDA-GPU_ready-76B900?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/tests-558_passing-22C55E?style=flat-square" alt="Tests">
</p>

---

## Why DMI.jl?

Traditional dMRI microstructure pipelines fit point estimates from hand-crafted models. **DMI.jl** takes a different approach — it learns the **full Bayesian posterior** over tissue parameters using score-based generative models, giving you uncertainty maps alongside parameter maps.

| What you get | How it works |
|:---|:---|
| 🧠 **Full posterior distributions** | Score-based diffusion models conditioned on observed signals |
| ⚡ **Native SDE/ODE solvers** | DifferentialEquations.jl — no XLA compilation wall |
| 🔬 **Monte Carlo ground truth** | MCMRSimulator.jl integration for physics-validated training data |
| 🗺️ **Non-parametric D(r) fields** | Recover spatially-varying diffusion tensors without geometric models |
| 🏗️ **Physics-informed surrogates** | Bloch-Torrey PINNs for fast forward simulation |
| 🎯 **Multiple tissue models** | Ball+Stick, DTI, NODDI via multiple dispatch |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          DMI.jl                                 │
├─────────────────┬──────────────────┬────────────────────────────┤
│  Forward Models │  Score Posterior │  Non-parametric Recovery   │
│  ─────────────  │  ──────────────  │  ──────────────────────── │
│  BallStickModel │  ScoreNetwork    │  DiffusionFieldProblem     │
│  DTIModel       │  FiLM + SinEmb   │  solve_diffusion_field     │
│  NODDIModel     │  train_score!    │  solve_diffusion_field_v2  │
│                 │  sample_posterior│  (direction-aware D(r))    │
├─────────────────┼──────────────────┼────────────────────────────┤
│  PINN / Surrogates                │  Validation                │
│  ─────────────────────────────────│  ──────────────────────────│
│  BlochTorreyResidual              │  KomaMRI oracle            │
│  build_surrogate / train_pinn!    │  MCMRSimulator compat      │
│  surrogate_sbi pipeline          │  angular_error, RMSE, r    │
├───────────────────────────────────┴────────────────────────────┤
│  Infrastructure: GPU auto-detect · Rician noise · Acquisition  │
│  Lux.jl · Zygote AD · ComponentArrays · DifferentialEquations  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```julia
using Pkg
Pkg.develop(path="/path/to/dmijl")
```

<details>
<summary><strong>GPU support</strong></summary>

DMI.jl auto-detects CUDA GPUs via `LuxCUDA`. Just ensure your system has a working CUDA toolkit:

```julia
using DMI
dev = select_device()   # auto-detects GPU or falls back to CPU
```

</details>

---

## Quick Start

### Score-based posterior inference

Train a conditional score model on synthetic Ball+Stick data, then sample the full posterior for observed signals:

```julia
using DMI, Random

rng = MersenneTwister(42)
acq = hcp_like_acquisition()

# Generate training data
model = BallStickModel()
sim = ModelSimulator(model, acq)
θ, S = sample_and_simulate(sim, rng, 50_000)
S_noisy = add_rician_noise(S, 0.05, rng)

# Train score network
net, ps, st = build_score_net(; obs_dim=size(S, 1), param_dim=size(θ, 1))
schedule = VPSchedule()
ps, st = train_score!(net, ps, st, schedule, θ, S_noisy, rng;
                      n_epochs=200, batch_size=512)

# Sample posterior for a new observation
S_obs = S_noisy[:, 1]
posterior_samples = sample_posterior(net, ps, st, schedule, S_obs, rng;
                                    n_samples=1000)
```

### Non-parametric D(r) recovery

Recover the spatially-varying diffusion tensor field directly from signal — no geometric model assumptions:

```julia
using DMI

problem = DiffusionFieldProblem(;
    observed_signal = S_obs,
    bvalues = acq.bvalues,
    gradient_directions = acq.gradient_directions,
    voxel_size = 2.0,
)

result = solve_diffusion_field_v2(problem;
    output_type = :diagonal,
    n_steps = 10_000,
)

maps = extract_maps(result.D_net, result.ps_D, result.st_D, result.D_type)
# maps.FA, maps.MD, maps.AD, maps.RD
```

---

## Examples

| Script | Description |
|:---|:---|
| [`ball2stick_score.jl`](examples/ball2stick_score.jl) | End-to-end score-based inference on Ball+2Stick |
| [`ds001957_inference.jl`](examples/ds001957_inference.jl) | Real data inference on BIDS dataset (NIfTI + bval/bvec) |
| [`mcmr_restricted_diffusion.jl`](examples/mcmr_restricted_diffusion.jl) | Score model trained on MCMRSimulator ground truth |
| [`pinn_bloch_torrey.jl`](examples/pinn_bloch_torrey.jl) | Bloch-Torrey PINN surrogate training |
| [`benchmark_samplers.jl`](examples/benchmark_samplers.jl) | Euler vs DiffEq SDE/ODE sampler comparison |
| [`benchmark_vs_conventional.jl`](examples/benchmark_vs_conventional.jl) | DMI.jl vs conventional fitting pipelines |

---

## Project Structure

```
dmijl/
├── src/
│   ├── DMI.jl                      # Main module
│   ├── gpu.jl                      # CUDA device auto-detection
│   ├── noise.jl                    # Rician noise model
│   ├── models/
│   │   ├── ball_stick.jl           # Ball+Stick forward model
│   │   ├── dti.jl                  # Diffusion Tensor Imaging
│   │   └── noddi.jl                # NODDI (Watson distribution)
│   ├── diffusion/
│   │   ├── schedule.jl             # VP noise schedule
│   │   ├── score_net.jl            # FiLM-conditioned score network
│   │   ├── train.jl                # Denoising score matching
│   │   ├── sample.jl               # Euler-Maruyama reverse SDE
│   │   └── sample_diffeq.jl        # DifferentialEquations.jl samplers
│   ├── pinn/
│   │   ├── bloch_torrey.jl         # Bloch-Torrey PINN surrogate
│   │   ├── diffusion_field.jl      # Non-parametric D(r) recovery
│   │   └── diffusion_field_v2.jl   # Direction-aware D(r) (tensor)
│   ├── pipeline/
│   │   ├── acquisition.jl          # dMRI acquisition protocols
│   │   ├── config.jl               # SBI configuration
│   │   ├── simulator.jl            # Forward simulation pipeline
│   │   ├── mcmr_generator.jl       # MCMRSimulator data generation
│   │   └── surrogate_sbi.jl        # Surrogate-accelerated SBI
│   ├── compat/
│   │   └── microstructure_jl.jl    # Ting Gong's Microstructure.jl compat
│   └── validation/
│       ├── metrics.jl              # Angular error, RMSE, Pearson r
│       └── koma_oracle.jl          # KomaMRI validation oracle
├── test/                           # 558 passing tests
├── examples/                       # Runnable scripts
├── slurm/                          # HPC batch scripts
└── results/                        # Surrogate sweep outputs
```

---

## Key Dependencies

| Package | Role |
|:---|:---|
| [Lux.jl](https://github.com/LuxDL/Lux.jl) | Neural network layers (pure functional) |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl) | Automatic differentiation |
| [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) | SDE/ODE reverse samplers |
| [MCMRSimulator.jl](https://git.fmrib.ox.ac.uk/ndcn0236/MCMRSimulator.jl) | Monte Carlo MR forward simulation |
| [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) | MRI sequence simulation oracle |
| [LuxCUDA](https://github.com/LuxDL/LuxCUDA.jl) | GPU acceleration |

---

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

---

## License

TBD

---

<p align="center">
  <sub>Built with Julia's <a href="https://sciml.ai/">SciML</a> ecosystem</sub>
</p>
