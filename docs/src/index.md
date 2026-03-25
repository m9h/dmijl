# DMI.jl

**Diffusion Microstructural Imaging in Julia**

DMI.jl recovers tissue microstructure from diffusion MRI using physics-informed
neural networks and the Julia SciML ecosystem.

## What it does

- **AxCaliber PINN**: Recovers axon radius from multi-delta dMRI using
  Van Gelderen restricted diffusion physics. Validated on real Siemens
  Connectom data (R = 3.15 μm on WAND sub-00395).

- **Neural diffusion tensor field**: Fits spatially-varying D(x,y,z) from
  multi-shell dMRI without assuming geometric compartment models.
  Direction-aware Stejskal-Tanner signal prediction recovers FA = 0.42
  and MD = 0.74 μm²/ms on real white matter.

- **Score-based posterior**: Full Bayesian posterior over microstructure
  parameters via denoising score matching with DDPM sampling.

## Why Julia?

- **No compilation wall**: Julia's JIT compiles incrementally — no 30-60 min
  XLA compilation per experiment like JAX. Critical for rapid iteration.
- **Native SDE/ODE solvers**: DifferentialEquations.jl solves the reverse
  diffusion SDE with adaptive stepping, error control, and automatic
  differentiation through the solver.
- **Composable AD**: Zygote (reverse-mode) + ForwardDiff (forward-mode)
  compose for the nested derivatives needed by PINN training (∂²M/∂x²
  for the Laplacian in Bloch-Torrey).

## Quick links

- [Getting Started](@ref) — install and run your first example
- [Results](@ref) — validated numbers on real data
- [AxCaliber PINN](@ref) — the main contribution
- [API Reference](@ref) — all exported functions
- [SBI4DWI companion](@ref) — the Python/JAX sibling project
