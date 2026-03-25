# Companion: SBI4DWI (Python/JAX)

**[SBI4DWI](https://github.com/m9h/sbi4dwi)** is the Python/JAX sibling
project. Both packages share the same forward models, validation data,
and research goals.

## Division of labor

| Feature | DMI.jl (Julia) | SBI4DWI (Python/JAX) |
|:--------|:---------------|:---------------------|
| AxCaliber PINN | Yes (Van Gelderen) | Bloch-Torrey prototype |
| Neural tensor field | Yes (v2, direction-aware) | — |
| Flow NPE | — | Yes (3.2° orientation, FlowJAX) |
| Score posterior | Yes (Lux.jl + DiffEq) | Yes (12.8°, MLP + DDPM) |
| MDN posterior | — | Yes (Equinox) |
| MCMC (NUTS) | — | Yes (BlackJAX) |
| Conformal prediction | — | Yes |
| SBC calibration | — | Yes |
| OOD detection | — | Yes |
| Ensembles | — | Yes |
| NIfTI deployment | — | Yes (SBIPredictor) |
| E(3) equivariant nets | — | Yes (e3nn-jax, prototype) |
| GPU training | CUDA.jl (LuxCUDA) | JAX (XLA) |
| Compilation | Incremental JIT | XLA whole-program |

## Why two languages?

**Julia advantages** (DMI.jl):
- No 30-60 min XLA compilation wall → fast autoresearch iteration
- DifferentialEquations.jl for native SDE/ODE sampling
- Composable AD (Zygote + ForwardDiff) for PINN Laplacian computation
- Direct integration with KomaMRI.jl and MCMRSimulator.jl

**Python/JAX advantages** (SBI4DWI):
- Mature normalizing flow libraries (FlowJAX) → 3.2° orientation error
- e3nn-jax for equivariant neural networks
- Broader ecosystem (nibabel, DIPY, PyPulseq)
- Clinical deployment tools (NIfTI volume inference)

## Shared infrastructure

- Cross-validated forward models (machine precision agreement)
- Same WAND Connectom test data (sub-00395)
- Same Ball+2Stick, DTI, NODDI model parameterizations
- Results tracked on common leaderboard
- Experiment tracking via Trackio (HuggingFace)
