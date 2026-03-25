# Neural Diffusion Tensor Field

## Approach

Instead of fitting a parametric compartment model (Ball+Stick, NODDI),
we learn the diffusion tensor D(x,y,z) directly as a neural field.

A small MLP maps each spatial position x to a diffusion tensor:
- **Scalar**: D(x) → single isotropic diffusivity
- **Diagonal**: D(x) → [D₁, D₂, D₃] eigenvalues
- **Full**: D(x) → 3×3 SPD matrix (via Cholesky LLᵀ)

The signal for each measurement (b, g) is predicted via the
Stejskal-Tanner equation with Monte Carlo integration over the voxel:

```
S(b, g) ≈ (1/N) Σᵢ exp(-b · gᵀ D(xᵢ) g)
```

where xᵢ are randomly sampled positions within the voxel.

## Key insight: log-space loss

MSE loss on raw signals is dominated by the b=0 shell (signal ≈ 1.0),
causing the network to underfit high-b attenuation and overestimate MD.

**Log-space loss** `(log S_pred - log S_obs)²` weights all shells
equally, which is critical for correct diffusivity magnitude:

| Loss | MD (m²/s) | FA | Correct? |
|:-----|:----------|:---|:---------|
| MSE | 2.7e-9 | 0.28 | MD 4x too high |
| **Log-space** | **7.4e-10** | **0.42** | Both correct |

## Honest label

This is **not a PINN**. The Stejskal-Tanner equation assumes Gaussian
diffusion — there is no Bloch-Torrey PDE residual in the loss. It works
well for white matter where diffusion is approximately Gaussian within
each voxel, but would fail for restricted diffusion (use the AxCaliber
PINN for that).

## Usage

```julia
using DMI

problem = DiffusionFieldProblem(signal, bvals, bvecs,
    delta, Delta, T2, voxel_size)

result = solve_diffusion_field_v2(problem;
    output_type = :diagonal,
    D_hidden = 64, D_depth = 4,
    n_steps = 5000,
)

maps = extract_maps(result; grid_resolution=8)
# maps.FA, maps.MD
```
