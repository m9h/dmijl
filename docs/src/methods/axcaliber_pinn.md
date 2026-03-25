# AxCaliber PINN

## The physics

Inside an axon (modeled as a cylinder of radius R), water molecules
diffuse freely along the fiber axis but are restricted perpendicular
to it. The dMRI signal depends on how far molecules travel during the
diffusion time Δ:

- **Short Δ**: molecules don't reach the cylinder wall → Gaussian
  behavior, signal ≈ exp(-bD)
- **Long Δ**: molecules hit the wall and bounce back → restricted
  diffusion, signal deviates from exp(-bD) and depends on R

The **Van Gelderen (1994) GPD approximation** gives the signal
attenuation for perpendicular diffusion inside a cylinder:

```
ln(S/S₀) = -Σₖ [2γ²G² / αₖ²(αₖ²R² - 1)] ×
            [2δ/(Dαₖ²) - (2 + e^{-Dαₖ²(Δ-δ)} - 2e^{-Dαₖ²δ}
             - 2e^{-Dαₖ²Δ} + e^{-Dαₖ²(Δ+δ)}) / (Dαₖ²)²]
```

where αₖ are roots of J'₁(αR) = 0. The key: the signal depends on
both b-value AND diffusion time Δ, and the Δ-dependence encodes R.

## Multi-compartment model

The full voxel signal combines intra-cellular (restricted) and
extra-cellular (hindered Gaussian) compartments:

```
S(b, g, δ, Δ) = f × S_restricted(b, g, δ, Δ, D_intra, R)
              + (1-f) × S_hindered(b, g, D_extra)
```

For the intra-cellular component, we decompose into parallel
(free diffusion) and perpendicular (Van Gelderen restricted):

```
S_restricted = S_perp(b⊥, δ, Δ, D_intra, R) × exp(-b∥ D_intra)
```

where b⊥ = b sin²θ and b∥ = b cos²θ, with θ the angle between
the gradient direction and the fiber axis.

## Network architecture

```
Concatenated multi-Δ signals (264 values)
         │
         ▼
┌─────────────────────┐
│  MLP (128-wide, 5   │
│  layers, GELU)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  7 outputs:         │
│  R (softplus)       │
│  D_intra (softplus) │
│  D_extra (softplus) │
│  f (sigmoid)        │
│  μx, μy, μz (L2)   │
└─────────────────────┘
```

The network sees ALL four AxCaliber acquisitions simultaneously.
The different Δ values create a distinct signal pattern for each
axon radius — the network learns this mapping.

## Why this is a PINN

The loss function evaluates the Van Gelderen model at the predicted
geometry parameters for each (b, Δ) measurement:

```
L = Σᵢ |S_observed(bᵢ, Δᵢ) - S_VanGelderen(bᵢ, Δᵢ; R, D, f, μ)|²
```

The Van Gelderen model IS the PDE solution (GPD approximation to the
Bloch-Torrey equation for a cylinder). The physics is in the forward
model, not in an explicit PDE residual term — but the network is
constrained to produce parameters that are consistent with the physics
of restricted diffusion.

## Usage

```julia
using DMI, Lux, Random

# Load multi-Δ data
data = AxCaliberData(signals, bvalues, bvecs, deltas, Deltas)

# Build and train
model = build_axcaliber_pinn(; signal_dim=264, hidden_dim=128, depth=5)
ps, st = Lux.setup(MersenneTwister(42), model)

ps, st, geom, losses = train_axcaliber_pinn!(model, ps, st, data;
    n_steps=5000, lambda_physics=1.0)

# Results
println("Axon radius: $(geom.R * 1e6) μm")
println("Intra fraction: $(geom.f_intra)")
println("Fiber direction: $(geom.mu)")
```

## Limitations

- Single-fiber model (no crossing fibers)
- Assumes cylindrical geometry (not elliptical or beaded axons)
- GPD approximation may be inaccurate for very short gradient pulses
- Currently fits one voxel at a time (no spatial regularization)
- D_intra and D_extra estimates less reliable than R and f
