# Forward Models

DMI.jl includes analytical forward models for standard dMRI compartments,
implemented with Julia's multiple dispatch.

## Ball+2Stick

Two intra-cellular stick compartments plus isotropic (ball) extra-cellular:

```
S = f₁ exp(-bD cos²θ₁) + f₂ exp(-bD cos²θ₂) + f_ball exp(-bD_ball)
```

```julia
model = BallStickModel(bvalues, gradient_directions)
signal = simulate(model, [d_ball, d_stick, f1, f2, mu1..., mu2...])
```

## DTI

Full diffusion tensor with 6 parameters (3 eigenvalues + 3 Euler angles):

```julia
model = DTIModel(bvalues, gradient_directions)
signal = simulate(model, [λ1, λ2, λ3, θ, φ, ψ])
```

Derived metrics: `compute_fa`, `compute_md`, `compute_ad`, `compute_rd`

## NODDI

Neurite orientation dispersion and density imaging with Watson distribution:

```julia
model = NODDIModel(bvalues, gradient_directions)
signal = simulate(model, [f_intra, f_iso, kappa, d_par, mu...])
```

Derived: `kappa_to_odi(kappa)` converts Watson κ to orientation dispersion index.

## Van Gelderen cylinder

Restricted diffusion inside a cylinder — the physics behind AxCaliber:

```julia
S = van_gelderen_cylinder(b, delta, Delta, D, R; n_terms=10)
```

Multi-compartment AxCaliber signal:
```julia
S = axcaliber_signal(b, delta, Delta, D_intra, D_extra, R, f_intra, g, mu)
```

## Cross-validation

All models cross-validated against Ting Gong's
[Microstructure.jl](https://github.com/TingGong/Microstructure.jl) (MGH/Martinos)
at machine precision (relative error < 1e-13).
