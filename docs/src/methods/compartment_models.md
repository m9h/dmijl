# Composable Compartment Models

## Design

DMI.jl uses dmipy-style composable compartments. Each compartment is a
Julia struct that implements the `AbstractCompartment` interface:

- `signal(compartment, acq, params)` — compute signal attenuation
- `parameter_names(compartment)` — list of parameter names
- `parameter_ranges(compartment)` — prior bounds per parameter

Compartments compose via `MultiCompartmentModel` with volume fractions.

## Available compartments

### G1Ball — isotropic Gaussian diffusion

```julia
ball = G1Ball()
# Parameters: [lambda_iso]
# Signal: exp(-b * lambda_iso)
```

### C1Stick — intra-axonal (zero-radius cylinder)

```julia
stick = C1Stick()
# Parameters: [lambda_par, mu_x, mu_y, mu_z]
# Signal: exp(-b * lambda_par * cos²θ)
```

### G2Zeppelin — extra-cellular (axially symmetric tensor)

```julia
zep = G2Zeppelin()
# Parameters: [lambda_par, lambda_perp, mu_x, mu_y, mu_z]
# Signal: exp(-b * (lambda_perp + (lambda_par - lambda_perp) * cos²θ))
```

### S1Dot — stationary water (no diffusion)

```julia
dot = S1Dot()
# Parameters: none
# Signal: 1.0 for all b-values
```

## Multi-compartment composition

```julia
mcm = MultiCompartmentModel([C1Stick(), G1Ball()])
# Automatically adds volume fractions: f_stick, f_ball (sum to 1)
# Full parameter vector: [lambda_par, mu_x, mu_y, mu_z, lambda_iso, f_stick]
```

## Constraints

```julia
# Fix a parameter to a known value
cm = ConstrainedModel(mcm)
set_fixed_parameter(cm, "G1Ball_lambda_iso", 3.0e-9)  # CSF diffusivity

# Tortuosity constraint: lambda_perp = lambda_par * (1 - f_intra)
set_tortuosity(cm, "G2Zeppelin_lambda_perp", "C1Stick_lambda_par", "f_stick")
```

## Watson orientation dispersion

Convolve any compartment with a Watson distribution:

```julia
watson = WatsonDistribution(; n_grid=300)
dm = DistributedModel(C1Stick(), watson)
# Parameters: [lambda_par, mu_x, mu_y, mu_z, kappa]
# Averages the stick signal over Watson-distributed orientations
```

## Fitting

```julia
result = fit_mcm(mcm, acquisition, signal; n_restarts=5)
# Returns best-fit parameters via NLLS optimization
```

Batch fitting over multiple voxels:
```julia
results = fit_mcm_batch(mcm, acquisition, signals; n_restarts=5)
```
