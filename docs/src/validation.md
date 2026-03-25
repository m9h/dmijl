# Validation

## Cross-validation with Microstructure.jl

Ting Gong's [Microstructure.jl](https://github.com/TingGong/Microstructure.jl)
(MGH/Martinos Center) provides MCMC-based microstructure fitting with
compartment models (Cylinder, Stick, Zeppelin, Iso, Sphere).

We cross-validate our forward model signals against their reference
implementations using the exact same protocol (Connectom-style, 8 b-values
up to 43,000 s/mm²):

| Compartment | Max relative error |
|:------------|:-------------------|
| Cylinder (da=2μm) | 2.1e-15 |
| Zeppelin (default) | 2.5e-13 |
| Iso (default) | 6.9e-14 |
| Sphere (default) | 4.7e-15 |

Machine precision agreement. This means:
- Data prepared for Microstructure.jl works with DMI.jl
- Results are directly comparable
- The forward models are numerically identical

```julia
using DMI
cross_validate_compartments()  # runs automatically
```

## KomaMRI validation

KomaMRI.jl provides independent Bloch simulation. We validate that
our forward models produce physically reasonable signals:

```julia
using DMI
acq = hcp_like_acquisition()
model = BallStickModel(acq.bvalues, acq.gradient_directions)
validate_signal_properties_koma(model; n_test=1000)
# → 1000/1000 PASS
```

Checks: signal positivity, bounds [0,1], b=0 normalization, monotonic
shell decay, Rician noise properties.

## Test suite

| Suite | Tests | Status |
|:------|:------|:-------|
| Analytical solutions | 88 | ALL PASS |
| Physics invariants | 264 | ALL PASS |
| Van Gelderen restricted diffusion | 112 | ALL PASS |
| Score posterior | 119 | PASS (3 specs) |
| Surrogate | 11 | 6 PASS, 5 specs |
| Microstructure.jl cross-validation | 4 | ALL PASS |

Run the full suite:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## WAND data preprocessing

Real data validation uses the WAND dataset (Welsh Advanced Neuroimaging
Database) from CUBRIC, Cardiff University. Acquired on a Siemens Connectom
with 300 mT/m gradients.

Preprocessing: FSL topup + eddy (CUDA-accelerated on NVIDIA GB10).

See `scripts/wand/` for the preprocessing pipeline.
