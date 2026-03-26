# DMI.jl — Open Tasks

## MRIBuilder.jl dependency — RESOLVED

MCMRSimulator.jl depends on MRIBuilder.jl (Cottaar, FMRIB), which is not in
the Julia General registry. Source downloaded to `~/dev/MRIBuilder.jl` and
linked via `Pkg.develop(path=...)`. Manifest now points to local checkout.

## softplus method overwrite (Julia 1.12)

`axcaliber_pinn.jl` and `diffusion_field.jl` both define `softplus(x)`.
Julia 1.12 forbids method overwriting during precompilation. Need to extract
`softplus` into a shared utility or use `NNlib.softplus` instead.

## Restricted diffusion compartments

Port from dmipy-jax to the composable framework:
- `RestrictedCylinder` (Soderman/Callaghan approximation) — needs delta/Delta from Acquisition
- `SphereGPD` (Gaussian Phase Distribution for soma) — needed for SANDI
- `PlaneCallaghan` — restricted diffusion between parallel planes

## NODDI convenience constructor

All pieces exist: `DistributedModel(C1Stick, Watson)` + tortuosity-constrained
`G2Zeppelin` + `G1Ball`. Build a `noddi_watson()` factory function that wires
them together with standard constraints (fixed d_par=1.7e-9, tortuosity, fraction unity).

## Examples migration

Scripts in `examples/` use raw `include("../src/...")` instead of `using DMI`.
Migrate to use the module API.
