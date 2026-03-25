# Getting Started

## Installation

```julia
using Pkg
Pkg.develop(url="https://github.com/m9h/dmijl")
```

### GPU support

DMI.jl auto-detects NVIDIA GPUs via LuxCUDA:

```julia
using DMI
dev = select_device()  # returns gpu_device() or cpu_device()
```

### Dependencies

Core dependencies are installed automatically. Optional:

- **KomaMRI.jl** — for Bloch simulation validation
- **MCMRSimulator.jl** — for Monte Carlo ground truth (requires FMRIB GitLab access)
- **Microstructure.jl** — for cross-validation with MGH/Martinos compartment models

## Your first AxCaliber fit

The AxCaliber PINN recovers axon radius from multi-delta dMRI data.
Here's a minimal example with synthetic data:

```julia
using DMI, Lux, Random

rng = MersenneTwister(42)

# Generate synthetic AxCaliber data for a 3μm cylinder
D_intra = 1.7e-9  # m²/s
R_true = 3e-6     # 3 μm
f_true = 0.5      # 50% intra-cellular

# 4 acquisitions with different Δ (diffusion time)
Deltas = [18e-3, 30e-3, 42e-3, 55e-3]
delta = 11e-3  # gradient pulse duration

signals = Vector{Float32}[]
bvalues = Vector{Float64}[]
bvecs_list = Matrix{Float64}[]

for Delta in Deltas
    bv = [0.0, 2000e6, 5000e6, 10000e6]  # s/m²
    n_dirs = 15
    gdir = randn(rng, n_dirs * length(bv), 3)
    for i in axes(gdir, 1)
        gdir[i, :] ./= max(norm(gdir[i, :]), 1e-8)
    end

    sig = Float32[]
    bv_full = Float64[]
    for b in bv
        for j in 1:n_dirs
            g = gdir[(length(sig) % size(gdir, 1)) + 1, :]
            S = axcaliber_signal(b, delta, Delta, D_intra, 0.8e-9, R_true, f_true,
                                g, [0.0, 0.0, 1.0])
            push!(sig, Float32(S + 0.02 * randn(rng)))
            push!(bv_full, b / 1e6)  # store as s/mm²
        end
    end
    push!(signals, sig)
    push!(bvalues, bv_full)
    push!(bvecs_list, gdir[1:length(sig), :])
end

data = AxCaliberData(signals, bvalues, bvecs_list,
                     fill(delta, 4), Deltas)

# Build and train PINN
model = build_axcaliber_pinn(; signal_dim=sum(length.(signals)),
                               hidden_dim=128, depth=5)
ps, st = Lux.setup(rng, model)

ps, st, geom, losses = train_axcaliber_pinn!(model, ps, st, data;
    n_steps=3000, learning_rate=1e-3)

println("Recovered R = $(round(geom.R * 1e6, digits=2)) μm (true: 3.0 μm)")
println("Recovered f = $(round(geom.f_intra, digits=2)) (true: 0.5)")
```

## Your first D(r) field recovery

Recover the diffusion tensor field from multi-shell data without
assuming any geometric compartment model:

```julia
using DMI

# Load your preprocessed dMRI data
signal = load_signal(...)  # Float32 vector, b0-normalized
bvals = load_bvals(...)    # s/m²
bvecs = load_bvecs(...)    # (n_meas, 3)

problem = DiffusionFieldProblem(signal, bvals, bvecs,
    10e-3,   # δ (gradient pulse duration)
    40e-3,   # Δ (diffusion time)
    80e-3,   # T₂
    2e-3,    # voxel size (meters)
)

result = solve_diffusion_field_v2(problem;
    output_type = :diagonal,  # or :scalar, :full
    n_steps = 5000,
    learning_rate = 1e-3,
)

maps = extract_maps(result; grid_resolution=8)
println("MD = $(maps.MD) m²/s")
println("FA = $(maps.FA)")
```

## Next steps

- [Results](@ref) — see validated numbers on WAND Connectom data
- [AxCaliber PINN](@ref) — deep dive into the physics
- [Validation](@ref) — how we verified against Microstructure.jl and KomaMRI
