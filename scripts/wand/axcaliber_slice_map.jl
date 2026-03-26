#!/usr/bin/env julia
"""
AxCaliber PINN: whole-slice axon radius map from WAND Connectom data.

Fits the Van Gelderen restricted diffusion model independently per voxel,
producing R (axon radius) and f (intra-cellular fraction) maps.

Usage:
    julia axcaliber_slice_map.jl [n_steps] [batch_start] [batch_end]
    # defaults: 2000 steps, all voxels
"""

using Random, Statistics, LinearAlgebra, Printf, NPZ, JSON, Lux

const DMIJL = "/home/mhough/dev/dmijl"
include(joinpath(DMIJL, "src/pinn/axcaliber_pinn.jl"))

# Parse args
n_steps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000
batch_start = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
batch_end = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0  # 0 = all

# Load slice data
data_file = "/data/datasets/wand/sub-00395/ses-02/axcaliber/slice_z33.npz"
d = NPZ.npzread(data_file)
coords = d["coords"]
mask_shape = Tuple(d["mask_shape"])
n_voxels = size(coords, 1)

if batch_end == 0
    batch_end = n_voxels
end
batch_end = min(batch_end, n_voxels)

println("=" ^ 60)
println("AxCaliber PINN: Whole-Slice Radius Map")
println("  Slice z=33, voxels $(batch_start):$(batch_end) of $(n_voxels)")
println("  Steps per voxel: $(n_steps)")
println("=" ^ 60)

# Timing
delta_all = 11.0e-3
Deltas = [18e-3, 30e-3, 42e-3, 55e-3]

# Pre-load all signals and bvals/bvecs
all_signals = [Float32.(d["signals_ax$i"]) for i in 1:4]
all_bvals = [Float64.(d["bvals_ax$i"]) for i in 1:4]
all_bvecs = [Float64.(d["bvecs_ax$i"]) for i in 1:4]

# Results arrays
R_map = fill(NaN32, batch_end - batch_start + 1)
f_map = fill(NaN32, batch_end - batch_start + 1)
D_intra_map = fill(NaN32, batch_end - batch_start + 1)
D_extra_map = fill(NaN32, batch_end - batch_start + 1)
loss_map = fill(NaN32, batch_end - batch_start + 1)

t0 = time()
n_done = 0

for idx in batch_start:batch_end
    local_idx = idx - batch_start + 1

    # Build per-voxel AxCaliberData
    signals = [all_signals[i][idx, :] for i in 1:4]

    # Skip voxels with bad signal
    if any(any(isnan, s) || any(isinf, s) || maximum(s) < 0.01 for s in signals)
        continue
    end

    axdata = AxCaliberData(signals, all_bvals, all_bvecs,
                           fill(delta_all, 4), Deltas)

    # Build and train
    rng = MersenneTwister(42)
    signal_dim = sum(length.(signals))
    model = build_axcaliber_pinn(; signal_dim=signal_dim, hidden_dim=64, depth=4)
    ps, st = Lux.setup(rng, model)

    try
        ps, st, geom, losses = train_axcaliber_pinn!(model, ps, st, axdata;
            n_steps=n_steps, learning_rate=1e-3, lambda_physics=1.0,
            print_every=n_steps + 1)  # suppress per-step output

        R_map[local_idx] = Float32(geom.R * 1e6)  # μm
        f_map[local_idx] = Float32(geom.f_intra)
        D_intra_map[local_idx] = Float32(geom.D_intra)
        D_extra_map[local_idx] = Float32(geom.D_extra)
        loss_map[local_idx] = Float32(losses.data[end])
    catch e
        # Skip voxels that fail
    end

    n_done += 1
    if n_done % 100 == 0
        elapsed = time() - t0
        rate = n_done / elapsed
        eta = (batch_end - batch_start + 1 - n_done) / max(rate, 0.01)
        @printf("  %d/%d voxels (%.1f vox/s, ETA %.0fs) R_median=%.2fμm\n",
                n_done, batch_end - batch_start + 1, rate, eta,
                median(filter(!isnan, R_map)))
    end
end

elapsed = time() - t0
@printf("\nDone: %d voxels in %.0fs (%.1f vox/s)\n", n_done, elapsed, n_done/elapsed)

# Summary stats
R_valid = filter(!isnan, R_map)
f_valid = filter(!isnan, f_map)
@printf("R: median=%.2fμm mean=%.2fμm std=%.2fμm (n=%d)\n",
        median(R_valid), mean(R_valid), std(R_valid), length(R_valid))
@printf("f: median=%.2f mean=%.2f\n", median(f_valid), mean(f_valid))

# Save
outfile = "/data/datasets/wand/sub-00395/ses-02/axcaliber/radius_map_z33_$(batch_start)_$(batch_end).json"
open(outfile, "w") do f
    JSON.print(f, Dict(
        "R_map" => R_map, "f_map" => f_map,
        "D_intra_map" => D_intra_map, "D_extra_map" => D_extra_map,
        "loss_map" => loss_map,
        "coords" => coords[batch_start:batch_end, :],
        "mask_shape" => collect(mask_shape),
        "n_steps" => n_steps,
        "R_median_um" => median(R_valid),
        "f_median" => median(f_valid),
        "elapsed_s" => elapsed,
    ), 2)
end
println("Saved: $outfile")
