#!/usr/bin/env julia
"""
AxCaliber PINN v2: compile once, fit many voxels.

Key fix: build the model and compile the training step ONCE,
then reinitialize parameters per voxel. No repeated JIT.
"""

using Random, Statistics, LinearAlgebra, Printf, NPZ, JSON, Lux, Optimisers, Zygote

const DMIJL = "/home/mhough/dev/dmijl"
include(joinpath(DMIJL, "src/pinn/axcaliber_pinn.jl"))

n_steps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000
batch_start = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
batch_end = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0

# Load data
d = NPZ.npzread("/data/datasets/wand/sub-00395/ses-02/axcaliber/slice_z33.npz")
coords = d["coords"]
n_voxels = size(coords, 1)
if batch_end == 0; batch_end = n_voxels; end
batch_end = min(batch_end, n_voxels)

all_signals = [Float32.(d["signals_ax$i"]) for i in 1:4]
all_bvals = [Float64.(d["bvals_ax$i"]) for i in 1:4]
all_bvecs = [Float64.(d["bvecs_ax$i"]) for i in 1:4]
delta_all = 11.0e-3
Deltas = [18e-3, 30e-3, 42e-3, 55e-3]

println("=" ^ 60)
println("AxCaliber PINN v2: compile-once, fit-many")
println("  Voxels $(batch_start):$(batch_end) of $(n_voxels)")
println("  Steps: $(n_steps)")
println("=" ^ 60)

# BUILD MODEL ONCE — all voxels share the same architecture
rng = MersenneTwister(42)
signal_dim = sum(length(all_signals[i][1, :]) for i in 1:4)
model = build_axcaliber_pinn(; signal_dim=signal_dim, hidden_dim=64, depth=4)
ps_template, st_template = Lux.setup(rng, model)

# COMPILE THE TRAINING STEP ONCE with a dummy voxel
println("Compiling training step...")
dummy_signals = [all_signals[i][1, :] for i in 1:4]
dummy_data = AxCaliberData(dummy_signals, all_bvals, all_bvecs,
                           fill(delta_all, 4), Deltas)
signal_all_dummy = Float32.(vcat(dummy_signals...))

# Pre-compile the gradient computation
opt = Optimisers.setup(Adam(1e-3), ps_template)

function train_one_voxel(model, ps, st, signal_all, axdata, n_steps, opt_state)
    for step in 1:n_steps
        (loss, (st_new,)), grads = Zygote.withgradient(ps) do p
            raw, new_st = model(reshape(signal_all, :, 1), p, st)
            geom = decode_geometry(raw[:, 1])

            l = 0.0f0
            n_total = 0
            for a in 1:4
                bvals = axdata.bvalues[a]
                bvecs_a = axdata.bvecs[a]
                delta_a = axdata.deltas[a]
                Delta_a = axdata.Deltas[a]
                obs = axdata.signals[a]

                for j in eachindex(bvals)
                    b = Float64(bvals[j]) * 1e6
                    b < 100e6 && continue
                    g = size(bvecs_a, 1) == 3 ? Float64.(bvecs_a[:, j]) : Float64.(bvecs_a[j, :])
                    gn = norm(g)
                    gn > 1e-8 && (g = g ./ gn)

                    S_phys = Float32(axcaliber_signal(
                        b, delta_a, Delta_a,
                        Float64(geom.D_intra), Float64(geom.D_extra),
                        Float64(geom.R), Float64(geom.f_intra),
                        g, Float64.(geom.mu)))
                    l += (S_phys - obs[j])^2
                    n_total += 1
                end
            end
            return l / max(n_total, 1), (new_st,)
        end
        st = st_new
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
    end

    raw, _ = model(reshape(signal_all, :, 1), ps, st)
    return decode_geometry(raw[:, 1])
end

# Warmup compile
@info "Warmup compilation..."
t_compile = @elapsed train_one_voxel(model, ps_template, st_template,
    signal_all_dummy, dummy_data, 5, Optimisers.setup(Adam(1e-3), ps_template))
@info "Compiled in $(round(t_compile, digits=1))s"

# NOW FIT ALL VOXELS — no recompilation
R_map = fill(NaN32, batch_end - batch_start + 1)
f_map = fill(NaN32, batch_end - batch_start + 1)
D_intra_map = fill(NaN32, batch_end - batch_start + 1)
D_extra_map = fill(NaN32, batch_end - batch_start + 1)

t0 = time()
n_done = 0

for idx in batch_start:batch_end
    local_idx = idx - batch_start + 1
    signals = [all_signals[i][idx, :] for i in 1:4]

    if any(any(isnan, s) || any(isinf, s) || maximum(s) < 0.01 for s in signals)
        continue
    end

    signal_all = Float32.(vcat(signals...))
    axdata = AxCaliberData(signals, all_bvals, all_bvecs,
                           fill(delta_all, 4), Deltas)

    # Fresh parameters, same compiled code
    ps_fresh = deepcopy(ps_template)
    opt_fresh = Optimisers.setup(Adam(1e-3), ps_fresh)

    try
        geom = train_one_voxel(model, ps_fresh, st_template, signal_all,
                               axdata, n_steps, opt_fresh)
        R_map[local_idx] = Float32(geom.R * 1e6)
        f_map[local_idx] = Float32(geom.f_intra)
        D_intra_map[local_idx] = Float32(geom.D_intra)
        D_extra_map[local_idx] = Float32(geom.D_extra)
    catch
    end

    n_done += 1
    if n_done % 50 == 0
        elapsed = time() - t0
        rate = n_done / elapsed
        R_valid = filter(!isnan, R_map[1:local_idx])
        @printf("  %d/%d (%.1f vox/s, ETA %.0fm) R_median=%.2fμm\n",
                n_done, batch_end - batch_start + 1, rate,
                (batch_end - batch_start + 1 - n_done) / max(rate, 0.01) / 60,
                isempty(R_valid) ? 0.0 : median(R_valid))
    end
end

elapsed = time() - t0
R_valid = filter(!isnan, R_map)
f_valid = filter(!isnan, f_map)
@printf("\nDone: %d voxels in %.0fs (%.1f vox/s)\n", n_done, elapsed, n_done/elapsed)
@printf("R: median=%.2fμm mean=%.2fμm (n=%d)\n",
        median(R_valid), mean(R_valid), length(R_valid))
@printf("f: median=%.2f mean=%.2f\n", median(f_valid), mean(f_valid))

outfile = "/data/datasets/wand/sub-00395/ses-02/axcaliber/radius_map_v2_$(batch_start)_$(batch_end).json"
open(outfile, "w") do f
    JSON.print(f, Dict(
        "R_map" => R_map, "f_map" => f_map,
        "D_intra_map" => D_intra_map, "D_extra_map" => D_extra_map,
        "coords" => coords[batch_start:batch_end, :],
        "mask_shape" => collect(Tuple(d["mask_shape"])),
        "n_steps" => n_steps, "elapsed_s" => elapsed,
        "R_median_um" => median(R_valid), "f_median" => median(f_valid),
    ), 2)
end
println("Saved: $outfile")
