#!/usr/bin/env julia
"""
Tight sweep: fix MD overestimation (2.7e-9 vs expected 0.7e-9).

Hypotheses:
1. D-network softplus scaling too large (1e-9 allows up to ~5e-9)
2. No regularization on D magnitude
3. Need log-space signal fitting (better for high-b attenuation)
4. b-values not in correct units
5. Need more spatial samples
6. Need all measurements per step (not subsampled)
"""

experiment_id = parse(Int, ARGS[1])

using Random, Statistics, LinearAlgebra, Printf, NPZ, JSON
using Lux, Optimisers, Zygote

const DMIJL = "/home/mhough/dev/dmijl"
include(joinpath(DMIJL, "src/pinn/diffusion_field.jl"))

# Modified v2 with variant-specific changes
data = NPZ.npzread("/data/datasets/wand/sub-00395/ses-02/dwi/wm_voxel.npz")
signal = Float32.(data["signal"])
bvals_smm2 = Float64.(data["bvals"])  # these are in s/mm²
bvecs = Float64.(data["bvecs"])       # (3, 266)

# Normalize bvecs
n_meas = length(signal)
gdir = size(bvecs, 1) == 3 ? bvecs' : bvecs  # → (n_meas, 3)
for i in 1:n_meas
    n = norm(@view gdir[i, :])
    if n > 1e-8; gdir[i, :] ./= n; end
end
gdir_f32 = Float32.(gdir)

# Experiment configs
configs = [
    # (name, bval_scale, D_scale, n_spatial, n_meas_per_step, lr, steps, log_loss, D_reg)
    ("baseline",       1e6,  1e-9,  32, 20, 1e-3, 3000, false, 0.0),
    ("bval_already_m", 1.0,  1e-9,  32, 20, 1e-3, 3000, false, 0.0),  # maybe bvals already in s/m²?
    ("D_scale_1e-10",  1e6,  1e-10, 32, 20, 1e-3, 3000, false, 0.0),  # tighter D range
    ("log_signal",     1e6,  1e-9,  32, 20, 1e-3, 3000, true,  0.0),  # log-space loss
    ("D_regularized",  1e6,  1e-9,  32, 20, 1e-3, 3000, false, 1.0),  # penalize large D
    ("more_spatial",   1e6,  1e-9,  128, 20, 1e-3, 3000, false, 0.0), # more MC samples
    ("all_meas",       1e6,  1e-9,  32, 266, 1e-3, 3000, false, 0.0), # all measurements
    ("combo_best",     1e6,  1e-10, 64, 40, 1e-3, 5000, true,  0.5),  # combine best ideas
]

name, bval_scale, D_scale, n_sp, n_mp, lr, steps, use_log, D_reg = configs[experiment_id]

println("=" ^ 60)
println("Experiment $experiment_id: $name")
println("  bval_scale=$bval_scale D_scale=$D_scale n_spatial=$n_sp")
println("  n_meas/step=$n_mp log_loss=$use_log D_reg=$D_reg steps=$steps")
println("=" ^ 60)

bvals_f32 = Float32.(bvals_smm2 .* bval_scale)

# Build D-network with configurable scale
function build_D_net(; hidden_dim=64, depth=4, scale=1e-9)
    layers = Any[Dense(3 => hidden_dim, gelu)]
    for _ in 2:depth
        push!(layers, Dense(hidden_dim => hidden_dim, gelu))
    end
    push!(layers, Dense(hidden_dim => 3))
    return Chain(layers...), scale
end

function eval_D_scaled(net, ps, st, x, scale)
    raw, _ = net(reshape(x, :, 1), ps, st)
    return Lux.softplus.(raw[:, 1]) .* Float32(scale)
end

function predict_signal_v3(net, ps, st, scale, b, g, x_samples)
    n = size(x_samples, 2)
    total = 0.0f0
    for i in 1:n
        D = eval_D_scaled(net, ps, st, @view(x_samples[:, i]), scale)
        adc = g[1]^2 * D[1] + g[2]^2 * D[2] + g[3]^2 * D[3]
        total += exp(-b * adc)
    end
    return total / n
end

function run_experiment(D_scale, lr, steps, n_sp, n_mp, use_log, D_reg, name,
                        signal, bvals_f32, gdir_f32, n_meas)
    rng = MersenneTwister(42)
    D_net, D_sc = build_D_net(; hidden_dim=64, depth=4, scale=D_scale)
    ps, st = Lux.setup(rng, D_net)
    opt_state = Optimisers.setup(Adam(Float64(lr)), ps)

    vr = Float32(2e-3 / 2)
    losses = Float64[]
    t0 = time()

    for step in 1:steps
    x_samples = (2.0f0 .* rand(rng, Float32, 3, n_sp) .- 1.0f0) .* vr
    meas_idx = [((step-1) * n_mp + m - 1) % n_meas + 1 for m in 1:n_mp]

    (loss, st), grads = Zygote.withgradient(ps) do p
        l = 0.0f0
        for idx in meas_idx
            b = bvals_f32[idx]
            g = gdir_f32[idx, :]
            S_obs = signal[idx]

            if b < 100f0
                S_pred = 1.0f0
            else
                S_pred = predict_signal_v3(D_net, p, st, D_sc, b, g, x_samples)
            end

            if use_log && S_obs > 0.01f0 && S_pred > 0.01f0
                l += (log(S_pred) - log(S_obs))^2
            else
                l += (S_pred - S_obs)^2
            end
        end
        l /= n_mp

        # D regularization: penalize D > 3e-9 (CSF)
        if D_reg > 0
            D_c = eval_D_scaled(D_net, p, st, Float32[0,0,0], D_sc)
            l += Float32(D_reg) * mean(max.(D_c .- 3f-9, 0f0).^2) / 1f-18
        end

        return l, st
    end

    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
    push!(losses, loss)

        if step % max(steps ÷ 5, 1) == 0 || step == 1
            D_c = eval_D_scaled(D_net, ps, st, Float32[0,0,0], D_sc)
            ds = sort(D_c, rev=true)
            md = mean(ds)
            fa = md > 0 ? sqrt(3/2) * sqrt(sum((ds .- md).^2)) / sqrt(sum(ds.^2)) : 0.0
            el = time() - t0
            @printf("[%s] step %d/%d  loss=%.4f  MD=%.2e  FA=%.3f  (%.0f steps/s)\n",
                    name, step, steps, loss, md, fa, step/el)
        end
    end  # training loop

    return D_net, ps, st, D_sc, losses, time() - t0
end  # function

D_net, ps, st, D_sc, losses, elapsed = run_experiment(
    D_scale, lr, steps, n_sp, n_mp, use_log, D_reg, name,
    signal, bvals_f32, gdir_f32, n_meas)

# Final results
D_c = eval_D_scaled(D_net, ps, st, Float32[0,0,0], D_sc)
ds = sort(D_c, rev=true)
md = mean(ds)
fa = md > 0 ? sqrt(3/2) * sqrt(sum((ds .- md).^2)) / sqrt(sum(ds.^2)) : 0.0
elapsed = time() - t0

result = Dict(
    "name" => name, "experiment_id" => experiment_id,
    "MD" => md, "FA" => fa,
    "D_eigenvalues" => ds,
    "AD" => ds[1], "RD" => mean(ds[2:3]),
    "loss_end" => losses[end],
    "elapsed_s" => elapsed,
    "steps_per_s" => steps / elapsed,
)

println()
@printf("RESULT: %s\n", name)
@printf("  D: [%.2e, %.2e, %.2e]\n", ds...)
@printf("  MD=%.2e  FA=%.3f  AD=%.2e  RD=%.2e\n", md, fa, ds[1], mean(ds[2:3]))
@printf("  Loss: %.4f  Time: %.0fs\n", losses[end], elapsed)
println("  Expected WM: MD~0.7e-9, FA~0.4-0.7")

outdir = "/data/datasets/wand/slurm/results"
mkpath(outdir)
open(joinpath(outdir, "sweep_$(name).json"), "w") do f
    JSON.print(f, result, 2)
end
