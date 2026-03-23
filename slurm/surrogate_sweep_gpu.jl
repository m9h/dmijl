#!/usr/bin/env julia
"""
Surrogate architecture sweep — designed to run as Slurm array jobs.

Usage: julia surrogate_sweep.jl <experiment_id>
  experiment_id 1-12 maps to different configs.

Or via Slurm:
  sbatch --array=1-12 slurm/run_sweep.sh
"""

experiment_id = parse(Int, ARGS[1])

using Random, Statistics, LinearAlgebra, Printf, JSON
using CUDA, LuxCUDA

# Move computation to GPU
const USE_GPU = CUDA.functional()
const dev = USE_GPU ? gpu_device() : cpu_device()
println("GPU available: $USE_GPU")
if USE_GPU
    println("GPU: ", CUDA.name(CUDA.device()))
end

# Include sources directly (no package registry needed)
const PROJECT_ROOT = joinpath(@__DIR__, "..")
include(joinpath(PROJECT_ROOT, "src/models/ball_stick.jl"))
include(joinpath(PROJECT_ROOT, "src/pipeline/acquisition.jl"))
include(joinpath(PROJECT_ROOT, "src/pinn/bloch_torrey.jl"))

# Experiment grid
configs = [
    # (hidden_dim, depth, lr, n_steps, loss_type, name)
    (256,  6, 1e-3, 10_000, :relative_mse, "h256_d6_rel"),
    (256,  6, 1e-3, 10_000, :log_cosh,     "h256_d6_logcosh"),
    (256,  6, 1e-3, 10_000, :mse,          "h256_d6_mse"),
    (512,  6, 1e-3, 10_000, :relative_mse, "h512_d6_rel"),
    (512,  8, 1e-3, 10_000, :relative_mse, "h512_d8_rel"),
    (512,  8, 5e-4, 10_000, :relative_mse, "h512_d8_lr5e4"),
    (1024, 6, 1e-3, 10_000, :relative_mse, "h1024_d6_rel"),
    (256,  6, 1e-3, 50_000, :relative_mse, "h256_d6_rel_50k"),
    (512,  6, 1e-3, 50_000, :relative_mse, "h512_d6_rel_50k"),
    (512,  8, 1e-3, 50_000, :relative_mse, "h512_d8_rel_50k"),
    (512,  8, 1e-3, 50_000, :log_cosh,     "h512_d8_logcosh_50k"),
    (1024, 6, 1e-3, 50_000, :relative_mse, "h1024_d6_rel_50k"),
]

if experiment_id < 1 || experiment_id > length(configs)
    error("experiment_id must be 1-$(length(configs))")
end

hidden_dim, depth, lr, n_steps, loss_type, name = configs[experiment_id]
println("=" ^ 60)
println("Experiment $experiment_id: $name")
println("  hidden=$hidden_dim depth=$depth lr=$lr steps=$n_steps loss=$loss_type")
println("=" ^ 60)

rng = MersenneTwister(42)
acq = hcp_like_acquisition()
model_phys = BallStickModel(acq.bvalues, acq.gradient_directions)

lows  = Float32[1e-9, 0.5e-9, 0.1, 0.05, -1, -1, 0, -1, -1, 0]
highs = Float32[3.5e-9, 2.5e-9, 0.8, 0.5, 1, 1, 1, 1, 1, 1]
spans = max.(highs .- lows, 1f-12)

function data_fn(rng, n)
    params_norm = rand(rng, Float32, 10, n)
    params_phys = lows .+ params_norm .* spans
    for j in 1:n
        mu1 = @view params_phys[5:7, j]; mu1 ./= max(norm(mu1), 1f-8)
        mu2 = @view params_phys[8:10, j]; mu2 ./= max(norm(mu2), 1f-8)
    end
    signals = zeros(Float32, length(acq.bvalues), n)
    for j in 1:n
        signals[:, j] = simulate(model_phys, params_phys[:, j])
    end
    return params_norm, signals
end

surrogate = build_surrogate(param_dim=10, signal_dim=length(acq.bvalues),
                            hidden_dim=hidden_dim, depth=depth)
ps, st = Lux.setup(rng, surrogate)

# Move params/state to GPU if available
if USE_GPU
    ps = ps |> dev
    st = st |> dev
end

# GPU-aware data function: generate on CPU, transfer to GPU
function data_fn_gpu(rng, n)
    p, s = data_fn(rng, n)
    if USE_GPU
        return dev(p), dev(s)
    end
    return p, s
end

t0 = time()
ps, st, losses = train_surrogate!(surrogate, ps, st, data_fn_gpu;
    n_steps=n_steps, batch_size=512, learning_rate=lr,
    print_every=max(1, n_steps ÷ 5), loss_type=loss_type)
train_time = time() - t0

# Evaluate on CPU
if USE_GPU
    ps_cpu = ps |> cpu_device()
    st_cpu = st |> cpu_device()
else
    ps_cpu = ps
    st_cpu = st
end
test_params, test_signals = data_fn(MersenneTwister(999), 1000)
pred_signals, _ = surrogate(test_params, ps_cpu, st_cpu)

rel_errors = Float64[]
for j in 1:1000
    pred = pred_signals[:, j]
    exact = test_signals[:, j]
    mask = exact .> 0.01
    if any(mask)
        re = mean(abs.(pred[mask] .- exact[mask]) ./ exact[mask])
        push!(rel_errors, re)
    end
end

result = Dict(
    "name" => name * "_gpu",
    "device" => USE_GPU ? "GB10" : "CPU",
    "experiment_id" => experiment_id,
    "hidden_dim" => hidden_dim,
    "depth" => depth,
    "lr" => lr,
    "n_steps" => n_steps,
    "loss_type" => string(loss_type),
    "final_loss" => losses[end],
    "median_rel_error" => median(rel_errors),
    "mean_rel_error" => mean(rel_errors),
    "pct_under_1pct" => count(rel_errors .< 0.01) / length(rel_errors) * 100,
    "pct_under_5pct" => count(rel_errors .< 0.05) / length(rel_errors) * 100,
    "train_time_s" => train_time,
    "spec1_passed" => median(rel_errors) < 0.01,
)

@printf("\nResults: %s\n", name)
@printf("  Median relative error: %.2f%%\n", result["median_rel_error"] * 100)
@printf("  Samples <1%% error:    %.1f%%\n", result["pct_under_1pct"])
@printf("  Samples <5%% error:    %.1f%%\n", result["pct_under_5pct"])
@printf("  Train time: %.0fs\n", train_time)
println(result["spec1_passed"] ? "  SPEC #1 PASSED ✓" : "  SPEC #1 not yet passed")

# Save result
results_dir = joinpath(PROJECT_ROOT, "results")
mkpath(results_dir)
open(joinpath(results_dir, "surrogate_$(name)_gpu.json"), "w") do f
    JSON.print(f, result, 2)
end
