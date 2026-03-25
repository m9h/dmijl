#!/usr/bin/env julia
"""
D(r) recovery on WAND CHARMED data — benchmarks CPU vs GPU.
Usage: julia run_dfield.jl [cpu|gpu]
"""

device_arg = length(ARGS) >= 1 ? ARGS[1] : "cpu"

using Random, Statistics, LinearAlgebra, Printf, NPZ, JSON
using Lux, Optimisers, Zygote

if device_arg == "gpu"
    using CUDA, LuxCUDA
end

# Include dmijl source
const DMIJL = "/home/mhough/dev/dmijl"
include(joinpath(DMIJL, "src/pinn/diffusion_field_v2.jl"))

# Load real WAND CHARMED voxel
data = NPZ.npzread("/data/datasets/wand/sub-00395/ses-02/dwi/wm_voxel.npz")
signal = Float32.(data["signal"])
bvals = Float64.(data["bvals"])
bvecs = Float64.(data["bvecs"])

println("=" ^ 60)
println("D(r) RECOVERY: WAND Connectom CHARMED — $(uppercase(device_arg))")
println("=" ^ 60)
println("Subject: sub-00395, voxel (54,54,33)")
println("Shells: b = ", join(sort(unique(round.(Int, bvals))), "/"), " s/mm²")
println("Volumes: $(length(signal))")
println()

prob = DiffusionFieldProblem(
    signal,
    bvals .* 1e6,
    bvecs,
    11.0e-3, 15.129e-3, 80e-3, 2e-3,
)

# Benchmark multiple configs
configs = [
    (steps=1000,  D_h=32,  D_d=3, name="tiny"),
    (steps=5000,  D_h=64,  D_d=4, name="small"),
    (steps=10000, D_h=128, D_d=4, name="medium"),
]

results = Dict[]

for cfg in configs
    println("\n--- Config: $(cfg.name) ($(cfg.steps) steps) ---")

    t_start = time()
    result = solve_diffusion_field_v2(prob;
        output_type = :diagonal,
        D_hidden = cfg.D_h, D_depth = cfg.D_d,
        n_steps = cfg.steps,
        n_spatial = 32,
        n_meas_per_step = 20,
        learning_rate = 1e-3,
        print_every = cfg.steps ÷ 5,
    )
    elapsed = time() - t_start

    # Extract D at center
    D_center = eval_D(result.D_net, result.ps_D, result.st_D,
                      Float32[0.0, 0.0, 0.0], :diagonal)
    D_sorted = sort(D_center, rev=true)
    md = mean(D_sorted)
    fa = md > 0 ? sqrt(3/2) * sqrt(sum((D_sorted .- md).^2)) / sqrt(sum(D_sorted.^2)) : 0.0

    r = Dict(
        "config" => cfg.name,
        "device" => device_arg,
        "steps" => cfg.steps,
        "elapsed_s" => elapsed,
        "steps_per_s" => cfg.steps / elapsed,
        "D_eigenvalues" => D_sorted,
        "MD" => md,
        "FA" => fa,
        "loss_start" => result.losses[1],
        "loss_end" => result.losses[end],
    )
    push!(results, r)

    @printf("  Time: %.1fs (%.0f steps/s)\n", elapsed, cfg.steps / elapsed)
    @printf("  D: [%.2e, %.2e, %.2e]\n", D_sorted...)
    @printf("  MD: %.2e  FA: %.3f\n", md, fa)
    @printf("  Loss: %.4f → %.4f\n", result.losses[1], result.losses[end])
end

# Save results
outfile = "/data/datasets/wand/slurm/results_dfield_$(device_arg).json"
open(outfile, "w") do f
    JSON.print(f, results, 2)
end

# Summary
println("\n" * "=" ^ 60)
println("BENCHMARK SUMMARY — $(uppercase(device_arg))")
println("=" ^ 60)
@printf("  %-10s  %8s  %10s  %8s  %8s\n", "Config", "Time", "Steps/s", "MD", "FA")
for r in results
    @printf("  %-10s  %7.1fs  %9.0f  %7.2e  %7.3f\n",
            r["config"], r["elapsed_s"], r["steps_per_s"],
            r["MD"], r["FA"])
end
println("\nResults saved to: $outfile")
