#!/usr/bin/env julia
"""
Ball+2Stick score-based posterior estimation.

Direct port of the JAX implementation — but with no XLA compilation wall.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Lux, Random, Statistics, LinearAlgebra, Printf

include("../src/models/ball_stick.jl")
include("../src/noise.jl")
include("../src/diffusion/schedule.jl")
include("../src/diffusion/score_net.jl")
include("../src/diffusion/train.jl")
include("../src/diffusion/sample.jl")

# --- Setup ---
rng = Random.default_rng()
Random.seed!(rng, 42)

# HCP-like acquisition
n_b0, n_b1, n_b2, n_b3 = 6, 30, 30, 24
bvals = vcat(zeros(n_b0), fill(1e9, n_b1), fill(2e9, n_b2), fill(3e9, n_b3))

function rand_vecs(rng, n)
    z = randn(rng, n, 3)
    return z ./ sqrt.(sum(z.^2, dims=2))
end

bvecs = vcat(
    repeat([1.0 0.0 0.0], n_b0, 1),
    rand_vecs(rng, n_b1),
    rand_vecs(rng, n_b2),
    rand_vecs(rng, n_b3),
)

model_phys = BallStickModel(bvals, bvecs)
n_meas = length(bvals)

# Parameter ranges for normalisation
lows  = Float32[1e-9, 0.5e-9, 0.1, 0.05, -1, -1, 0, -1, -1, 0]
highs = Float32[3.5e-9, 2.5e-9, 0.8, 0.5, 1, 1, 1, 1, 1, 1]
spans = max.(highs .- lows, 1f-12)

b0_mask = bvals .< 100e6

function sample_prior(rng, n)
    # Sample normalised parameters (10, n)
    u = rand(rng, Float32, 10, n)
    theta_norm = u  # already in [0, 1]

    # Enforce f1 >= f2 (label-switching fix)
    theta = lows .+ theta_norm .* spans
    for j in 1:n
        if theta[3, j] < theta[4, j]
            theta[3, j], theta[4, j] = theta[4, j], theta[3, j]
            theta[5:7, j], theta[8:10, j] = theta[8:10, j], theta[5:7, j]
        end
    end
    return (theta .- lows) ./ spans
end

function sim_fn(rng, theta_norm)
    n = size(theta_norm, 2)
    theta = lows .+ theta_norm .* spans
    signals = zeros(Float32, n_meas, n)
    for j in 1:n
        signals[:, j] = simulate(model_phys, @view theta[:, j])
    end
    # Add Rician noise with variable SNR
    noisy = add_rician_noise(rng, signals'; snr_range=(10.0, 50.0))'
    # b0-normalise
    b0_mean = mean(noisy[b0_mask, :], dims=1)
    b0_mean = max.(b0_mean, 1f-6)
    return noisy ./ b0_mean
end

# --- Build network ---
println("Building score network...")
model_spec = build_score_net(
    param_dim=10, signal_dim=n_meas,
    hidden_dim=512, depth=6, cond_dim=128,
)

# Initialize with Lux
lux_model = Lux.Chain(
    signal_encoder = model_spec.signal_encoder,
    time_encoder = model_spec.time_encoder,
    input_proj = model_spec.input_proj,
    output_proj = model_spec.output_proj,
)

# For the prototype, use the Lux-initialized params
ps_lux, st_lux = Lux.setup(rng, lux_model)

println("Note: Full training loop requires Lux-compatible FiLM blocks.")
println("This example validates the forward model + noise pipeline.")

# Test forward model
println("\n--- Testing forward model ---")
theta_test = sample_prior(rng, 5)
signals_test = sim_fn(rng, theta_test)
println("Theta shape: ", size(theta_test))
println("Signal shape: ", size(signals_test))
println("Signal range: ", extrema(signals_test))

println("\nMicrostructure.jl scaffold ready.")
println("Next: wire up FiLM blocks as custom Lux layers.")
