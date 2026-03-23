"""
ModelSimulator: unified wrapper for forward model + prior + noise.
Julia port of dmipy_jax/pipeline/simulator.py.

Uses multiple dispatch instead of class methods.
"""

struct ModelSimulator{F, P}
    forward_fn::F          # (params, acq) -> signal
    parameter_names::Vector{String}
    parameter_ranges::Dict{String, Tuple{Float64, Float64}}
    bvalues::Vector{Float64}
    gradient_directions::Matrix{Float64}
    noise_type::Symbol     # :rician or :gaussian
    sigma::Float64
    snr_range::Union{Nothing, Tuple{Float64, Float64}}
    prior_fn::P            # (rng, n) -> (param_dim, n) or nothing
end

function ModelSimulator(forward_fn, names, ranges, bvals, bvecs;
                        noise_type=:rician, snr=30.0,
                        snr_range=nothing, prior_fn=nothing)
    ModelSimulator(forward_fn, names, ranges, bvals, bvecs,
                   noise_type, 1.0 / snr, snr_range, prior_fn)
end

theta_dim(s::ModelSimulator) = length(s.parameter_names)
signal_dim(s::ModelSimulator) = length(s.bvalues)

# --- Prior sampling ---

function sample_prior(s::ModelSimulator, rng, n::Int)
    if s.prior_fn !== nothing
        return s.prior_fn(rng, n)
    end
    return default_prior(s, rng, n)
end

function default_prior(s::ModelSimulator, rng, n::Int)
    d = theta_dim(s)
    lows = Float32[s.parameter_ranges[name][1] for name in s.parameter_names]
    highs = Float32[s.parameter_ranges[name][2] for name in s.parameter_names]
    u = rand(rng, Float32, d, n)
    return lows .+ u .* (highs .- lows)
end

# --- Simulation ---

function simulate(s::ModelSimulator, theta::AbstractMatrix)
    n = size(theta, 2)
    sig = zeros(Float32, signal_dim(s), n)
    for j in 1:n
        sig[:, j] = s.forward_fn(@view(theta[:, j]), s)
    end
    return sig
end

# --- Noise ---

function get_sigma(s::ModelSimulator, rng, n::Int)
    if s.snr_range !== nothing
        lo, hi = s.snr_range
        snr = rand(rng, Float32, 1, n) .* Float32(hi - lo) .+ Float32(lo)
        return 1.0f0 ./ snr
    end
    return fill(Float32(s.sigma), 1, n)
end

function add_noise(s::ModelSimulator, rng, signal::AbstractMatrix)
    n = size(signal, 2)
    sigma = get_sigma(s, rng, n)
    if s.noise_type == :rician
        n1 = randn(rng, Float32, size(signal)) .* sigma
        n2 = randn(rng, Float32, size(signal)) .* sigma
        return @. sqrt((signal + n1)^2 + n2^2)
    else
        noise = randn(rng, Float32, size(signal)) .* sigma
        return signal .+ noise
    end
end

# --- Sample + simulate + noise + b0-normalise ---

function sample_and_simulate(s::ModelSimulator, rng, n::Int)
    theta = sample_prior(s, rng, n)
    signal = simulate(s, theta)
    noisy = add_noise(s, rng, signal)

    # b0 normalise
    b0_mask = s.bvalues .< 100e6
    if any(b0_mask)
        b0_mean = mean(noisy[b0_mask, :], dims=1)
        b0_mean = max.(b0_mean, 1f-6)
        noisy = noisy ./ b0_mean
    end
    return theta, noisy
end
