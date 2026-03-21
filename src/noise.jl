"""Noise models for dMRI simulation."""

using Random, LinearAlgebra

function add_rician_noise(rng::AbstractRNG, signal::AbstractMatrix, sigma::Real)
    n1 = randn(rng, size(signal)) .* sigma
    n2 = randn(rng, size(signal)) .* sigma
    return @. sqrt((signal + n1)^2 + n2^2)
end

function add_rician_noise(rng::AbstractRNG, signal::AbstractMatrix;
                          snr_range::Tuple{Float64,Float64}=(10.0, 50.0))
    n = size(signal, 1)
    snr = rand(rng, n) .* (snr_range[2] - snr_range[1]) .+ snr_range[1]
    sigma = 1.0 ./ snr  # (n,)
    n1 = randn(rng, size(signal)) .* sigma
    n2 = randn(rng, size(signal)) .* sigma
    return @. sqrt((signal + n1)^2 + n2^2)
end
