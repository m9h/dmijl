"""
Pipeline configuration — Julia equivalent of SBIPipelineConfig.
Uses Julia's type system instead of Python dataclasses.
"""

Base.@kwdef struct SBIConfig
    # Model
    model_name::String = "DTI"
    parameter_names::Vector{String} = ["FA", "MD"]
    parameter_ranges::Dict{String, Tuple{Float64, Float64}} = Dict()

    # Acquisition
    bvalues::Vector{Float64} = Float64[]
    gradient_directions::Matrix{Float64} = zeros(0, 3)

    # Inference
    inference_mode::Symbol = :score  # :mdn, :flow, :score
    prediction::Symbol = :eps        # :eps or :v

    # Network
    hidden_dim::Int = 512
    depth::Int = 6
    cond_dim::Int = 128

    # Noise
    noise_type::Symbol = :rician
    snr::Float64 = 30.0
    snr_range::Union{Nothing, Tuple{Float64, Float64}} = (10.0, 50.0)

    # Training
    learning_rate::Float64 = 3e-4
    batch_size::Int = 512
    n_steps::Int = 50_000

    # Diffusion schedule
    beta_min::Float64 = 0.01
    beta_max::Float64 = 5.0

    # Sampling
    sampler::Symbol = :ddpm
    n_sde_steps::Int = 500
    n_posterior_samples::Int = 500

    # Reproducibility
    seed::Int = 0
end

"""Number of inferred parameters."""
theta_dim(c::SBIConfig) = length(c.parameter_names)

"""Signal dimension (number of measurements)."""
signal_dim(c::SBIConfig) = length(c.bvalues)

"""Noise sigma from SNR."""
noise_sigma(c::SBIConfig) = 1.0 / c.snr
