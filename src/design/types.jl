"""
    OED type definitions for dMRI protocol optimization.
"""

"""
    DesignSpace(; kwargs...)

Specification of the design space for dMRI protocol optimization.
"""
@kwdef struct DesignSpace
    b_range::Tuple{Float64, Float64} = (0.0, 1e10)      # s/m²
    n_measurements::Int = 90
    n_b0::Int = 6
    delta_range::Tuple{Float64, Float64} = (5e-3, 40e-3) # seconds
    Delta_range::Tuple{Float64, Float64} = (10e-3, 80e-3)
    G_max::Float64 = 0.08                                 # T/m (80 mT/m)
    optimize_timing::Bool = false
end

"""
    DesignResult

Output of protocol optimization.
"""
struct DesignResult
    acquisition::Acquisition
    fim::Matrix{Float64}
    crlb::Vector{Float64}
    criterion_value::Float64
    parameter_names::Vector{String}
end

"""
    OEDProblem(; kwargs...)

Full specification of an OED problem.
"""
@kwdef struct OEDProblem{M}
    model::M
    design_space::DesignSpace = DesignSpace()
    criterion::Symbol = :D
    weights::Union{Nothing, Vector{Float64}} = nothing
    noise_model::Symbol = :gaussian
    sigma::Float64 = 0.02
    prior_samples::Union{Nothing, Matrix{Float64}} = nothing
end
