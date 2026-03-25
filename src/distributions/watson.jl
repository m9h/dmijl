"""
Watson distribution and DistributedModel for orientation dispersion.

The Watson distribution is an axially symmetric distribution on the sphere:
    f(n; μ, κ) ∝ exp(κ (n·μ)²)

κ > 0: concentrated around μ (higher = narrower)
κ = 0: uniform on the sphere
κ < 0: girdle distribution (concentrated perpendicular to μ)
"""

using LinearAlgebra, Statistics

# ---- Fibonacci sphere grid ----

"""Generate approximately uniform points on the unit sphere using Fibonacci spiral."""
function fibonacci_sphere(n::Int)
    golden_ratio = (1 + sqrt(5)) / 2
    vectors = Vector{Vector{Float64}}(undef, n)

    for i in 0:n-1
        theta = acos(1 - 2 * (i + 0.5) / n)
        phi = 2pi * i / golden_ratio
        vectors[i+1] = [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
    end

    return vectors
end

# ---- Watson Distribution ----

struct WatsonDistribution
    grid_vectors::Vector{Vector{Float64}}
    grid_weights::Vector{Float64}  # equal-area weights (1/n each)
end

function WatsonDistribution(; n_grid::Int=300)
    vecs = fibonacci_sphere(n_grid)
    weights = fill(1.0 / n_grid, n_grid)
    return WatsonDistribution(vecs, weights)
end

"""
    watson_weights(w::WatsonDistribution, mu, kappa)

Compute normalized Watson distribution weights on the sphere grid.
Returns a vector of weights that sum to 1.
"""
function watson_weights(w::WatsonDistribution, mu::AbstractVector, kappa::Real)
    mu_unit = mu ./ max(norm(mu), 1e-12)
    n = length(w.grid_vectors)

    # Watson PDF: f(n) ∝ exp(κ (n·μ)²)
    log_pdf = zeros(n)
    for i in 1:n
        cos_angle = dot(w.grid_vectors[i], mu_unit)
        log_pdf[i] = kappa * cos_angle^2
    end

    # Numerically stable normalization: subtract max, exponentiate, normalize
    log_pdf .-= maximum(log_pdf)
    pdf = exp.(log_pdf) .* w.grid_weights
    pdf ./= sum(pdf)

    return pdf
end

# ---- DistributedModel ----

"""
    DistributedModel(base, distribution, target_param, mu, kappa)

Wraps a compartment model with an orientation distribution.

The `target_param` (e.g. `:mu`) is replaced by integration over the distribution
at each grid point. The distribution is parameterized by `mu` (mean direction)
and `kappa` (concentration).

This is itself an `AbstractCompartment` and can be composed into MCMs.
"""
struct DistributedModel <: AbstractCompartment
    base::AbstractCompartment
    distribution::WatsonDistribution
    target_param::Symbol
    mu::Vector{Float64}       # mean orientation
    kappa::Float64            # concentration parameter
end

function parameter_names(dm::DistributedModel)
    # The target parameter (orientation) is absorbed by the distribution;
    # remaining base parameters + distribution parameters
    base_names = parameter_names(dm.base)
    remaining = filter(n -> n != dm.target_param, base_names)
    return (remaining..., :mu, :kappa)
end

function parameter_cardinality(dm::DistributedModel)
    base_card = parameter_cardinality(dm.base)
    result = Dict{Symbol, Int}()
    for (k, v) in base_card
        k == dm.target_param && continue
        result[k] = v
    end
    result[:mu] = 3
    result[:kappa] = 1
    return result
end

function parameter_ranges(dm::DistributedModel)
    base_ranges = parameter_ranges(dm.base)
    result = Dict{Symbol, Tuple{Float64, Float64}}()
    for (k, v) in base_ranges
        k == dm.target_param && continue
        result[k] = v
    end
    result[:mu] = (-1.0, 1.0)
    result[:kappa] = (0.0, 128.0)
    return result
end

function signal(dm::DistributedModel, acq::Acquisition)
    weights = watson_weights(dm.distribution, dm.mu, dm.kappa)
    n_meas = length(acq.bvalues)
    result = zeros(n_meas)

    for (i, w) in enumerate(weights)
        w < 1e-15 && continue  # skip negligible weights

        # Create a copy of the base compartment with orientation replaced
        comp_i = _set_orientation(dm.base, dm.target_param, dm.distribution.grid_vectors[i])
        result .+= w .* signal(comp_i, acq)
    end

    return result
end

# ---- Helpers to set orientation on compartments ----

function _set_orientation(stick::C1Stick, param::Symbol, vec::Vector{Float64})
    param == :mu || error("C1Stick only supports :mu as orientation parameter")
    return C1Stick(mu=vec, lambda_par=stick.lambda_par)
end

function _set_orientation(zep::G2Zeppelin, param::Symbol, vec::Vector{Float64})
    param == :mu || error("G2Zeppelin only supports :mu as orientation parameter")
    return G2Zeppelin(mu=vec, lambda_par=zep.lambda_par, lambda_perp=zep.lambda_perp)
end

# ---- _reconstruct for DistributedModel ----

function _reconstruct(dm::DistributedModel, p::AbstractVector)
    # Reconstruct: base params (minus target), then mu(3), kappa(1)
    base_names = parameter_names(dm.base)
    base_card = parameter_cardinality(dm.base)

    idx = 1
    base_params = Float64[]
    for name in base_names
        c = base_card[name]
        if name == dm.target_param
            # Skip — absorbed by distribution
            continue
        end
        append!(base_params, p[idx:idx+c-1])
        idx += c
    end

    mu_new = p[idx:idx+2]
    kappa_new = p[idx+3]

    # Reconstruct the base compartment with dummy orientation
    base_new = _reconstruct_without_orientation(dm.base, dm.target_param, base_params)

    return DistributedModel(base_new, dm.distribution, dm.target_param, mu_new, kappa_new)
end

function _reconstruct_without_orientation(stick::C1Stick, target::Symbol, params::AbstractVector)
    target == :mu || error("Expected :mu")
    return C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=params[1])  # dummy orientation
end

function _reconstruct_without_orientation(zep::G2Zeppelin, target::Symbol, params::AbstractVector)
    target == :mu || error("Expected :mu")
    return G2Zeppelin(mu=[0.0, 0.0, 1.0], lambda_par=params[1], lambda_perp=params[2])
end
