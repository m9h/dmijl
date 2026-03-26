"""
RestrictedCylinder — cylinder compartment with Soderman approximation
for perpendicular restricted diffusion.

Signal: S = S_par * S_perp

- S_par = exp(-b_par * D_par) where b_par = b * (g·μ)²
- S_perp = |2*J1(2π*q_perp*R) / (2π*q_perp*R)|² (Soderman short-pulse approx)

where q_perp = q * sqrt(1 - (g·μ)²), q = sqrt(b / (Δ - δ/3)) / (2π).
"""

using SpecialFunctions: besselj

Base.@kwdef struct RestrictedCylinder <: AbstractCompartment
    mu::Vector{Float64}       # unit orientation vector (3D Cartesian)
    lambda_par::Float64       # parallel diffusivity (m²/s)
    diameter::Float64         # cylinder diameter in meters (e.g., 4e-6 for 4 μm)
end

parameter_names(::RestrictedCylinder) = (:mu, :lambda_par, :diameter)
parameter_cardinality(::RestrictedCylinder) = Dict(:mu => 3, :lambda_par => 1, :diameter => 1)
parameter_ranges(::RestrictedCylinder) = Dict(
    :mu => (-1.0, 1.0),
    :lambda_par => (0.0, 3.0e-9),
    :diameter => (1e-7, 20e-6)
)

"""
    _soderman_perp(x)

Compute |2*J1(x)/x|² with proper handling of the singularity at x=0.
At x=0, J1(x)/x → 1/2, so 2*J1(x)/x → 1, and the square is 1.
"""
function _soderman_perp(x::Real)
    if abs(x) < 1e-10
        return 1.0
    end
    val = 2.0 * besselj(1, x) / x
    return val * val
end

function signal(cyl::RestrictedCylinder, acq::Acquisition)
    if acq.delta === nothing || acq.Delta === nothing
        throw(ArgumentError(
            "RestrictedCylinder requires delta (pulse duration) and Delta (diffusion time) " *
            "in the Acquisition. Use Acquisition(bvals, bvecs, delta, Delta)."
        ))
    end

    mu = cyl.mu ./ max(norm(cyl.mu), 1e-12)
    R = cyl.diameter / 2.0
    delta = acq.delta
    Delta = acq.Delta

    n = length(acq.bvalues)
    sig = Vector{Float64}(undef, n)

    for i in 1:n
        b = acq.bvalues[i]
        g = @view acq.gradient_directions[i, :]

        # Dot product with cylinder axis
        cos_angle = dot(g, mu)

        # Parallel component: standard stick signal
        b_par = b * cos_angle^2
        S_par = exp(-b_par * cyl.lambda_par)

        # Perpendicular component: Soderman approximation
        if b < 1e-10
            # b ≈ 0 → no diffusion weighting → S_perp = 1
            S_perp = 1.0
        else
            # q = |g_eff| * delta / (2π) but expressed via b-value:
            # b = (2π*q)² * (Delta - delta/3), so q = sqrt(b/(Delta-delta/3)) / (2π)
            q = sqrt(b / (Delta - delta / 3)) / (2π)
            sin2_angle = 1.0 - cos_angle^2
            q_perp = q * sqrt(max(sin2_angle, 0.0))
            x = 2π * q_perp * R
            S_perp = _soderman_perp(x)
        end

        sig[i] = S_par * S_perp
    end

    return sig
end

_reconstruct(::RestrictedCylinder, p::AbstractVector) =
    RestrictedCylinder(mu=p[1:3], lambda_par=p[4], diameter=p[5])
