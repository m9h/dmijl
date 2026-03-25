"""G2Zeppelin — axially-symmetric diffusion tensor. Signal: exp(-b * (λ_par cos²θ + λ_perp sin²θ))."""

Base.@kwdef struct G2Zeppelin <: AbstractCompartment
    mu::Vector{Float64}       # unit orientation vector (3D Cartesian)
    lambda_par::Float64       # parallel diffusivity (m²/s)
    lambda_perp::Float64      # perpendicular diffusivity (m²/s)
end

parameter_names(::G2Zeppelin) = (:mu, :lambda_par, :lambda_perp)
parameter_cardinality(::G2Zeppelin) = Dict(:mu => 3, :lambda_par => 1, :lambda_perp => 1)
parameter_ranges(::G2Zeppelin) = Dict(
    :mu => (-1.0, 1.0),
    :lambda_par => (0.0, 3.0e-9),
    :lambda_perp => (0.0, 3.0e-9)
)

function signal(zep::G2Zeppelin, acq::Acquisition)
    mu = zep.mu ./ max(norm(zep.mu), 1e-12)
    dot_prod = acq.gradient_directions * mu
    b = acq.bvalues
    return @. exp(-b * (zep.lambda_par * dot_prod^2 + zep.lambda_perp * (1 - dot_prod^2)))
end

_reconstruct(::G2Zeppelin, p::AbstractVector) = G2Zeppelin(mu=p[1:3], lambda_par=p[4], lambda_perp=p[5])
