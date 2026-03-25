"""C1Stick — zero-radius cylinder (stick) compartment. Signal: exp(-b * λ_par * (g·μ)²)."""

Base.@kwdef struct C1Stick <: AbstractCompartment
    mu::Vector{Float64}       # unit orientation vector (3D Cartesian)
    lambda_par::Float64       # parallel diffusivity (m²/s)
end

parameter_names(::C1Stick) = (:mu, :lambda_par)
parameter_cardinality(::C1Stick) = Dict(:mu => 3, :lambda_par => 1)
parameter_ranges(::C1Stick) = Dict(
    :mu => (-1.0, 1.0),
    :lambda_par => (0.0, 3.0e-9)
)

function signal(stick::C1Stick, acq::Acquisition)
    mu = stick.mu ./ max(norm(stick.mu), 1e-12)
    dot_prod = acq.gradient_directions * mu
    return @. exp(-acq.bvalues * stick.lambda_par * dot_prod^2)
end

_reconstruct(::C1Stick, p::AbstractVector) = C1Stick(mu=p[1:3], lambda_par=p[4])
