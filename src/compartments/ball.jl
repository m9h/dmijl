"""G1Ball — isotropic free diffusion compartment. Signal: exp(-b * λ_iso)."""

Base.@kwdef struct G1Ball <: AbstractCompartment
    lambda_iso::Float64
end

parameter_names(::G1Ball) = (:lambda_iso,)
parameter_cardinality(::G1Ball) = Dict(:lambda_iso => 1)
parameter_ranges(::G1Ball) = Dict(:lambda_iso => (0.0, 3.0e-9))

function signal(ball::G1Ball, acq::Acquisition)
    return @. exp(-acq.bvalues * ball.lambda_iso)
end

_reconstruct(::G1Ball, p::AbstractVector) = G1Ball(lambda_iso=p[1])
