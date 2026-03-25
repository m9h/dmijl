"""S1Dot — non-diffusing (stationary) compartment. Signal is always 1."""

struct S1Dot <: AbstractCompartment end

parameter_names(::S1Dot) = ()
parameter_cardinality(::S1Dot) = Dict{Symbol,Int}()
parameter_ranges(::S1Dot) = Dict{Symbol,Tuple{Float64,Float64}}()

function signal(::S1Dot, acq::Acquisition)
    return ones(length(acq.bvalues))
end

_reconstruct(::S1Dot, ::AbstractVector) = S1Dot()
