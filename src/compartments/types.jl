"""
Composable compartment model framework for dMRI signal simulation.

Each compartment is a subtype of `AbstractCompartment` and implements:
- `signal(comp, acq)` → Vector{Float64} of signal attenuation
- `parameter_names(comp)` → tuple of Symbol
- `parameter_cardinality(comp)` → Dict{Symbol, Int}
- `parameter_ranges(comp)` → Dict{Symbol, Tuple{Float64, Float64}}
"""

abstract type AbstractCompartment end

"""Return parameter name symbols for a compartment."""
function parameter_names end

"""Return Dict{Symbol, Tuple{Float64,Float64}} of parameter bounds."""
function parameter_ranges end

"""Return Dict{Symbol, Int} mapping parameter name to its dimensionality."""
function parameter_cardinality end

"""Total number of scalar values in the flat parameter vector."""
function nparams(c::AbstractCompartment)
    card = parameter_cardinality(c)
    isempty(card) ? 0 : sum(values(card))
end

"""
    signal(compartment::AbstractCompartment, acq::Acquisition)

Compute predicted signal attenuation. Returns Vector{Float64} of length n_measurements(acq).
"""
function signal end

"""
    _reconstruct(comp::AbstractCompartment, params::AbstractVector)

Create a new compartment instance from a flat parameter vector.
Each concrete compartment type must implement this.
"""
function _reconstruct end
