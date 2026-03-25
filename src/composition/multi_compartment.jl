"""
Multi-compartment model composition.

Combines multiple AbstractCompartment instances into a weighted mixture model.
Parameter layout: [model1_params..., model2_params..., ..., f_1, f_2, ..., f_N]
"""

struct MultiCompartmentModel{T <: Tuple}
    compartments::T
end

MultiCompartmentModel(models...) = MultiCompartmentModel(models)

# ---- Trait functions ----

function parameter_names(mcm::MultiCompartmentModel)
    seen = Dict{Symbol, Int}()  # name → count of times seen
    names = Symbol[]

    for (i, comp) in enumerate(mcm.compartments)
        for pname in parameter_names(comp)
            if haskey(seen, pname)
                # Collision: this is the 2nd+ occurrence
                seen[pname] += 1
                push!(names, Symbol(pname, :_, i))
            else
                seen[pname] = 1
                push!(names, pname)
            end
        end
    end

    # Volume fractions
    for i in 1:length(mcm.compartments)
        push!(names, Symbol(:partial_volume_, i))
    end

    return Tuple(names)
end

function parameter_cardinality(mcm::MultiCompartmentModel)
    result = Dict{Symbol, Int}()
    seen = Dict{Symbol, Int}()

    for (i, comp) in enumerate(mcm.compartments)
        card = parameter_cardinality(comp)
        for pname in parameter_names(comp)
            if haskey(seen, pname)
                seen[pname] += 1
                result[Symbol(pname, :_, i)] = card[pname]
            else
                seen[pname] = 1
                result[pname] = card[pname]
            end
        end
    end

    for i in 1:length(mcm.compartments)
        result[Symbol(:partial_volume_, i)] = 1
    end

    return result
end

function parameter_ranges(mcm::MultiCompartmentModel)
    result = Dict{Symbol, Tuple{Float64, Float64}}()
    seen = Dict{Symbol, Int}()

    for (i, comp) in enumerate(mcm.compartments)
        ranges = parameter_ranges(comp)
        for pname in parameter_names(comp)
            if haskey(seen, pname)
                seen[pname] += 1
                result[Symbol(pname, :_, i)] = ranges[pname]
            else
                seen[pname] = 1
                result[pname] = ranges[pname]
            end
        end
    end

    for i in 1:length(mcm.compartments)
        result[Symbol(:partial_volume_, i)] = (0.0, 1.0)
    end

    return result
end

function nparams(mcm::MultiCompartmentModel)
    sum(nparams(c) for c in mcm.compartments) + length(mcm.compartments)
end

# ---- Signal computation ----

function signal(mcm::MultiCompartmentModel, acq::Acquisition, params::AbstractVector)
    N = length(mcm.compartments)
    n_meas = length(acq.bvalues)
    result = zeros(n_meas)
    idx = 1

    for i in 1:N
        comp = mcm.compartments[i]
        np = nparams(comp)
        comp_params = params[idx:idx+np-1]
        idx += np
        comp_new = _reconstruct(comp, comp_params)
        frac = params[end - N + i]
        result .+= frac .* signal(comp_new, acq)
    end

    return result
end

# ---- Parameter packing utilities ----

function parameter_dictionary_to_array(mcm::MultiCompartmentModel, dict::Dict{Symbol})
    names = parameter_names(mcm)
    card = parameter_cardinality(mcm)
    result = Float64[]
    for name in names
        vals = dict[name]
        append!(result, vals)
    end
    return result
end

function parameter_array_to_dictionary(mcm::MultiCompartmentModel, arr::AbstractVector)
    names = parameter_names(mcm)
    card = parameter_cardinality(mcm)
    result = Dict{Symbol, Vector{Float64}}()
    idx = 1
    for name in names
        c = card[name]
        result[name] = arr[idx:idx+c-1]
        idx += c
    end
    return result
end

# ---- Bounds ----

function get_flat_bounds(mcm::MultiCompartmentModel)
    names = parameter_names(mcm)
    card = parameter_cardinality(mcm)
    ranges = parameter_ranges(mcm)

    lower = Float64[]
    upper = Float64[]

    for name in names
        lo, hi = ranges[name]
        c = card[name]
        for _ in 1:c
            push!(lower, lo)
            push!(upper, hi)
        end
    end

    return lower, upper
end
