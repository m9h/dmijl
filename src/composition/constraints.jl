"""
Parameter constraints for multi-compartment models.

Constraints wrap a MultiCompartmentModel, removing constrained parameters from
the free parameter list and computing their values at signal-evaluation time.
"""

# ---- Abstract constraint ----

abstract type AbstractConstraint end

# ---- ConstrainedModel wrapper ----

struct ConstrainedModel{M, C <: Tuple}
    base_model::M
    constraints::C
end

# ---- FixedParameter ----

struct FixedParameter <: AbstractConstraint
    name::Symbol
    value::Float64
    index::Int      # position in the base model's flat parameter vector
end

function set_fixed_parameter(mcm::MultiCompartmentModel, name::Symbol, value::Float64)
    names = parameter_names(mcm)
    card = parameter_cardinality(mcm)

    # Find flat index for the named parameter
    idx = 1
    found = false
    target_idx = 0
    target_card = 0
    for n in names
        c = card[n]
        if n == name
            target_idx = idx
            target_card = c
            found = true
            break
        end
        idx += c
    end
    found || error("Parameter $name not found in model")
    target_card == 1 || error("FixedParameter only supports scalar parameters (cardinality=1), got $target_card for $name")

    constraint = FixedParameter(name, value, target_idx)
    return ConstrainedModel(mcm, (constraint,))
end

# ---- VolumeFractionUnity ----

struct VolumeFractionUnity <: AbstractConstraint
    removed_name::Symbol
    removed_index::Int   # flat index of the removed (last) fraction
    other_indices::Vector{Int}  # flat indices of the other fractions
end

function set_volume_fraction_unity(mcm::MultiCompartmentModel)
    names = parameter_names(mcm)
    card = parameter_cardinality(mcm)
    N = length(mcm.compartments)

    # Find indices of all partial_volume_* parameters
    frac_indices = Int[]
    frac_names = Symbol[]
    idx = 1
    for n in names
        c = card[n]
        if startswith(String(n), "partial_volume_")
            push!(frac_indices, idx)
            push!(frac_names, n)
        end
        idx += c
    end

    removed_name = frac_names[end]
    removed_idx = frac_indices[end]
    other_idx = frac_indices[1:end-1]

    constraint = VolumeFractionUnity(removed_name, removed_idx, other_idx)
    return ConstrainedModel(mcm, (constraint,))
end

# ---- LinkedParameter (copies value from one parameter to another) ----

struct LinkedParameter <: AbstractConstraint
    target_name::Symbol         # name of the derived parameter
    target_indices::Vector{Int} # flat indices of the target parameter
    source_indices::Vector{Int} # flat indices of the source parameter to copy from
end

# ---- TortuosityConstraint ----

struct TortuosityConstraint <: AbstractConstraint
    target::Symbol         # parameter to derive (e.g. :lambda_perp)
    target_index::Int      # flat index of target in base model
    lambda_par_index::Int  # flat index of lambda_par
    frac_index::Int        # flat index of volume fraction
end

function set_tortuosity(mcm::MultiCompartmentModel;
                        target::Symbol,
                        lambda_par_name::Symbol,
                        volume_fraction_name::Symbol)
    names = parameter_names(mcm)
    card = parameter_cardinality(mcm)

    function find_index(name)
        idx = 1
        for n in names
            c = card[n]
            n == name && return idx
            idx += c
        end
        error("Parameter $name not found")
    end

    target_idx = find_index(target)
    lp_idx = find_index(lambda_par_name)
    frac_idx = find_index(volume_fraction_name)

    constraint = TortuosityConstraint(target, target_idx, lp_idx, frac_idx)
    return ConstrainedModel(mcm, (constraint,))
end

# ---- Trait functions for ConstrainedModel ----

function _removed_names(cm::ConstrainedModel)
    removed = Symbol[]
    for c in cm.constraints
        if c isa FixedParameter
            push!(removed, c.name)
        elseif c isa VolumeFractionUnity
            push!(removed, c.removed_name)
        elseif c isa TortuosityConstraint
            push!(removed, c.target)
        elseif c isa LinkedParameter
            push!(removed, c.target_name)
        end
    end
    return removed
end

function _removed_count(cm::ConstrainedModel)
    length(_removed_flat_indices(cm))
end

function parameter_names(cm::ConstrainedModel)
    base_names = parameter_names(cm.base_model)
    removed = Set(_removed_names(cm))
    return Tuple(n for n in base_names if n ∉ removed)
end

function parameter_cardinality(cm::ConstrainedModel)
    base_card = parameter_cardinality(cm.base_model)
    removed = Set(_removed_names(cm))
    return Dict(k => v for (k, v) in base_card if k ∉ removed)
end

function parameter_ranges(cm::ConstrainedModel)
    base_ranges = parameter_ranges(cm.base_model)
    removed = Set(_removed_names(cm))
    return Dict(k => v for (k, v) in base_ranges if k ∉ removed)
end

function nparams(cm::ConstrainedModel)
    nparams(cm.base_model) - _removed_count(cm)
end

function get_flat_bounds(cm::ConstrainedModel)
    base_lower, base_upper = get_flat_bounds(cm.base_model)
    removed_indices = _removed_flat_indices(cm)
    keep = setdiff(1:length(base_lower), removed_indices)
    return base_lower[keep], base_upper[keep]
end

function _removed_flat_indices(cm::ConstrainedModel)
    indices = Int[]
    for c in cm.constraints
        if c isa FixedParameter
            push!(indices, c.index)
        elseif c isa VolumeFractionUnity
            push!(indices, c.removed_index)
        elseif c isa TortuosityConstraint
            push!(indices, c.target_index)
        elseif c isa LinkedParameter
            append!(indices, c.target_indices)
        end
    end
    return sort(indices)
end

# ---- Signal computation ----

function signal(cm::ConstrainedModel, acq::Acquisition, free_params::AbstractVector)
    # Expand free_params into full_params by inserting constrained values
    removed_indices = _removed_flat_indices(cm)
    n_full = nparams(cm.base_model)
    full_params = zeros(n_full)

    # Fill in free params first (skipping removed positions)
    free_idx = 1
    for i in 1:n_full
        if i ∉ removed_indices
            full_params[i] = free_params[free_idx]
            free_idx += 1
        end
    end

    # Now fill constrained values (order matters: linked/fixed first, then derived)
    for c in cm.constraints
        if c isa FixedParameter
            full_params[c.index] = c.value
        elseif c isa LinkedParameter
            for (ti, si) in zip(c.target_indices, c.source_indices)
                full_params[ti] = full_params[si]
            end
        elseif c isa VolumeFractionUnity
            # Derive last fraction = 1 - sum(others)
            other_sum = sum(full_params[i] for i in c.other_indices)
            full_params[c.removed_index] = clamp(1.0 - other_sum, 0.0, 1.0)
        elseif c isa TortuosityConstraint
            # lambda_perp = lambda_par * (1 - f_intra)
            lp = full_params[c.lambda_par_index]
            f = full_params[c.frac_index]
            full_params[c.target_index] = lp * (1.0 - f)
        end
    end

    return signal(cm.base_model, acq, full_params)
end
