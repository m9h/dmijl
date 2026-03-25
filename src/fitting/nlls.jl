"""
Non-linear least squares fitting for multi-compartment models.

Uses LsqFit.jl (Levenberg-Marquardt) with box constraints.
"""

using LsqFit

"""Smart default initialization that handles orientation vectors."""
function _default_init(model)
    names = parameter_names(model)
    card = parameter_cardinality(model)
    ranges = parameter_ranges(model)

    init = Float64[]
    for name in names
        lo, hi = ranges[name]
        c = card[name]
        if c == 3 && lo == -1.0 && hi == 1.0
            # Orientation vector: use (0, 0, 1) instead of (0, 0, 0)
            append!(init, [0.0, 0.0, 1.0])
        elseif startswith(String(name), "partial_volume_")
            # Volume fractions: equal split
            push!(init, 0.5)
        else
            push!(init, (lo + hi) / 2)
        end
    end
    return init
end

"""
    fit_mcm(model, acq, data; init=nothing)

Fit a multi-compartment model (or constrained model) to observed signal data.

Returns Dict(:parameters => fitted_params, :residuals => residual_vector).
"""
function fit_mcm(model, acq::Acquisition, data::AbstractVector;
                 init::Union{Nothing, AbstractVector}=nothing)
    lower, upper = get_flat_bounds(model)
    np = nparams(model)

    if init === nothing
        init = _default_init(model)
    end

    # Clamp init to bounds
    init = clamp.(init, lower .+ 1e-15, upper .- 1e-15)

    function model_fn(_, params)
        sig = signal(model, acq, params)
        # Guard against NaN/Inf for numerical stability
        return map(s -> isfinite(s) ? s : 0.0, sig)
    end

    xdata = collect(1.0:length(data))
    fit = curve_fit(model_fn, xdata, Float64.(data), Float64.(init);
                    lower=Float64.(lower), upper=Float64.(upper),
                    autodiff=:finiteforward)

    return Dict(:parameters => fit.param, :residuals => fit.resid)
end

"""
    fit_mcm_batch(model, acq, data_matrix; init=nothing)

Fit multi-compartment model to each row of data_matrix (n_voxels × n_measurements).

Returns Dict(:parameters => (n_voxels × n_params) matrix).
"""
function fit_mcm_batch(model, acq::Acquisition, data::AbstractMatrix;
                       init::Union{Nothing, AbstractVector}=nothing)
    n_voxels = size(data, 1)
    np = nparams(model)
    all_params = zeros(n_voxels, np)
    all_residuals = zeros(n_voxels, size(data, 2))

    for i in 1:n_voxels
        result = fit_mcm(model, acq, data[i, :]; init=init)
        all_params[i, :] = result[:parameters]
        all_residuals[i, :] = result[:residuals]
    end

    return Dict(:parameters => all_params, :residuals => all_residuals)
end
