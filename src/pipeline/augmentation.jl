"""
Training data augmentation utilities for SBI pipelines.

Provides variable-SNR Rician noise, b0 normalization, and label-switching
fix for multi-fiber models.  These augmentations improved the flow NPE
from 6.4 deg to 2.8 deg angular error in dmipy-jax.
"""

using Random, Statistics

# ---- Fiber layout descriptor ----

"""
    FiberLayout(; fraction_indices, orientation_indices)

Describes which parameter indices correspond to volume fractions and their
associated orientation vectors in a multi-fiber model.

# Fields
- `fraction_indices::Vector{Int}` — indices into the parameter vector for f1, f2, …
- `orientation_indices::Vector{UnitRange{Int}}` — index ranges for mu1, mu2, …
  (must be same length as `fraction_indices`)
"""
struct FiberLayout
    fraction_indices::Vector{Int}
    orientation_indices::Vector{UnitRange{Int}}

    function FiberLayout(; fraction_indices, orientation_indices)
        @assert length(fraction_indices) == length(orientation_indices) "Must have one orientation range per fraction"
        new(fraction_indices, orientation_indices)
    end
end

# ---- Variable SNR noise ----

"""
    add_variable_snr_noise(signals, rng; snr_range=(10.0, 50.0))

Add Rician noise with a per-sample SNR drawn uniformly from `snr_range`.

For each sample (column), draws SNR ~ Uniform(snr_range...), computes
sigma = 1/SNR, and produces Rician noise:
    noisy = sqrt((signal + sigma*n1)^2 + (sigma*n2)^2)
where n1, n2 ~ Normal(0, 1).

# Arguments
- `signals::AbstractMatrix` — clean signals, size `(n_meas, n_samples)`
- `rng::AbstractRNG` — random number generator

# Returns
- `noisy::Matrix{Float64}` — noisy signals, same size as input, all values >= 0
"""
function add_variable_snr_noise(signals::AbstractMatrix, rng::AbstractRNG;
                                 snr_range::Tuple{Real,Real}=(10.0, 50.0))
    n_meas, n_samples = size(signals)
    lo, hi = Float64.(snr_range)
    # Per-sample SNR: shape (1, n_samples)
    snr = rand(rng, n_samples) .* (hi - lo) .+ lo
    sigma = reshape(1.0 ./ snr, 1, n_samples)
    # Rician noise: two independent Gaussian channels
    n1 = randn(rng, n_meas, n_samples) .* sigma
    n2 = randn(rng, n_meas, n_samples) .* sigma
    return @. sqrt((signals + n1)^2 + n2^2)
end

# ---- B0 normalization ----

"""
    normalize_b0(signals, bvalues)

Normalize signals by dividing each sample by its mean b=0 signal.

b=0 volumes are identified as those with b-value < 100e6 s/m^2 (FSL convention
for b-values in s/mm^2 stored internally as s/m^2).

# Arguments
- `signals::AbstractMatrix` — size `(n_meas, n_samples)`
- `bvalues::AbstractVector` — b-values, length `n_meas`

# Returns
- `normalized::Matrix{Float64}` — signals divided by mean b=0 per sample
"""
function normalize_b0(signals::AbstractMatrix, bvalues::AbstractVector)
    b0_mask = bvalues .< 100e6
    if !any(b0_mask)
        return copy(signals)
    end
    # Mean b=0 signal per sample: shape (1, n_samples)
    b0_mean = mean(signals[b0_mask, :], dims=1)
    b0_mean = max.(b0_mean, 1e-10)  # avoid division by zero
    return signals ./ b0_mean
end

# ---- Label switching fix ----

"""
    fix_label_switching(params, layout::FiberLayout)

Reorder multi-fiber parameters so that volume fractions are in decreasing
order (f1 >= f2 >= ...).  When fractions are swapped, the corresponding
orientation vectors are swapped too.

# Arguments
- `params::AbstractMatrix` — size `(n_params, n_samples)`
- `layout::FiberLayout` — describes fraction and orientation indices

# Returns
- `fixed::Matrix` — reordered copy of params
"""
function fix_label_switching(params::AbstractMatrix, layout::FiberLayout)
    fixed = copy(params)
    n_fibers = length(layout.fraction_indices)
    n_samples = size(params, 2)

    for j in 1:n_samples
        # Get volume fractions for this sample
        fracs = [fixed[layout.fraction_indices[k], j] for k in 1:n_fibers]
        # Sort indices by decreasing fraction
        order = sortperm(fracs, rev=true)

        if order != 1:n_fibers
            # Need to reorder — collect values first, then assign
            new_fracs = fracs[order]
            new_oris = [fixed[layout.orientation_indices[k], j] for k in order]

            for k in 1:n_fibers
                fixed[layout.fraction_indices[k], j] = new_fracs[k]
                fixed[layout.orientation_indices[k], j] = new_oris[k]
            end
        end
    end
    return fixed
end

# ---- Full augmentation pipeline ----

"""
    augment_training_batch(params, signals, bvalues, rng; kwargs...) -> (params_aug, signals_aug)

Apply a sequence of training data augmentations:
1. Fix label switching (if `fix_switching=true` and `param_layout` provided)
2. Add variable-SNR Rician noise
3. Normalize by b=0 signal (if `normalize=true`)

# Arguments
- `params::AbstractMatrix` — size `(n_params, n_samples)`
- `signals::AbstractMatrix` — size `(n_meas, n_samples)`, clean signals
- `bvalues::AbstractVector` — b-values, length `n_meas`
- `rng::AbstractRNG` — random number generator
- `snr_range::Tuple` — SNR range for variable noise (default `(10.0, 50.0)`)
- `normalize::Bool` — whether to apply b0 normalization (default `true`)
- `fix_switching::Bool` — whether to fix label switching (default `false`)
- `param_layout::Union{Nothing, FiberLayout}` — layout for label switching

# Returns
- `(params_aug, signals_aug)` — augmented parameters and signals
"""
function augment_training_batch(params::AbstractMatrix, signals::AbstractMatrix,
                                 bvalues::AbstractVector, rng::AbstractRNG;
                                 snr_range::Tuple{Real,Real}=(10.0, 50.0),
                                 normalize::Bool=true,
                                 fix_switching::Bool=false,
                                 param_layout::Union{Nothing, FiberLayout}=nothing)
    params_aug = copy(params)
    signals_aug = copy(signals)

    # 1. Fix label switching
    if fix_switching && param_layout !== nothing
        params_aug = fix_label_switching(params_aug, param_layout)
    end

    # 2. Add variable-SNR Rician noise
    signals_aug = add_variable_snr_noise(signals_aug, rng; snr_range=snr_range)

    # 3. Normalize by b=0 signal
    if normalize
        signals_aug = normalize_b0(signals_aug, bvalues)
    end

    return params_aug, signals_aug
end
