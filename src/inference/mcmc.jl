"""
MCMC posterior sampling with Rician log-likelihood.

Metropolis-Hastings sampler with Gaussian random walk proposals and
reflecting boundary conditions. Self-contained — no external MCMC
library dependencies.

Reference: Harms et al. (2017) "Robust and fast Markov chain Monte Carlo
sampling of diffusion MRI microstructure models", NeuroImage.
"""

using Random, Statistics, LinearAlgebra
using SpecialFunctions: besseli

# ─── Log-space Bessel I0 ──────────────────────────────────────────────────────

"""
    log_besseli0(z)

Numerically stable log of modified Bessel function of the first kind, order 0.

For small z (< 500), uses `log(besseli(0, z))`.
For large z, uses the asymptotic expansion: `z - 0.5 * log(2π * z)`.
"""
function log_besseli0(z::Real)
    if z < 500.0
        return log(besseli(0, z))
    else
        return z - 0.5 * log(2π * z)
    end
end

# ─── Rician log-likelihood ────────────────────────────────────────────────────

"""
    rician_loglikelihood(observed, predicted, sigma)

Compute the Rician log-likelihood for observed dMRI signal given predicted
(noise-free) signal and noise standard deviation sigma.

The Rician distribution:
    p(x | v, σ) = (x/σ²) exp(-(x² + v²)/(2σ²)) I₀(xv/σ²)

Log form (summed over measurements):
    Σᵢ [log(xᵢ) - 2log(σ) - (xᵢ² + vᵢ²)/(2σ²) + log(I₀(xᵢvᵢ/σ²))]

# Arguments
- `observed`: vector of observed signal values (positive)
- `predicted`: vector of predicted signal values (non-negative)
- `sigma`: noise standard deviation (scalar > 0)

# Returns
- Scalar log-likelihood value
"""
function rician_loglikelihood(observed::AbstractVector, predicted::AbstractVector, sigma::Real)
    @assert length(observed) == length(predicted) "observed and predicted must have same length"
    @assert sigma > 0 "sigma must be positive"

    ll = 0.0
    sigma2 = sigma^2

    for i in eachindex(observed)
        x = observed[i]
        v = predicted[i]
        z = x * v / sigma2

        # log(x) - 2*log(sigma) - (x^2 + v^2)/(2*sigma^2) + log(I0(z))
        ll += log(max(x, 1e-30)) - 2.0 * log(sigma) - (x^2 + v^2) / (2.0 * sigma2) + log_besseli0(z)
    end

    return ll
end

# ─── Reflect proposals off bounds ─────────────────────────────────────────────

"""
    _reflect(x, lo, hi)

Reflect a scalar value x into the interval [lo, hi].
"""
function _reflect(x::Real, lo::Real, hi::Real)
    range = hi - lo
    if range <= 0
        return lo
    end
    # Shift to [0, range], reflect, shift back
    y = x - lo
    # Number of full periods
    y = mod(y, 2 * range)
    if y > range
        y = 2 * range - y
    end
    return lo + clamp(y, 0.0, range)
end

# ─── Metropolis-Hastings sampler ──────────────────────────────────────────────

"""
    mcmc_sample(forward_fn, acq, observed, rng;
                n_samples=1000, n_warmup=500, sigma=0.02,
                proposal_std=nothing, init=nothing,
                lower=nothing, upper=nothing)

Draw posterior samples via Metropolis-Hastings with Gaussian random walk
proposals and Rician log-likelihood.

# Arguments
- `forward_fn`: callable `(model, acq, params) -> predicted_signal::Vector`.
  The `model` argument is passed as `nothing` (the closure captures the model).
- `acq`: Acquisition object (or anything the forward_fn expects)
- `observed`: vector of observed signal values
- `rng`: random number generator
- `n_samples`: number of post-warmup samples to collect
- `n_warmup`: number of warmup (burn-in) samples to discard
- `sigma`: Rician noise standard deviation
- `proposal_std`: vector of per-parameter proposal standard deviations
  (default: 5% of parameter range, or 0.01 if no bounds given)
- `init`: initial parameter vector (default: midpoint of bounds, or zeros)
- `lower`: vector of lower bounds (default: -Inf)
- `upper`: vector of upper bounds (default: +Inf)

# Returns
- `(samples, accept_rate)`: matrix of shape `(n_params, n_samples)` and
  scalar acceptance rate over the full chain (warmup + sampling).
"""
function mcmc_sample(
    forward_fn, acq, observed::AbstractVector, rng::AbstractRNG;
    n_samples::Int = 1000,
    n_warmup::Int = 500,
    sigma::Real = 0.02,
    proposal_std::Union{Nothing, AbstractVector} = nothing,
    init::Union{Nothing, AbstractVector} = nothing,
    lower::Union{Nothing, AbstractVector} = nothing,
    upper::Union{Nothing, AbstractVector} = nothing,
)
    # Determine number of parameters from init or bounds
    if init !== nothing
        n_params = length(init)
    elseif lower !== nothing
        n_params = length(lower)
    else
        error("Must provide either `init` or `lower`/`upper` to determine n_params")
    end

    # Defaults
    lo = lower === nothing ? fill(-Inf, n_params) : Float64.(lower)
    hi = upper === nothing ? fill(Inf, n_params) : Float64.(upper)

    if init !== nothing
        current = Float64.(copy(init))
    else
        current = [(isfinite(lo[i]) && isfinite(hi[i])) ? 0.5 * (lo[i] + hi[i]) : 0.0 for i in 1:n_params]
    end

    if proposal_std !== nothing
        prop_std = Float64.(proposal_std)
    else
        # Auto: 5% of parameter range, or 0.01 if unbounded
        prop_std = Float64[
            (isfinite(lo[i]) && isfinite(hi[i])) ? 0.05 * (hi[i] - lo[i]) : 0.01
            for i in 1:n_params
        ]
    end

    # Compute initial log-likelihood
    predicted = forward_fn(nothing, acq, current)
    current_ll = rician_loglikelihood(observed, predicted, sigma)

    # Storage
    total_steps = n_warmup + n_samples
    samples = Matrix{Float64}(undef, n_params, n_samples)
    n_accept = 0

    for step in 1:total_steps
        # Propose new parameters
        proposal = current .+ prop_std .* randn(rng, n_params)

        # Reflect proposals off bounds
        for i in 1:n_params
            if isfinite(lo[i]) && isfinite(hi[i])
                proposal[i] = _reflect(proposal[i], lo[i], hi[i])
            elseif isfinite(lo[i])
                proposal[i] = max(proposal[i], lo[i])
            elseif isfinite(hi[i])
                proposal[i] = min(proposal[i], hi[i])
            end
        end

        # Compute proposed log-likelihood
        predicted_prop = forward_fn(nothing, acq, proposal)
        proposed_ll = rician_loglikelihood(observed, predicted_prop, sigma)

        # Metropolis-Hastings acceptance (symmetric proposal => just likelihood ratio)
        log_alpha = proposed_ll - current_ll

        if log(rand(rng)) < log_alpha
            current = proposal
            current_ll = proposed_ll
            n_accept += 1
        end

        # Store sample (post-warmup only)
        if step > n_warmup
            samples[:, step - n_warmup] = current
        end
    end

    accept_rate = n_accept / total_steps
    return samples, accept_rate
end

# ─── Summary statistics ──────────────────────────────────────────────────────

"""
    mcmc_summary(samples)

Compute summary statistics from MCMC posterior samples.

# Arguments
- `samples`: matrix of shape (n_params, n_samples)

# Returns
- Dict with keys :mean, :std, :median, :q025, :q975 (each a vector of length n_params)
"""
function mcmc_summary(samples::AbstractMatrix)
    n_params = size(samples, 1)

    m = vec(mean(samples, dims=2))
    s = vec(std(samples, dims=2))
    med = [median(samples[i, :]) for i in 1:n_params]
    q025 = [quantile(samples[i, :], 0.025) for i in 1:n_params]
    q975 = [quantile(samples[i, :], 0.975) for i in 1:n_params]

    return Dict(
        :mean => m,
        :std => s,
        :median => med,
        :q025 => q025,
        :q975 => q975,
    )
end
