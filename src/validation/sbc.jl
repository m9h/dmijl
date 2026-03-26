"""
    Simulation-Based Calibration (SBC) for validating posterior inference.

Implements the rank-histogram diagnostic from Talts et al. (2018).
If the posterior is well-calibrated, the rank of the true parameter among
posterior samples should be uniformly distributed over [0, n_posterior_samples].

References
----------
Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
Validating Bayesian inference algorithms with simulation-based calibration.
arXiv:1804.06788.
"""

using Random, Statistics
using SpecialFunctions: logabsgamma

"""
    compute_rank(theta_true::Real, posterior_samples::AbstractVector{<:Real}) -> Int

Compute the rank of `theta_true` among `posterior_samples`, defined as the
number of posterior samples strictly less than `theta_true`.

Returns an integer in `[0, length(posterior_samples)]`.
"""
function compute_rank(theta_true::Real, posterior_samples::AbstractVector{<:Real})::Int
    return count(s -> s < theta_true, posterior_samples)
end

"""
    sbc_ranks(prior_sampler, simulator, posterior_sampler, n_simulations;
              n_posterior_samples=99, rng=Random.default_rng()) -> Vector{Int}

Run the full SBC loop and return a vector of ranks.

# Arguments
- `prior_sampler`: `(rng) -> theta` — sample a single parameter from the prior.
- `simulator`: `(theta, rng) -> data` — simulate observed data given a parameter.
- `posterior_sampler`: `(data, rng, n_samples) -> Vector` — draw `n_samples`
  posterior samples given observed data.
- `n_simulations`: number of SBC repetitions.

# Keyword arguments
- `n_posterior_samples`: number of posterior draws per repetition (default 99).
- `rng`: random number generator (default `Random.default_rng()`).

# Returns
A `Vector{Int}` of length `n_simulations` where each entry is in
`[0, n_posterior_samples]`.
"""
function sbc_ranks(
    prior_sampler,
    simulator,
    posterior_sampler,
    n_simulations::Int;
    n_posterior_samples::Int=99,
    rng::AbstractRNG=Random.default_rng(),
)::Vector{Int}
    ranks = Vector{Int}(undef, n_simulations)
    for i in 1:n_simulations
        theta_true = prior_sampler(rng)
        data = simulator(theta_true, rng)
        posterior_samples = posterior_sampler(data, rng, n_posterior_samples)
        ranks[i] = compute_rank(theta_true, posterior_samples)
    end
    return ranks
end

"""
    sbc_histogram(ranks, n_posterior_samples; n_bins=20) -> (bin_edges, counts)

Bin the SBC ranks into a histogram.

# Arguments
- `ranks`: vector of integer ranks in `[0, n_posterior_samples]`.
- `n_posterior_samples`: the maximum possible rank (L).

# Keyword arguments
- `n_bins`: number of bins (default 20, clamped to at most `n_posterior_samples + 1`).

# Returns
- `bin_edges`: `Vector{Float64}` of length `n_bins + 1`.
- `counts`: `Vector{Int}` of length `n_bins`.
"""
function sbc_histogram(
    ranks::AbstractVector{<:Integer},
    n_posterior_samples::Int;
    n_bins::Int=20,
)
    n_bins = min(n_bins, n_posterior_samples + 1)
    bin_edges = range(0.0, stop=Float64(n_posterior_samples + 1), length=n_bins + 1)
    edges_vec = collect(bin_edges)
    counts = zeros(Int, n_bins)
    for r in ranks
        # Find which bin this rank falls into
        bin_idx = searchsortedlast(edges_vec, Float64(r))
        bin_idx = clamp(bin_idx, 1, n_bins)
        counts[bin_idx] += 1
    end
    return (edges_vec, counts)
end

"""
    sbc_uniformity_test(ranks, n_posterior_samples; alpha=0.01) -> (p_value, is_calibrated)

Test whether the rank distribution is consistent with Uniform(0, n_posterior_samples)
using a chi-squared goodness-of-fit test.

The ranks are binned and compared against the expected uniform counts.
A Kolmogorov-Smirnov-style test on discrete data is tricky, so we use
chi-squared which is straightforward and well-suited for rank histograms.

# Arguments
- `ranks`: vector of integer ranks.
- `n_posterior_samples`: maximum possible rank (L).

# Keyword arguments
- `alpha`: significance level (default 0.01).

# Returns
- `p_value::Float64`: p-value from the chi-squared test.
- `is_calibrated::Bool`: `true` if `p_value > alpha`.
"""
function sbc_uniformity_test(
    ranks::AbstractVector{<:Integer},
    n_posterior_samples::Int;
    alpha::Float64=0.01,
)
    n = length(ranks)
    # Choose bins so expected count per bin >= 5 (chi-squared requirement)
    n_bins = min(20, n_posterior_samples + 1)
    while n / n_bins < 5 && n_bins > 2
        n_bins -= 1
    end

    _, counts = sbc_histogram(ranks, n_posterior_samples; n_bins=n_bins)

    # Expected count per bin under uniformity
    expected = n / n_bins

    # Chi-squared statistic
    chi2 = sum((Float64(c) - expected)^2 / expected for c in counts)

    # Degrees of freedom = n_bins - 1
    df = n_bins - 1

    # Compute p-value from chi-squared distribution using the regularized
    # upper incomplete gamma function: p = 1 - gamma_inc(df/2, chi2/2)
    # We use the series/continued-fraction expansion of the regularized
    # lower incomplete gamma function.
    p_value = _chi2_pvalue(chi2, df)

    return (p_value, p_value > alpha)
end

"""
    _chi2_pvalue(chi2, df)

Compute the p-value for a chi-squared statistic with `df` degrees of freedom.
P(X >= chi2) = 1 - regularized_lower_gamma(df/2, chi2/2).

Uses the series expansion of the regularized lower incomplete gamma function.
"""
function _chi2_pvalue(chi2::Real, df::Int)
    a = df / 2.0
    x = chi2 / 2.0

    if x < 0
        return 1.0
    end
    if x == 0
        return 1.0
    end

    # Use series expansion for lower incomplete gamma: gamma(a, x) / Gamma(a)
    # P(a, x) = sum_{k=0}^{inf} (-1)^k * x^(a+k) / (k! * (a+k)) / Gamma(a)
    # Better: use the regularized series:
    # P(a, x) = e^(-x) * x^a * sum_{k=0}^{inf} x^k / Gamma(a + k + 1)

    if x < a + 1.0
        # Series expansion for lower incomplete gamma
        p_lower = _gamma_inc_series(a, x)
        return 1.0 - p_lower
    else
        # Continued fraction expansion for upper incomplete gamma
        p_upper = _gamma_inc_cf(a, x)
        return p_upper
    end
end

"""Series expansion for regularized lower incomplete gamma P(a,x)."""
function _gamma_inc_series(a::Float64, x::Float64)
    max_iter = 200
    eps_val = 1e-12

    if x == 0.0
        return 0.0
    end

    ap = a
    sum_val = 1.0 / a
    del = 1.0 / a

    for _ in 1:max_iter
        ap += 1.0
        del *= x / ap
        sum_val += del
        if abs(del) < abs(sum_val) * eps_val
            break
        end
    end

    # P(a,x) = e^(-x) * x^a * sum / Gamma(a)
    # log version to avoid overflow
    log_val = -x + a * log(x) + log(sum_val) - first(logabsgamma(a))
    return exp(log_val)
end

"""Continued fraction expansion for regularized upper incomplete gamma Q(a,x)."""
function _gamma_inc_cf(a::Float64, x::Float64)
    max_iter = 200
    eps_val = 1e-12
    fpmin = 1e-30

    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d

    for i in 1:max_iter
        an = -Float64(i) * (Float64(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin
            d = fpmin
        end
        c = b + an / c
        if abs(c) < fpmin
            c = fpmin
        end
        d = 1.0 / d
        del = d * c
        h *= del
        if abs(del - 1.0) < eps_val
            break
        end
    end

    log_val = -x + a * log(x) + log(h) - first(logabsgamma(a))
    return exp(log_val)
end
