"""
Posterior Predictive Checks (PPC) for SBI model diagnostics.

Verifies that data simulated from posterior samples are consistent
with the observed data.  Complements SBC (calibration) by checking
data-space fit quality.

Pure Julia — no neural networks or AD needed, just statistics.
"""

using Statistics, Random

"""
    posterior_predictive_check(observed, posterior_samples, forward_fn, rng; n_ppc=100) → Vector{Float64}

Compute per-observation p-values via posterior predictive checks.

For each posterior sample, simulate a dataset via `forward_fn(theta, rng)`.
The test statistic for each observation is its squared residual from the
posterior-predictive mean.  The p-value is the fraction of simulated
datasets whose test statistic (per observation) is >= the observed one.

# Arguments
- `observed::AbstractVector`: observed data, length `n_obs`.
- `posterior_samples::AbstractVector`: vector of posterior parameter draws.
- `forward_fn(theta, rng)`: function that takes a single parameter value
  and an RNG, returns a simulated dataset of length `n_obs`.
- `rng`: random number generator.
- `n_ppc`: number of posterior-predictive replications per posterior sample.

# Returns
`Vector{Float64}` of p-values, one per observation.
"""
function posterior_predictive_check(
    observed::AbstractVector,
    posterior_samples::AbstractVector,
    forward_fn,
    rng::AbstractRNG;
    n_ppc::Int = 100,
)::Vector{Float64}
    n_obs = length(observed)
    n_post = length(posterior_samples)

    # 1. Generate posterior-predictive replicates
    #    For each posterior sample, draw n_ppc simulated datasets
    simulated = Vector{Vector{Float64}}()
    for s in 1:n_post
        theta = posterior_samples[s]
        for _ in 1:n_ppc
            y_sim = forward_fn(theta, rng)
            push!(simulated, collect(Float64, y_sim))
        end
    end
    n_total = length(simulated)

    # 2. Compute posterior-predictive mean per observation
    pp_mean = zeros(n_obs)
    for y in simulated
        pp_mean .+= y
    end
    pp_mean ./= n_total

    # 3. Observed test statistic per observation: squared residual from pp mean
    T_obs = (observed .- pp_mean).^2

    # 4. For each replicate, compute the same test statistic and count exceedances
    counts = zeros(Int, n_obs)
    for y in simulated
        T_sim = (y .- pp_mean).^2
        for i in 1:n_obs
            if T_sim[i] >= T_obs[i]
                counts[i] += 1
            end
        end
    end

    # 5. p-value = fraction of replicates with T_sim >= T_obs
    pvals = counts ./ n_total
    return pvals
end

"""
    ppc_summary(p_values; alpha=0.05) → NamedTuple

Summarise posterior predictive check p-values.

# Returns
Named tuple with fields:
- `n_flagged`: number of observations with p < alpha
- `fraction_flagged`: fraction of observations with p < alpha
- `median_pvalue`: median p-value across all observations
"""
function ppc_summary(
    p_values::AbstractVector;
    alpha::Real = 0.05,
)
    flagged = p_values .< alpha
    n_flagged = sum(flagged)
    fraction_flagged = n_flagged / length(p_values)
    med = median(p_values)
    return (n_flagged=n_flagged, fraction_flagged=fraction_flagged, median_pvalue=med)
end
