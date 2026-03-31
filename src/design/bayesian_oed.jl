"""
    Bayesian Optimal Experimental Design via Expected Information Gain.

Provides two estimators:
- PCE (Prior Contrastive Estimation): lower bound on EIG
- Variational: upper bound on EIG using an MDN approximate posterior

The key insight: the existing MDN from Phase 1 serves directly as the
variational approximate posterior q(θ|y,ξ) for the EIG upper bound.

References:
- Foster et al. (2019) "Variational Bayesian Optimal Experimental Design"
  NeurIPS 2019, arXiv:1903.05480
- Kleinegesse & Gutmann (2020) ICML, arXiv:2002.08129
"""

"""
    logsumexp(x) -> Float64

Numerically stable log-sum-exp.
"""
function logsumexp(x::AbstractVector)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

"""
    eig_pce(model, acq, prior_samples; sigma=0.02, n_contrastive=100, seed=42) -> Float64

Estimate Expected Information Gain via Prior Contrastive Estimation (PCE).

This is a lower bound on EIG (tighter with more contrastive samples):

    EIG(ξ) ≥ E_{θ,y}[log p(y|θ,ξ) - log(1/L ∑_l p(y|θ_l,ξ))]

# Arguments
- `model`: forward model (MultiCompartmentModel or ConstrainedModel)
- `acq`: candidate acquisition to evaluate
- `prior_samples`: (n_params × N) matrix of prior parameter samples
- `sigma`: noise standard deviation
- `n_contrastive`: number of contrastive prior samples (L)
- `seed`: random seed
"""
function eig_pce(model, acq::Acquisition, prior_samples::AbstractMatrix;
                 sigma::Float64=0.02, n_contrastive::Int=100, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    N = size(prior_samples, 2)
    eig_sum = 0.0

    for n in 1:N
        theta_n = prior_samples[:, n]
        S_n = signal(model, acq, theta_n)
        y_n = S_n .+ sigma .* randn(rng, length(S_n))

        # log p(y|θ_n) under Gaussian noise
        log_p_true = -0.5 * sum((y_n .- S_n).^2) / sigma^2

        # Contrastive: log (1/L) ∑_l p(y|θ_l)
        indices = rand(rng, 1:N, n_contrastive)
        log_probs = map(indices) do l
            S_l = signal(model, acq, prior_samples[:, l])
            -0.5 * sum((y_n .- S_l).^2) / sigma^2
        end
        log_marginal = logsumexp(log_probs) - log(n_contrastive)

        eig_sum += log_p_true - log_marginal
    end

    return eig_sum / N
end

"""
    mdn_log_density(theta, pi_w, mu, log_sigma) -> Float64

Log density of `theta` under a Gaussian mixture with weights `pi_w`,
means `mu`, and log-standard-deviations `log_sigma`.

- `theta`: (D,) parameter vector
- `pi_w`: (K,) mixture weights (summing to 1)
- `mu`: (D, K) component means
- `log_sigma`: (D, K) component log-std-devs
"""
function mdn_log_density(theta::AbstractVector, pi_w::AbstractVector,
                          mu::AbstractMatrix, log_sigma::AbstractMatrix)
    K = length(pi_w)
    D = length(theta)
    log_probs = map(1:K) do k
        sigma_k = exp.(log_sigma[:, k])
        diff = theta .- mu[:, k]
        log_p = -0.5 * sum((diff ./ sigma_k).^2) - sum(log.(sigma_k)) -
                0.5 * D * log(2pi) + log(max(pi_w[k], 1e-30))
        log_p
    end
    return logsumexp(log_probs)
end

"""
    eig_variational(model, acq, mdn_model, mdn_ps, mdn_st, prior_samples;
                    sigma=0.02, n_components=5, param_dim, seed=42) -> Float64

Estimate EIG via the variational upper bound using an MDN approximate posterior:

    EIG(ξ) ≤ E_{θ,y}[log q_MDN(θ|y,ξ)] - H[prior]

The MDN provides q(θ|y) as a Gaussian mixture density.

# Arguments
- `model`: forward model
- `acq`: candidate acquisition
- `mdn_model, mdn_ps, mdn_st`: trained MDN (from `build_mdn` + `train_mdn!`)
- `prior_samples`: (n_params × N) prior samples
- `sigma`: noise standard deviation
- `n_components`: number of MDN mixture components
- `param_dim`: number of tissue parameters
- `seed`: random seed
"""
function eig_variational(model, acq::Acquisition,
                          mdn_model, mdn_ps, mdn_st,
                          prior_samples::AbstractMatrix;
                          sigma::Float64=0.02,
                          n_components::Int=5,
                          param_dim::Int,
                          seed::Int=42)
    rng = Random.MersenneTwister(seed)
    N = size(prior_samples, 2)
    eig_sum = 0.0

    for n in 1:N
        theta_n = prior_samples[:, n]
        S_n = signal(model, acq, theta_n)
        y_n = S_n .+ sigma .* randn(rng, length(S_n))

        # Evaluate log q_MDN(θ_n | y_n)
        y_input = reshape(Float32.(y_n), :, 1)
        pi_w, mu, log_sigma, _ = mdn_forward(mdn_model, mdn_ps, mdn_st, y_input;
                                              n_components=n_components,
                                              param_dim=param_dim)
        log_q = mdn_log_density(Float64.(theta_n),
                                 Float64.(pi_w[:, 1]),
                                 Float64.(mu[:, :, 1]),
                                 Float64.(log_sigma[:, :, 1]))
        eig_sum += log_q
    end

    # Subtract uniform prior entropy: H[prior] = ∑ log(range_i)
    lo, hi = get_flat_bounds(model)
    prior_entropy = sum(log.(max.(hi .- lo, 1e-30)))

    return eig_sum / N - prior_entropy
end
