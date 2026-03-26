"""
Mixture Density Network (MDN) for amortized posterior inference.

Maps observed dMRI signals to a Gaussian mixture posterior over tissue
parameters. Lightweight baseline -- no normalizing flow, just a shared MLP
backbone with three output heads (mixture weights, means, log-std-devs).

Architecture:
  signal (obs_dim,) --> MLP backbone --> concat output vector
  split into: pi_logits (K,) | mu_flat (K*D,) | log_sigma_flat (K*D,)

Loss: negative log-likelihood of true parameters under the predicted mixture.

Reference: Bishop (1994) "Mixture Density Networks".
Julia port follows the Lux.jl patterns used in src/diffusion/score_net.jl.
"""

# ─── Builder ──────────────────────────────────────────────────────────────────

"""
    build_mdn(; obs_dim, param_dim, n_components=5, hidden_dim=128, depth=3)

Build a Mixture Density Network as a Lux Chain.

Returns a `Lux.Chain` whose output is a single concatenated vector:
    [pi_logits (K); mu_flat (K*D); log_sigma_flat (K*D)]
where K = n_components, D = param_dim.

Use `mdn_forward` to split the raw output into structured (pi, mu, log_sigma).

# Arguments
- `obs_dim`: dimension of observed signal (number of measurements)
- `param_dim`: dimension of tissue parameters
- `n_components`: number of Gaussian mixture components (default 5)
- `hidden_dim`: width of hidden layers (default 128)
- `depth`: number of hidden layers in backbone (default 3)
"""
function build_mdn(;
    obs_dim::Int,
    param_dim::Int,
    n_components::Int = 5,
    hidden_dim::Int = 128,
    depth::Int = 3,
)
    output_dim = n_components + 2 * n_components * param_dim

    # Build backbone layers: input -> hidden (with relu), repeated `depth` times, then output
    layers = []
    push!(layers, Dense(obs_dim => hidden_dim, relu))
    for _ in 2:depth
        push!(layers, Dense(hidden_dim => hidden_dim, relu))
    end
    push!(layers, Dense(hidden_dim => output_dim))

    return Chain(layers...)
end

# ─── Forward pass (split raw output) ─────────────────────────────────────────

"""
    mdn_forward(model, ps, st, x; n_components, param_dim)

Run the MDN forward pass and split the concatenated output into structured
mixture parameters.

# Arguments
- `model`: Lux Chain built by `build_mdn`
- `ps`: model parameters
- `st`: model state
- `x`: input signals, shape (obs_dim, batch) or (obs_dim, 1)
- `n_components`: K, number of mixture components
- `param_dim`: D, dimension of parameter space

# Returns
- `pi`: softmax mixture weights, shape (K, batch)
- `mu`: component means, shape (D, K, batch)
- `log_sigma`: component log-std-devs, shape (D, K, batch)
- `st_new`: updated model state
"""
function mdn_forward(model, ps, st, x::AbstractMatrix;
    n_components::Int, param_dim::Int,
)
    raw, st_new = model(x, ps, st)
    # raw shape: (output_dim, batch)
    batch = size(x, 2)
    K = n_components
    D = param_dim

    # Split: [pi_logits (K); mu_flat (K*D); log_sigma_flat (K*D)]
    pi_logits = raw[1:K, :]                           # (K, batch)
    mu_flat = raw[K+1:K+K*D, :]                       # (K*D, batch)
    log_sigma_flat = raw[K+K*D+1:K+2*K*D, :]          # (K*D, batch)

    # Softmax over components for each sample in batch
    # Numerically stable softmax
    pi_shifted = pi_logits .- maximum(pi_logits, dims=1)
    pi_exp = exp.(pi_shifted)
    pi_weights = pi_exp ./ sum(pi_exp, dims=1)  # (K, batch)

    # Reshape mu and log_sigma to (D, K, batch)
    mu = reshape(mu_flat, D, K, batch)
    log_sigma = reshape(log_sigma_flat, D, K, batch)

    return pi_weights, mu, log_sigma, st_new
end

# ─── Loss: negative log-likelihood ───────────────────────────────────────────

"""
    mdn_loss(model, ps, st, params, signals; n_components, param_dim)

Compute the mean negative log-likelihood of `params` under the Gaussian
mixture predicted by the MDN from `signals`.

# Arguments
- `params`: true parameter vectors, shape (D, batch)
- `signals`: observed signal vectors, shape (obs_dim, batch)

# Returns
- `(loss, st_new)`: scalar NLL averaged over batch, and updated state
"""
function mdn_loss(model, ps, st, params::AbstractMatrix, signals::AbstractMatrix;
    n_components::Int, param_dim::Int,
)
    pi_w, mu, log_sigma, st_new = mdn_forward(model, ps, st, signals;
        n_components = n_components, param_dim = param_dim)

    # pi_w: (K, batch), mu: (D, K, batch), log_sigma: (D, K, batch)
    # params: (D, batch)
    K = n_components
    D = param_dim
    batch = size(params, 2)

    # Clamp log_sigma for numerical stability
    log_sigma_c = clamp.(log_sigma, -10.0f0, 5.0f0)

    # Expand params to (D, K, batch) for broadcasting
    # params_expanded[:, k, b] = params[:, b] for all k
    params_expanded = reshape(params, D, 1, batch)  # (D, 1, batch) -- broadcasts over K

    # Gaussian log-prob per component:
    # log N(y | mu_k, sigma_k) = -0.5 * sum_d [(y_d - mu_kd)^2 / sigma_kd^2]
    #                           - sum_d [log_sigma_kd] - 0.5 * D * log(2pi)
    diff_sq = (params_expanded .- mu).^2  # (D, K, batch)
    var = exp.(2 .* log_sigma_c) .+ 1.0f-8
    mahalanobis = sum(diff_sq ./ var, dims=1)  # (1, K, batch)
    log_det = sum(log_sigma_c, dims=1)          # (1, K, batch)
    constant = 0.5f0 * D * log(2.0f0 * Float32(pi))

    log_prob_k = -0.5f0 .* mahalanobis .- log_det .- constant  # (1, K, batch)
    log_prob_k = dropdims(log_prob_k, dims=1)  # (K, batch)

    # Combine with mixture weights: log p(y|x) = logsumexp_k(log pi_k + log_prob_k)
    log_pi = log.(pi_w .+ 1.0f-8)  # (K, batch)
    log_joint = log_pi .+ log_prob_k  # (K, batch)

    # logsumexp over K for each batch element
    max_log = maximum(log_joint, dims=1)  # (1, batch)
    log_prob = max_log .+ log.(sum(exp.(log_joint .- max_log), dims=1))  # (1, batch)

    # Mean negative log-likelihood
    nll = -mean(log_prob)

    return nll, st_new
end

# ─── Training loop ────────────────────────────────────────────────────────────

"""
    train_mdn!(model, ps, st, params, signals, rng;
               n_epochs=100, batch_size=64, lr=1e-3, n_components, param_dim)

Train the MDN with mini-batch SGD (Adam optimizer).

# Arguments
- `params`: training parameter vectors, shape (D, N)
- `signals`: training signal vectors, shape (obs_dim, N)
- `rng`: random number generator for shuffling

# Returns
- `(ps, st, losses)`: updated parameters, state, and per-epoch mean losses
"""
function train_mdn!(model, ps, st, params::AbstractMatrix, signals::AbstractMatrix,
    rng::AbstractRNG;
    n_epochs::Int = 100,
    batch_size::Int = 64,
    lr::Real = 1e-3,
    n_components::Int,
    param_dim::Int,
)
    N = size(params, 2)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
    losses = Float64[]

    for epoch in 1:n_epochs
        # Shuffle indices
        perm = randperm(rng, N)
        epoch_loss = 0.0
        n_batches = 0

        for start_idx in 1:batch_size:N
            end_idx = min(start_idx + batch_size - 1, N)
            idx = perm[start_idx:end_idx]

            batch_params = params[:, idx]
            batch_signals = signals[:, idx]

            (loss, st), grads = Zygote.withgradient(ps) do p
                mdn_loss(model, p, st, batch_params, batch_signals;
                    n_components = n_components, param_dim = param_dim)
            end

            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            epoch_loss += loss
            n_batches += 1
        end

        push!(losses, epoch_loss / n_batches)
    end

    return ps, st, losses
end

# ─── Sampling ─────────────────────────────────────────────────────────────────

"""
    sample_mdn(model, ps, st, signal, rng;
               n_samples=1000, n_components, param_dim)

Draw posterior samples from the MDN for a single observed signal.

# Arguments
- `signal`: observed signal vector, shape (obs_dim, 1)
- `rng`: random number generator
- `n_samples`: number of posterior samples to draw

# Returns
- `samples`: matrix of shape (param_dim, n_samples)
"""
function sample_mdn(model, ps, st, signal::AbstractMatrix, rng::AbstractRNG;
    n_samples::Int = 1000,
    n_components::Int,
    param_dim::Int,
)
    pi_w, mu, log_sigma, _ = mdn_forward(model, ps, st, signal;
        n_components = n_components, param_dim = param_dim)

    # For single observation: pi_w (K, 1), mu (D, K, 1), log_sigma (D, K, 1)
    K = n_components
    D = param_dim
    pi_vec = pi_w[:, 1]            # (K,)
    mu_mat = mu[:, :, 1]           # (D, K)
    sigma_mat = exp.(log_sigma[:, :, 1])  # (D, K)

    # Sample component indices from categorical distribution
    # Use cumulative sum trick for categorical sampling
    samples = zeros(Float32, D, n_samples)
    cumprobs = cumsum(pi_vec)

    for i in 1:n_samples
        # Select component
        u = rand(rng, Float32)
        k = 1
        while k < K && u > cumprobs[k]
            k += 1
        end

        # Sample from selected Gaussian component
        eps = randn(rng, Float32, D)
        samples[:, i] = mu_mat[:, k] .+ sigma_mat[:, k] .* eps
    end

    return samples
end
