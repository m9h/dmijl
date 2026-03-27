"""
Variational Inference (VI) for amortized approximate posterior inference.

Maps observed dMRI signals to variational parameters (mu, log_sigma) of a
mean-field Gaussian posterior q(theta|y) = N(mu(y), diag(exp(2*log_sigma(y)))).

Training minimizes the negative ELBO:
  -ELBO = -E_q[log p(y|theta)] + KL(q || prior)

With uniform (improper) prior, KL reduces to negative entropy of q.
Reparameterization trick: z = mu + exp(log_sigma) * epsilon, epsilon ~ N(0,1).

Two reconstruction modes:
  1. Signal-space (default): forward_fn must be Zygote-differentiable.
     Reconstruction = E_q[log p(y | theta)] using Gaussian log-likelihood on
     signal residuals.
  2. Parameter-space (use_parameter_loss=true): forward_fn is not differentiated.
     Reconstruction = Gaussian log-likelihood of true_params under q(theta|y).
     Use this mode for non-differentiable physics simulators.

Architecture:
  signal (obs_dim,) --> MLP backbone --> [mu(param_dim); log_sigma(param_dim)]

Reference: Kingma & Welling (2014) "Auto-Encoding Variational Bayes".
Julia port follows the Lux.jl patterns used in src/inference/mdn.jl.
"""

# ─── Builder ──────────────────────────────────────────────────────────────────

"""
    build_vi_net(; obs_dim, param_dim, hidden_dim=128, depth=3)

Build a Variational Inference network as a Lux Chain.

Returns a `Lux.Chain` whose output is a single concatenated vector:
    [mu (param_dim); log_sigma (param_dim)]

Use `vi_forward` to split the raw output into structured (mu, log_sigma).

# Arguments
- `obs_dim`: dimension of observed signal (number of measurements)
- `param_dim`: dimension of tissue parameters
- `hidden_dim`: width of hidden layers (default 128)
- `depth`: number of hidden layers in backbone (default 3)
"""
function build_vi_net(;
    obs_dim::Int,
    param_dim::Int,
    hidden_dim::Int = 128,
    depth::Int = 3,
)
    output_dim = 2 * param_dim

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
    vi_forward(model, ps, st, x; param_dim)

Run the VI network forward pass and split the concatenated output into
mu and log_sigma.

# Arguments
- `model`: Lux Chain built by `build_vi_net`
- `ps`: model parameters
- `st`: model state
- `x`: input signals, shape (obs_dim, batch)
- `param_dim`: D, dimension of parameter space

# Returns
- `mu`: posterior means, shape (D, batch)
- `log_sigma`: posterior log-std-devs, shape (D, batch)
- `st_new`: updated model state
"""
function vi_forward(model, ps, st, x::AbstractMatrix; param_dim::Int)
    raw, st_new = model(x, ps, st)
    # raw shape: (2 * param_dim, batch)
    D = param_dim

    mu = raw[1:D, :]              # (D, batch)
    log_sigma = raw[D+1:2*D, :]   # (D, batch)

    return mu, log_sigma, st_new
end

# ─── Loss: negative ELBO ─────────────────────────────────────────────────────

"""
    elbo_loss(model, ps, st, true_params, signals, forward_fn;
              sigma_noise=0.1f0, n_mc_samples=8, use_parameter_loss=false)

Compute the negative ELBO for the variational posterior.

ELBO = E_q[log p(data|theta)] - KL(q||prior)

With uniform prior, KL(q||prior) = -H[q] (negative entropy), so:
  ELBO = E_q[log p(data|theta)] + H[q]
  -ELBO = -E_q[log p(data|theta)] - H[q]

Two modes for the reconstruction term:

**Signal-space mode** (default, `use_parameter_loss=false`):
Uses the reparameterization trick with a Zygote-differentiable `forward_fn`.
Reconstruction = Gaussian log-likelihood of signal residuals.

**Parameter-space mode** (`use_parameter_loss=true`):
For non-differentiable forward models (e.g. physics simulators). Computes
the Gaussian log-likelihood of `true_params` under the predicted posterior
q(theta|y) = N(mu, diag(sigma^2)). No MC sampling or forward_fn needed.

# Arguments
- `model`: Lux Chain built by `build_vi_net`
- `ps`: model parameters
- `st`: model state
- `true_params`: true parameter vectors, shape (D, batch)
- `signals`: observed signal vectors, shape (obs_dim, batch)
- `forward_fn`: function theta -> predicted_signals, accepts (D, batch) -> (obs_dim, batch).
  Only used when `use_parameter_loss=false`.
- `sigma_noise`: noise standard deviation for Gaussian likelihood
- `n_mc_samples`: number of Monte Carlo samples for ELBO estimation (signal-space mode only)
- `use_parameter_loss`: if true, use parameter-space loss (no forward_fn differentiation)

# Returns
- `(loss, st_new)`: scalar negative ELBO averaged over batch, and updated state
"""
function elbo_loss(model, ps, st, true_params::AbstractMatrix, signals::AbstractMatrix,
    forward_fn;
    sigma_noise::Real = 0.1f0,
    n_mc_samples::Int = 8,
    use_parameter_loss::Bool = false,
)
    mu, log_sigma, st_new = vi_forward(model, ps, st, signals; param_dim = size(true_params, 1))

    D = size(mu, 1)
    batch = size(mu, 2)

    # Clamp log_sigma for numerical stability
    log_sigma_c = clamp.(log_sigma, -10.0f0, 5.0f0)
    sigma = exp.(log_sigma_c)

    # --- Entropy of mean-field Gaussian ---
    # H[q] = 0.5 * D * (1 + log(2pi)) + sum(log_sigma)
    # Per-sample entropy, averaged over batch
    entropy = 0.5f0 * D * (1.0f0 + log(2.0f0 * Float32(pi))) .+ sum(log_sigma_c, dims=1)  # (1, batch)
    mean_entropy = mean(entropy)

    if use_parameter_loss
        # --- Parameter-space reconstruction (no forward_fn differentiation) ---
        # log q(true_params | mu, sigma) = -0.5 * sum((true_params - mu)^2 / sigma^2) - sum(log_sigma) - 0.5*D*log(2pi)
        diff_sq = (true_params .- mu).^2
        var = sigma.^2 .+ 1.0f-8
        log_lik = -0.5f0 .* sum(diff_sq ./ var, dims=1) .-
                   sum(log_sigma_c, dims=1) .-
                   0.5f0 * D * log(2.0f0 * Float32(pi))  # (1, batch)
        mean_recon = mean(log_lik)
    else
        # --- Signal-space reconstruction via MC sampling (reparameterization trick) ---
        total_recon = 0.0f0
        for _ in 1:n_mc_samples
            # epsilon ~ N(0, I)
            eps = randn(Float32, D, batch)
            # z = mu + sigma * epsilon
            z = mu .+ sigma .* eps

            # Predicted signals from sampled parameters
            predicted = forward_fn(z)  # (obs_dim, batch)

            # Gaussian log-likelihood: sum over obs_dim, mean over batch
            residuals = signals .- predicted
            log_lik = -0.5f0 .* sum(residuals.^2, dims=1) ./ (sigma_noise^2) .-
                       0.5f0 * size(signals, 1) * log(2.0f0 * Float32(pi) * sigma_noise^2)
            total_recon += mean(log_lik)
        end
        mean_recon = total_recon / n_mc_samples
    end

    # Negative ELBO = -(reconstruction + entropy)
    neg_elbo = -(mean_recon + mean_entropy)

    return neg_elbo, st_new
end

# ─── Training loop ────────────────────────────────────────────────────────────

"""
    train_vi!(model, ps, st, params, signals, rng, forward_fn;
              n_epochs=100, batch_size=64, lr=1e-3,
              sigma_noise=0.1f0, n_mc_samples=8, use_parameter_loss=false)

Train the VI network with mini-batch SGD (Adam optimizer).

# Arguments
- `params`: training parameter vectors, shape (D, N)
- `signals`: training signal vectors, shape (obs_dim, N)
- `rng`: random number generator for shuffling
- `forward_fn`: function theta -> predicted_signals
- `use_parameter_loss`: if true, use parameter-space loss (for non-differentiable forward models)

# Returns
- `(ps, st, losses)`: updated parameters, state, and per-epoch mean losses
"""
function train_vi!(model, ps, st, params::AbstractMatrix, signals::AbstractMatrix,
    rng::AbstractRNG, forward_fn;
    n_epochs::Int = 100,
    batch_size::Int = 64,
    lr::Real = 1e-3,
    sigma_noise::Real = 0.1f0,
    n_mc_samples::Int = 8,
    use_parameter_loss::Bool = false,
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
                elbo_loss(model, p, st, batch_params, batch_signals, forward_fn;
                    sigma_noise = sigma_noise, n_mc_samples = n_mc_samples,
                    use_parameter_loss = use_parameter_loss)
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
    sample_vi(model, ps, st, signal, rng;
              n_samples=1000, param_dim)

Draw posterior samples from the VI network for a single observed signal.

Uses the reparameterization trick: z = mu + exp(log_sigma) * epsilon.

# Arguments
- `signal`: observed signal vector, shape (obs_dim, 1)
- `rng`: random number generator
- `n_samples`: number of posterior samples to draw
- `param_dim`: dimension of parameter space

# Returns
- `samples`: matrix of shape (param_dim, n_samples)
"""
function sample_vi(model, ps, st, signal::AbstractMatrix, rng::AbstractRNG;
    n_samples::Int = 1000,
    param_dim::Int,
)
    mu, log_sigma, _ = vi_forward(model, ps, st, signal; param_dim = param_dim)

    # For single observation: mu (D, 1), log_sigma (D, 1)
    D = param_dim
    mu_vec = mu[:, 1]             # (D,)
    sigma_vec = exp.(log_sigma[:, 1])  # (D,)

    # Sample: z = mu + sigma * epsilon
    samples = zeros(Float32, D, n_samples)
    for i in 1:n_samples
        eps = randn(rng, Float32, D)
        samples[:, i] = mu_vec .+ sigma_vec .* eps
    end

    return samples
end
