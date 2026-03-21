"""
DDPM sampling for the score-based posterior.
"""

using LinearAlgebra

function sample_posterior(
    model, ps, st, signal::AbstractVector;
    schedule::VPSchedule = VPSchedule(),
    n_samples::Int = 500,
    n_steps::Int = 500,
    n_scalars::Int = 4,
    n_vectors::Int = 2,
    prediction::Symbol = :eps,
)
    rng = Random.default_rng()
    param_dim = n_scalars + n_vectors * 3

    # Start from noise
    theta_t = randn(rng, Float32, param_dim, n_samples)

    # Discrete timesteps
    timesteps = range(1.0f0, 1f-4, length=n_steps)

    for (idx, t) in enumerate(timesteps)
        t_prev = idx < n_steps ? timesteps[idx + 1] : 0.0f0

        ab_t = alpha_bar(schedule, t)
        ab_prev = alpha_bar(schedule, t_prev)
        sqrt_ab_t = sqrt(ab_t)
        sqrt_1m_ab_t = sqrt(max(1.0f0 - ab_t, 0.0f0))

        # Predict for each sample
        for j in 1:n_samples
            net_out, st = score_forward(model, ps, st,
                                        @view(theta_t[:, j]), t, signal)

            if prediction == :v
                pred_x0 = sqrt_ab_t .* theta_t[:, j] .- sqrt_1m_ab_t .* net_out
                eps_pred = (theta_t[:, j] .- sqrt_ab_t .* pred_x0) ./
                           max(sqrt_1m_ab_t, 1f-8)
            else
                eps_pred = net_out
                pred_x0 = (theta_t[:, j] .- sqrt_1m_ab_t .* eps_pred) ./
                           max(sqrt_ab_t, 1f-8)
            end

            sqrt_ab_prev = sqrt(ab_prev)
            sqrt_1m_ab_prev = sqrt(max(1.0f0 - ab_prev, 0.0f0))

            mean = sqrt_ab_prev .* pred_x0 .+ sqrt_1m_ab_prev .* eps_pred

            # Posterior variance
            beta_t = 1.0f0 - ab_t / max(ab_prev, 1f-8)
            sigma_t = sqrt(clamp(beta_t, 0.0f0, 1.0f0))

            noise = idx < n_steps ? randn(rng, Float32, param_dim) : zeros(Float32, param_dim)
            theta_t[:, j] = mean .+ sigma_t .* noise
        end
    end

    # Normalise orientation vectors back to unit sphere
    for v in 1:n_vectors
        start = n_scalars + (v - 1) * 3 + 1
        stop = start + 2
        for j in 1:n_samples
            vec = @view theta_t[start:stop, j]
            n = norm(vec)
            if n > 1e-8
                vec ./= n
            end
        end
    end

    return theta_t  # (param_dim, n_samples)
end
