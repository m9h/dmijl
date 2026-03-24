"""
Reverse SDE sampler using DifferentialEquations.jl.

Replaces the hand-rolled DDPM loop with native SDE solvers
that support adaptive stepping and error control.

This is Julia's key advantage: the reverse SDE is a first-class
mathematical object, not a hand-coded loop.
"""

using DifferentialEquations, LinearAlgebra, Random

"""
    sample_posterior_diffeq(
        score_fn, signal, schedule;
        n_samples, n_scalars, n_vectors, solver, dt, prediction
    )

Draw posterior samples by solving the reverse-time SDE:

    dθ = [-½β(t)θ - β(t)·score(θ,t,signal)] dt + √β(t) dW̄

using DifferentialEquations.jl solvers.

`score_fn(theta, t, signal)` should return the noise prediction
(or v-prediction, depending on `prediction`).
"""
function sample_posterior_diffeq(
    score_fn,                  # (theta, t, signal) -> prediction
    signal::AbstractVector,
    schedule::VPSchedule;
    n_samples::Int = 500,
    n_scalars::Int = 4,
    n_vectors::Int = 2,
    solver = EM(),             # Euler-Maruyama (default), or SOSRI(), SOSRA()
    dt::Float64 = 0.002,      # step size for fixed-step solvers
    prediction::Symbol = :eps,
)
    param_dim = n_scalars + n_vectors * 3
    rng = Random.default_rng()

    # Solve one SDE per sample (could batch, but SDE solvers are per-trajectory)
    all_samples = zeros(Float32, param_dim, n_samples)

    for j in 1:n_samples
        theta_T = randn(rng, Float64, param_dim)

        # Reverse-time SDE drift
        function drift!(du, u, p, t)
            # Reverse time: we solve from t=1 to t=0
            # In forward time coordinates: τ = 1 - t
            tau = 1.0 - t

            # Get score from noise prediction
            net_out = score_fn(Float32.(u), Float32(tau), signal)

            ab = alpha_bar(schedule, tau)
            sqrt_ab = sqrt(ab)
            sqrt_1m_ab = sqrt(max(1.0 - ab, 0.0))

            if prediction == :v
                pred_x0 = sqrt_ab .* u .- sqrt_1m_ab .* Float64.(net_out)
                eps_pred = (u .- sqrt_ab .* pred_x0) ./ max(sqrt_1m_ab, 1e-8)
            else
                eps_pred = Float64.(net_out)
            end

            score = -eps_pred ./ max(sqrt_1m_ab, 1e-8)

            beta_t = beta(schedule, tau)
            du .= -0.5 .* beta_t .* u .- beta_t .* score
        end

        # Reverse-time SDE diffusion
        function diffusion!(du, u, p, t)
            tau = 1.0 - t
            beta_t = beta(schedule, tau)
            du .= sqrt(beta_t)
        end

        # Solve from t=0 (which is τ=1, pure noise) to t=1 (which is τ=0, clean)
        tspan = (0.0, 1.0 - 1e-4)
        prob = SDEProblem(drift!, diffusion!, theta_T, tspan)
        sol = solve(prob, solver; dt=dt, save_everystep=false)

        all_samples[:, j] = Float32.(sol.u[end])
    end

    # Normalize orientations to unit sphere
    for v in 1:n_vectors
        start = n_scalars + (v - 1) * 3 + 1
        stop = start + 2
        for j in 1:n_samples
            vec = @view all_samples[start:stop, j]
            n = norm(vec)
            if n > 1e-8
                vec ./= n
            end
        end
    end

    return all_samples
end

"""
    sample_posterior_ode(score_fn, signal, schedule; ...)

Probability flow ODE (deterministic) using DifferentialEquations.jl.
No stochasticity — useful for MAP estimation and likelihood computation.
"""
function sample_posterior_ode(
    score_fn,
    signal::AbstractVector,
    schedule::VPSchedule;
    n_samples::Int = 500,
    n_scalars::Int = 4,
    n_vectors::Int = 2,
    solver = Tsit5(),          # adaptive RK solver
    dt::Float64 = 0.002,
    prediction::Symbol = :eps,
)
    param_dim = n_scalars + n_vectors * 3
    rng = Random.default_rng()

    all_samples = zeros(Float32, param_dim, n_samples)

    for j in 1:n_samples
        theta_T = randn(rng, Float64, param_dim)

        function ode_drift!(du, u, p, t)
            tau = 1.0 - t
            net_out = score_fn(Float32.(u), Float32(tau), signal)

            ab = alpha_bar(schedule, tau)
            sqrt_ab = sqrt(ab)
            sqrt_1m_ab = sqrt(max(1.0 - ab, 0.0))

            if prediction == :v
                pred_x0 = sqrt_ab .* u .- sqrt_1m_ab .* Float64.(net_out)
                eps_pred = (u .- sqrt_ab .* pred_x0) ./ max(sqrt_1m_ab, 1e-8)
            else
                eps_pred = Float64.(net_out)
            end

            score = -eps_pred ./ max(sqrt_1m_ab, 1e-8)

            beta_t = beta(schedule, tau)
            # Probability flow ODE: drift = f(x,t) - ½g²(t)score
            du .= -0.5 .* beta_t .* (u .+ score)
        end

        tspan = (0.0, 1.0 - 1e-4)
        prob = ODEProblem(ode_drift!, theta_T, tspan)
        sol = solve(prob, solver; saveat=[tspan[2]])

        all_samples[:, j] = Float32.(sol.u[end])
    end

    # Normalize orientations
    for v in 1:n_vectors
        start = n_scalars + (v - 1) * 3 + 1
        stop = start + 2
        for j in 1:n_samples
            vec = @view all_samples[start:stop, j]
            n = norm(vec)
            if n > 1e-8
                vec ./= n
            end
        end
    end

    return all_samples
end
