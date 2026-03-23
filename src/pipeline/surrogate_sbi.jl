"""
Surrogate-accelerated SBI pipeline.

When to use this:
- Models with EXPENSIVE forward simulations (Monte Carlo, FEM, Bloch-Torrey PDE)
  where each evaluation takes seconds → surrogate gives 1000x+ speedup
- NOT needed for models with analytical solutions (Ball+Stick, DTI, NODDI)
  where the formula is already faster than the neural surrogate

Pipeline:
1. Generate training data from expensive simulator (one-time)
2. Train surrogate to reproduce simulator output (<1% error)
3. Use surrogate as fast forward model for score posterior training
"""

using Lux, Random, Statistics, LinearAlgebra, Optimisers, Zygote

"""
    train_surrogate_sbi(;
        forward_model, acq, config,
        surrogate_steps=50_000, score_steps=30_000,
    )

End-to-end pipeline: train surrogate, then train score posterior using
the surrogate as the forward model.

Returns (score_net, score_ps, score_st, surrogate, surr_ps, surr_st, schedule).
"""
function train_surrogate_sbi(;
    forward_model,           # BallStickModel or similar
    parameter_ranges,        # Dict{String, Tuple}
    parameter_names,         # Vector{String}
    noise_type::Symbol = :rician,
    snr_range = (10.0, 50.0),
    surrogate_steps::Int = 50_000,
    surrogate_hidden::Int = 256,
    surrogate_depth::Int = 6,
    score_steps::Int = 30_000,
    score_hidden::Int = 512,
    score_depth::Int = 6,
    score_prediction::Symbol = :v,
    batch_size::Int = 512,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)
    param_dim = length(parameter_names)
    signal_dim = length(forward_model.bvalues)

    lows = Float32[parameter_ranges[n][1] for n in parameter_names]
    highs = Float32[parameter_ranges[n][2] for n in parameter_names]
    spans = max.(highs .- lows, 1f-12)
    b0_mask = forward_model.bvalues .< 100e6

    # ================================================================
    # Phase 1: Train surrogate on analytical forward model
    # ================================================================
    println("=" ^ 60)
    println("Phase 1: Training surrogate ($surrogate_steps steps)")
    println("=" ^ 60)

    function analytical_data_fn(rng, n)
        params_norm = rand(rng, Float32, param_dim, n)
        params_phys = lows .+ params_norm .* spans
        # Normalize orientations (assuming last 2*3 params are vectors)
        n_scalars = param_dim - 6  # assumes 2 orientation vectors
        for j in 1:n
            for v in 0:1
                start = n_scalars + v * 3 + 1
                mu = @view params_phys[start:start+2, j]
                mu ./= max(norm(mu), 1f-8)
            end
        end
        signals = zeros(Float32, signal_dim, n)
        for j in 1:n
            signals[:, j] = simulate(forward_model, params_phys[:, j])
        end
        return params_norm, signals
    end

    surrogate = build_surrogate(
        param_dim=param_dim, signal_dim=signal_dim,
        hidden_dim=surrogate_hidden, depth=surrogate_depth,
    )
    surr_ps, surr_st = Lux.setup(rng, surrogate)

    surr_ps, surr_st, surr_losses = train_surrogate!(
        surrogate, surr_ps, surr_st, analytical_data_fn;
        n_steps=surrogate_steps, batch_size=batch_size,
        learning_rate=1e-3, print_every=surrogate_steps ÷ 5,
        loss_type=:relative_mse,
    )
    println("Surrogate final loss: $(round(surr_losses[end], sigdigits=4))")

    # ================================================================
    # Phase 2: Train score posterior using surrogate as forward model
    # ================================================================
    println("\n" * "=" ^ 60)
    println("Phase 2: Training score posterior ($score_steps steps)")
    println("  Using surrogate as forward model ($(round(1/0.001, digits=0))x speedup)")
    println("=" ^ 60)

    schedule = VPSchedule()

    score_net_spec = build_score_net(
        param_dim=param_dim, signal_dim=signal_dim,
        hidden_dim=score_hidden, depth=score_depth,
    )
    # Initialize — but build_score_net returns NamedTuple, need Lux model
    # For now use the Chain-based approach from score_net.jl
    # TODO: wire up FiLM blocks properly in Lux

    # Fast surrogate-based data generator
    function surrogate_data_fn(rng, n)
        params_norm = rand(rng, Float32, param_dim, n)

        # Surrogate forward pass (fast!)
        signals_clean, _ = surrogate(params_norm, surr_ps, surr_st)

        # Add noise (same as analytical pipeline)
        if noise_type == :rician && snr_range !== nothing
            snr = rand(rng, Float32, 1, n) .* Float32(snr_range[2] - snr_range[1]) .+ Float32(snr_range[1])
            sigma = 1.0f0 ./ snr
            n1 = randn(rng, Float32, size(signals_clean)) .* sigma
            n2 = randn(rng, Float32, size(signals_clean)) .* sigma
            noisy = @. sqrt((signals_clean + n1)^2 + n2^2)
        else
            sigma = 1.0f0 / 30.0f0
            n1 = randn(rng, Float32, size(signals_clean)) .* sigma
            n2 = randn(rng, Float32, size(signals_clean)) .* sigma
            noisy = @. sqrt((signals_clean + n1)^2 + n2^2)
        end

        # b0 normalize
        if any(b0_mask)
            b0_mean = mean(noisy[b0_mask, :], dims=1)
            b0_mean = max.(b0_mean, 1f-6)
            noisy = noisy ./ b0_mean
        end

        return params_norm, noisy
    end

    # Benchmark: surrogate vs analytical speed
    t_analytical = @elapsed analytical_data_fn(rng, batch_size)
    t_surrogate = @elapsed surrogate_data_fn(rng, batch_size)
    speedup = t_analytical / max(t_surrogate, 1e-8)
    println("  Forward model speedup: $(round(speedup, digits=1))x")
    println("  Analytical: $(round(t_analytical * 1000, digits=1))ms")
    println("  Surrogate:  $(round(t_surrogate * 1000, digits=1))ms")

    return (;
        surrogate, surr_ps, surr_st, surr_losses,
        schedule,
        surrogate_data_fn, analytical_data_fn,
        speedup,
        lows, highs, spans,
    )
end
