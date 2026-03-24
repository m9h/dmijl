"""
Tests for the FiLM-conditioned ScoreNetwork.

Tests forward pass, Zygote gradients, score_forward wrapper, and
individual components (SinusoidalEmbedding, FiLMLayer).
All tests use small networks and tiny batch sizes for speed.
"""

using Test, Random, Statistics, LinearAlgebra
using Lux, Zygote, Optimisers

# Include source directly (before package is registered)
include("../src/diffusion/schedule.jl")
include("../src/diffusion/score_net.jl")

@testset "FiLM Network" begin

    # ---- Small network dimensions for fast tests ----
    param_dim = 4
    signal_dim = 8
    hidden_dim = 16
    depth = 3
    cond_dim = 16
    batch_size = 4

    rng = MersenneTwister(42)

    @testset "SinusoidalEmbedding" begin
        emb = SinusoidalEmbedding(16)
        ps_emb, st_emb = Lux.setup(rng, emb)

        # Single timestep
        t_single = reshape(Float32[0.5], 1, 1)
        out, st_out = emb(t_single, ps_emb, st_emb)
        @test size(out) == (16, 1)

        # Batch of timesteps
        t_batch = rand(rng, Float32, 1, batch_size)
        out_batch, _ = emb(t_batch, ps_emb, st_emb)
        @test size(out_batch) == (16, batch_size)

        # t=0 and t=1 produce different embeddings
        t0 = reshape(Float32[0.0], 1, 1)
        t1 = reshape(Float32[1.0], 1, 1)
        out0, _ = emb(t0, ps_emb, st_emb)
        out1, _ = emb(t1, ps_emb, st_emb)
        @test out0 != out1

        # No trainable parameters
        @test isempty(Lux.initialparameters(rng, emb))
    end

    @testset "FiLMLayer forward pass" begin
        dense = Dense(hidden_dim => hidden_dim)
        film = FiLMLayer(dense)
        ps_film, st_film = Lux.setup(rng, film)

        x = randn(rng, Float32, hidden_dim, batch_size)
        gamma = randn(rng, Float32, hidden_dim, batch_size)
        beta = randn(rng, Float32, hidden_dim, batch_size)

        out, st_out = film((x, gamma, beta), ps_film, st_film)
        @test size(out) == (hidden_dim, batch_size)

        # Zero gamma and beta => just gelu(Wx + b)
        gamma_zero = zeros(Float32, hidden_dim, batch_size)
        beta_zero = zeros(Float32, hidden_dim, batch_size)
        out_no_mod, _ = film((x, gamma_zero, beta_zero), ps_film, st_film)
        # With gamma=0 and beta=0: (1 + 0) .* gelu(Wx+b) .+ 0 = gelu(Wx+b)
        h_dense, _ = dense(x, ps_film.dense, st_film.dense)
        expected = gelu.(h_dense)
        @test out_no_mod ≈ expected atol=1e-6
    end

    @testset "build_score_net returns ScoreNetwork" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        @test model isa ScoreNetwork
        @test model.n_layers == depth - 1
    end

    @testset "ScoreNetwork forward pass" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(rng, model)

        theta_t = randn(rng, Float32, param_dim, batch_size)
        t = rand(rng, Float32, 1, batch_size)
        signal = randn(rng, Float32, signal_dim, batch_size)

        x = (; theta_t = theta_t, t = t, signal = signal)
        out, st_new = model(x, ps, st)

        # Output shape matches param_dim
        @test size(out) == (param_dim, batch_size)
        # Output is finite
        @test all(isfinite, out)
    end

    @testset "ScoreNetwork single-sample forward pass" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(rng, model)

        # Single sample: (param_dim, 1), scalar-ish t, (signal_dim, 1)
        theta_t = randn(rng, Float32, param_dim, 1)
        t_val = Float32(0.5)
        signal = randn(rng, Float32, signal_dim, 1)

        x = (; theta_t = theta_t, t = t_val, signal = signal)
        out, _ = model(x, ps, st)
        @test size(out) == (param_dim, 1)
        @test all(isfinite, out)
    end

    @testset "score_forward wrapper" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(rng, model)

        theta_t = randn(rng, Float32, param_dim, batch_size)
        t = rand(rng, Float32, 1, batch_size)
        signal = randn(rng, Float32, signal_dim, batch_size)

        # score_forward should produce same result as direct call
        out_direct, st_direct = model((; theta_t, t, signal), ps, st)
        out_wrapper, st_wrapper = score_forward(model, ps, st, theta_t, t, signal)

        @test out_direct ≈ out_wrapper atol=1e-6
    end

    @testset "Deterministic forward pass" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(rng, model)

        theta_t = randn(rng, Float32, param_dim, batch_size)
        t = rand(rng, Float32, 1, batch_size)
        signal = randn(rng, Float32, signal_dim, batch_size)

        x = (; theta_t = theta_t, t = t, signal = signal)
        out1, _ = model(x, ps, st)
        out2, _ = model(x, ps, st)
        @test out1 ≈ out2 atol=1e-10
    end

    @testset "Zygote gradient through ScoreNetwork" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(rng, model)

        theta_t = randn(rng, Float32, param_dim, batch_size)
        t = rand(rng, Float32, 1, batch_size)
        signal = randn(rng, Float32, signal_dim, batch_size)
        target = randn(rng, Float32, param_dim, batch_size)

        # Gradient computation should succeed without errors
        (loss, _), grads = Zygote.withgradient(ps) do p
            x = (; theta_t = theta_t, t = t, signal = signal)
            pred, new_st = model(x, p, st)
            l = mean((pred .- target).^2)
            return l, new_st
        end

        @test isfinite(loss)
        @test loss > 0.0  # random predictions won't match random target
        # Gradients exist and are the right type
        @test grads[1] !== nothing
    end

    @testset "Gradient step reduces loss" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(MersenneTwister(0), model)

        # Fixed data
        theta_t = randn(MersenneTwister(1), Float32, param_dim, batch_size)
        t = rand(MersenneTwister(2), Float32, 1, batch_size)
        signal = randn(MersenneTwister(3), Float32, signal_dim, batch_size)
        target = zeros(Float32, param_dim, batch_size)  # simple target

        opt_state = Optimisers.setup(Optimisers.Adam(1e-3), ps)

        function compute_loss(ps, st)
            x = (; theta_t = theta_t, t = t, signal = signal)
            pred, new_st = model(x, ps, st)
            return mean((pred .- target).^2), new_st
        end

        loss_before, _ = compute_loss(ps, st)

        # Take 10 gradient steps
        for _ in 1:10
            (loss, st), grads = Zygote.withgradient(ps) do p
                compute_loss(p, st)
            end
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        end

        loss_after, _ = compute_loss(ps, st)
        @test loss_after < loss_before
    end

    @testset "Different signals produce different scores" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(rng, model)

        theta_t = randn(rng, Float32, param_dim, 1)
        t_val = Float32(0.5)

        sig1 = ones(Float32, signal_dim, 1)
        sig2 = zeros(Float32, signal_dim, 1)

        out1, _ = model((; theta_t, t = t_val, signal = sig1), ps, st)
        out2, _ = model((; theta_t, t = t_val, signal = sig2), ps, st)

        @test !(out1 ≈ out2 atol=1e-6)
    end

    @testset "Different timesteps produce different scores" begin
        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            cond_dim = cond_dim,
        )
        ps, st = Lux.setup(rng, model)

        theta_t = randn(rng, Float32, param_dim, 1)
        signal = randn(rng, Float32, signal_dim, 1)

        out_t0, _ = model((; theta_t, t = Float32(0.01), signal = signal), ps, st)
        out_t1, _ = model((; theta_t, t = Float32(0.99), signal = signal), ps, st)

        @test !(out_t0 ≈ out_t1 atol=1e-6)
    end
end
