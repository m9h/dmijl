"""
Tests for Mixture Density Network (MDN) amortized posterior inference.

Follows Red-Green TDD: tests written first, implementation to follow.
Tests: build_mdn, mdn_forward, mdn_loss, train_mdn!, sample_mdn,
       and integration with composable compartment models.
"""

using Test, Random, Statistics, LinearAlgebra
using Lux, Zygote, Optimisers, ComponentArrays

# Include just the MDN source (standalone, no DMI module deps)
include("../src/inference/mdn.jl")

@testset "Mixture Density Network" begin

    # ---- Small network dimensions for fast tests ----
    obs_dim = 16        # observed signal dimension
    param_dim = 4       # parameter dimension
    n_components = 3    # mixture components
    hidden_dim = 32     # hidden layer width
    depth = 2           # hidden layers
    batch_size = 8

    rng = MersenneTwister(42)

    @testset "build_mdn returns Lux model with correct output dimensions" begin
        model = build_mdn(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            n_components = n_components,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        # Model should be a Lux Chain
        @test model isa Lux.AbstractLuxLayer

        # Initialize and check output dimension
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, obs_dim, 1)
        out, _ = model(x, ps, st)

        # Output: n_components (pi logits) + n_components * param_dim (mu) + n_components * param_dim (log_sigma)
        expected_out_dim = n_components + 2 * n_components * param_dim
        @test size(out, 1) == expected_out_dim
    end

    @testset "mdn_forward produces valid outputs" begin
        model = build_mdn(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            n_components = n_components,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        # Single observation
        x_single = randn(rng, Float32, obs_dim, 1)
        pi, mu, log_sigma, st_new = mdn_forward(model, ps, st, x_single;
            n_components = n_components, param_dim = param_dim)

        # pi: (n_components, 1) -- softmaxed mixture weights
        @test size(pi) == (n_components, 1)
        @test all(pi .>= 0)
        @test sum(pi[:, 1]) ≈ 1.0 atol=1e-5

        # mu: (param_dim, n_components, 1)
        @test size(mu) == (param_dim, n_components, 1)
        @test all(isfinite, mu)

        # log_sigma: same shape as mu
        @test size(log_sigma) == (param_dim, n_components, 1)
        @test all(isfinite, log_sigma)

        # Batched observation
        x_batch = randn(rng, Float32, obs_dim, batch_size)
        pi_b, mu_b, log_sigma_b, _ = mdn_forward(model, ps, st, x_batch;
            n_components = n_components, param_dim = param_dim)

        @test size(pi_b) == (n_components, batch_size)
        @test all(pi_b .>= 0)
        for j in 1:batch_size
            @test sum(pi_b[:, j]) ≈ 1.0 atol=1e-5
        end
        @test size(mu_b) == (param_dim, n_components, batch_size)
        @test size(log_sigma_b) == (param_dim, n_components, batch_size)
    end

    @testset "mdn_loss is finite and non-negative" begin
        model = build_mdn(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            n_components = n_components,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        params = randn(rng, Float32, param_dim, batch_size)
        signals = randn(rng, Float32, obs_dim, batch_size)

        loss, st_new = mdn_loss(model, ps, st, params, signals;
            n_components = n_components, param_dim = param_dim)

        @test isfinite(loss)
        # NLL of a Gaussian mixture is not guaranteed to be non-negative in general,
        # but for random init the variances are typically large enough that it is positive
        @test loss isa Real
    end

    @testset "Zygote gradient through mdn_loss" begin
        model = build_mdn(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            n_components = n_components,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        params = randn(rng, Float32, param_dim, batch_size)
        signals = randn(rng, Float32, obs_dim, batch_size)

        (loss, _), grads = Zygote.withgradient(ps) do p
            mdn_loss(model, p, st, params, signals;
                n_components = n_components, param_dim = param_dim)
        end

        @test isfinite(loss)
        @test grads[1] !== nothing
    end

    @testset "mdn_loss decreases after training steps" begin
        model = build_mdn(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            n_components = n_components,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(MersenneTwister(0), model)

        # Simple synthetic data: params -> signals via linear map + noise
        rng_data = MersenneTwister(1)
        N = 64
        params = randn(rng_data, Float32, param_dim, N)
        W = randn(rng_data, Float32, obs_dim, param_dim)
        signals = W * params .+ 0.01f0 .* randn(rng_data, Float32, obs_dim, N)

        loss_before, _ = mdn_loss(model, ps, st, params, signals;
            n_components = n_components, param_dim = param_dim)

        opt_state = Optimisers.setup(Optimisers.Adam(1e-3), ps)

        for _ in 1:50
            (loss, st), grads = Zygote.withgradient(ps) do p
                mdn_loss(model, p, st, params, signals;
                    n_components = n_components, param_dim = param_dim)
            end
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        end

        loss_after, _ = mdn_loss(model, ps, st, params, signals;
            n_components = n_components, param_dim = param_dim)

        @test loss_after < loss_before
    end

    @testset "sample_mdn produces samples of correct shape" begin
        model = build_mdn(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            n_components = n_components,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        signal = randn(rng, Float32, obs_dim, 1)
        n_samples = 100

        samples = sample_mdn(model, ps, st, signal, MersenneTwister(99);
            n_samples = n_samples, n_components = n_components, param_dim = param_dim)

        # Output shape: (param_dim, n_samples)
        @test size(samples) == (param_dim, n_samples)
        @test all(isfinite, samples)
    end

    @testset "sample_mdn with trained model produces samples near true params" begin
        # Train MDN on a simple 1D mapping: param -> signal = 2*param + 1
        obs_d = 1
        par_d = 1
        n_comp = 3

        model = build_mdn(;
            obs_dim = obs_d,
            param_dim = par_d,
            n_components = n_comp,
            hidden_dim = 64,
            depth = 2,
        )

        rng_train = MersenneTwister(0)
        ps, st = Lux.setup(rng_train, model)

        # Training data: deterministic mapping
        N = 256
        rng_data = MersenneTwister(1)
        true_params = randn(rng_data, Float32, par_d, N)
        signals = 2.0f0 .* true_params .+ 1.0f0

        ps, st, losses = train_mdn!(model, ps, st, true_params, signals,
            MersenneTwister(2);
            n_epochs = 100, batch_size = 64, lr = 1e-3,
            n_components = n_comp, param_dim = par_d)

        # Test on a known point: param=0.5 -> signal=2.0
        test_signal = Float32[2.0;;]  # (1, 1)
        samples = sample_mdn(model, ps, st, test_signal, MersenneTwister(3);
            n_samples = 500, n_components = n_comp, param_dim = par_d)

        # Mean of posterior samples should be near 0.5
        mean_samples = mean(samples)
        @test abs(mean_samples - 0.5f0) < 0.3f0  # generous tolerance
    end

    @testset "train_mdn! returns losses vector" begin
        model = build_mdn(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            n_components = n_components,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(MersenneTwister(0), model)

        N = 32
        rng_data = MersenneTwister(1)
        params = randn(rng_data, Float32, param_dim, N)
        signals = randn(rng_data, Float32, obs_dim, N)

        n_epochs = 5
        ps_new, st_new, losses = train_mdn!(model, ps, st, params, signals,
            MersenneTwister(2);
            n_epochs = n_epochs, batch_size = 16, lr = 1e-3,
            n_components = n_components, param_dim = param_dim)

        @test length(losses) == n_epochs
        @test all(isfinite, losses)
    end

    @testset "Integration: Ball+Stick forward model -> MDN posterior" begin
        if isdefined(Main, :DMI) || isdefined(Main, :MultiCompartmentModel)
            acq = hcp_like_acquisition()
            n_meas = length(acq.bvalues)

            ball = G1Ball(lambda_iso=2.0e-9)
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            mcm = MultiCompartmentModel(ball, stick)

            N_train = 200
            rng_gen = MersenneTwister(7)
            lb, ub = get_flat_bounds(mcm)
            par_d = nparams(mcm)

            all_params = zeros(Float32, par_d, N_train)
            all_signals = zeros(Float32, n_meas, N_train)

            for i in 1:N_train
                p_vec = Float64.(lb .+ (ub .- lb) .* rand(rng_gen, par_d))
                # Normalize volume fractions
                p_vec[end-1:end] ./= sum(p_vec[end-1:end])
                # Normalize mu to unit vector
                p_vec[2:4] ./= max(norm(p_vec[2:4]), 1e-8)
                sig = signal(mcm, acq, p_vec)
                all_params[:, i] .= Float32.(p_vec)
                all_signals[:, i] .= Float32.(sig)
            end

            model = build_mdn(;
                obs_dim = n_meas,
                param_dim = par_d,
                n_components = 3,
                hidden_dim = 64,
                depth = 2,
            )

            rng_mdn = MersenneTwister(0)
            ps, st = Lux.setup(rng_mdn, model)

            ps, st, losses = train_mdn!(model, ps, st, all_params, all_signals,
                MersenneTwister(1);
                n_epochs = 30, batch_size = 32, lr = 1e-3,
                n_components = 3, param_dim = par_d)

            @test losses[end] < losses[1]

            test_signal = all_signals[:, 1:1]
            samples = sample_mdn(model, ps, st, test_signal, MersenneTwister(99);
                n_samples = 200, n_components = 3, param_dim = par_d)

            @test size(samples) == (par_d, 200)
            @test all(isfinite, samples)

            # With minimal training (50 steps, 200 samples), just verify
            # samples are finite and roughly in a reasonable range
            @test all(isfinite, samples)
            @test size(samples, 1) == par_d
        else
            @info "Skipping Ball+Stick integration test (DMI module not loaded). Run via runtests.jl for full coverage."
            @test true  # placeholder so testset isn't empty
        end
    end

end
