"""
Tests for Variational Inference (VI) amortized posterior inference.

Follows Red-Green TDD: tests written first, implementation to follow.
Tests: build_vi_net, vi_forward, elbo_loss, train_vi!, sample_vi,
       and integration with composable compartment models.
"""

using Test, Random, Statistics, LinearAlgebra
using Lux, Zygote, Optimisers, ComponentArrays

# Include just the VI source (standalone, no DMI module deps)
include("../src/inference/variational.jl")

@testset "Variational Inference" begin

    # ---- Small network dimensions for fast tests ----
    obs_dim = 16        # observed signal dimension
    param_dim = 4       # parameter dimension
    hidden_dim = 32     # hidden layer width
    depth = 2           # hidden layers
    batch_size = 8

    rng = MersenneTwister(42)

    @testset "build_vi_net returns Lux model with correct output (2 * param_dim)" begin
        model = build_vi_net(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        # Model should be a Lux Chain
        @test model isa Lux.AbstractLuxLayer

        # Initialize and check output dimension
        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, obs_dim, 1)
        out, _ = model(x, ps, st)

        # Output: param_dim (mu) + param_dim (log_sigma) = 2 * param_dim
        @test size(out, 1) == 2 * param_dim
    end

    @testset "vi_forward produces mu and log_sigma of correct shape" begin
        model = build_vi_net(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        # Single observation
        x_single = randn(rng, Float32, obs_dim, 1)
        mu, log_sigma, st_new = vi_forward(model, ps, st, x_single; param_dim = param_dim)

        # mu: (param_dim, 1)
        @test size(mu) == (param_dim, 1)
        @test all(isfinite, mu)

        # log_sigma: (param_dim, 1)
        @test size(log_sigma) == (param_dim, 1)
        @test all(isfinite, log_sigma)

        # Batched observation
        x_batch = randn(rng, Float32, obs_dim, batch_size)
        mu_b, log_sigma_b, _ = vi_forward(model, ps, st, x_batch; param_dim = param_dim)

        @test size(mu_b) == (param_dim, batch_size)
        @test size(log_sigma_b) == (param_dim, batch_size)
        @test all(isfinite, mu_b)
        @test all(isfinite, log_sigma_b)
    end

    @testset "elbo_loss is finite" begin
        model = build_vi_net(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        params = randn(rng, Float32, param_dim, batch_size)
        signals = randn(rng, Float32, obs_dim, batch_size)

        # Simple linear forward model for testing
        W = randn(rng, Float32, obs_dim, param_dim)
        forward_fn = (theta) -> W * theta

        loss, st_new = elbo_loss(model, ps, st, params, signals, forward_fn;
            sigma_noise = 0.1f0, n_mc_samples = 4)

        @test isfinite(loss)
        @test loss isa Real
    end

    @testset "Zygote gradient through elbo_loss works" begin
        model = build_vi_net(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        params = randn(rng, Float32, param_dim, batch_size)
        signals = randn(rng, Float32, obs_dim, batch_size)

        W = randn(rng, Float32, obs_dim, param_dim)
        forward_fn = (theta) -> W * theta

        (loss, _), grads = Zygote.withgradient(ps) do p
            elbo_loss(model, p, st, params, signals, forward_fn;
                sigma_noise = 0.1f0, n_mc_samples = 4)
        end

        @test isfinite(loss)
        @test grads[1] !== nothing
    end

    @testset "train_vi! reduces loss over training" begin
        model = build_vi_net(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(MersenneTwister(0), model)

        # Simple synthetic data: params -> signals via linear map + noise
        rng_data = MersenneTwister(1)
        N = 128
        true_params = randn(rng_data, Float32, param_dim, N)
        W = randn(rng_data, Float32, obs_dim, param_dim)
        signals = W * true_params .+ 0.01f0 .* randn(rng_data, Float32, obs_dim, N)

        forward_fn = (theta) -> W * theta

        ps, st, losses = train_vi!(model, ps, st, true_params, signals,
            MersenneTwister(2), forward_fn;
            n_epochs = 50, batch_size = 32, lr = 1e-3,
            sigma_noise = 0.1f0, n_mc_samples = 4)

        @test length(losses) == 50
        @test all(isfinite, losses)
        # Loss should decrease: compare first few to last few
        @test mean(losses[end-4:end]) < mean(losses[1:5])
    end

    @testset "sample_vi produces correct-shape samples" begin
        model = build_vi_net(;
            obs_dim = obs_dim,
            param_dim = param_dim,
            hidden_dim = hidden_dim,
            depth = depth,
        )
        ps, st = Lux.setup(rng, model)

        signal_in = randn(rng, Float32, obs_dim, 1)
        n_samples = 100

        samples = sample_vi(model, ps, st, signal_in, MersenneTwister(99);
            n_samples = n_samples, param_dim = param_dim)

        # Output shape: (param_dim, n_samples)
        @test size(samples) == (param_dim, n_samples)
        @test all(isfinite, samples)
    end

    @testset "Trained VI on toy problem recovers approximate posterior mean" begin
        # Train VI on a simple 1D mapping: param -> signal = 2*param + 1
        obs_d = 1
        par_d = 1

        model = build_vi_net(;
            obs_dim = obs_d,
            param_dim = par_d,
            hidden_dim = 64,
            depth = 2,
        )

        rng_train = MersenneTwister(0)
        ps, st = Lux.setup(rng_train, model)

        # Training data: deterministic mapping
        N = 256
        rng_data = MersenneTwister(1)
        true_params = randn(rng_data, Float32, par_d, N)
        W_toy = Float32[2.0;;]  # (1, 1)
        b_toy = Float32[1.0;]   # (1,)
        signals = W_toy * true_params .+ b_toy

        forward_fn = (theta) -> W_toy * theta .+ b_toy

        ps, st, losses = train_vi!(model, ps, st, true_params, signals,
            MersenneTwister(2), forward_fn;
            n_epochs = 150, batch_size = 64, lr = 1e-3,
            sigma_noise = 0.01f0, n_mc_samples = 8)

        # Test on a known point: param=0.5 -> signal=2.0
        test_signal = Float32[2.0;;]  # (1, 1)
        samples = sample_vi(model, ps, st, test_signal, MersenneTwister(3);
            n_samples = 500, param_dim = par_d)

        # Mean of posterior samples should be near 0.5
        mean_samples = mean(samples)
        @test abs(mean_samples - 0.5f0) < 0.3f0  # generous tolerance
    end

    @testset "Integration: Ball+Stick data, train VI, sample posterior" begin
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

            # Use parameter-space loss for non-differentiable physics forward model
            # (the MCM signal function uses mutation internally, incompatible with Zygote)
            forward_fn = identity  # placeholder; not called in parameter-space mode

            model = build_vi_net(;
                obs_dim = n_meas,
                param_dim = par_d,
                hidden_dim = 64,
                depth = 2,
            )

            rng_vi = MersenneTwister(0)
            ps, st = Lux.setup(rng_vi, model)

            ps, st, losses = train_vi!(model, ps, st, all_params, all_signals,
                MersenneTwister(1), forward_fn;
                n_epochs = 30, batch_size = 32, lr = 1e-3,
                sigma_noise = 0.05f0, n_mc_samples = 4,
                use_parameter_loss = true)

            @test losses[end] < losses[1]

            test_signal = all_signals[:, 1:1]
            samples = sample_vi(model, ps, st, test_signal, MersenneTwister(99);
                n_samples = 200, param_dim = par_d)

            @test size(samples) == (par_d, 200)
            @test all(isfinite, samples)
        else
            @info "Skipping Ball+Stick VI integration test (DMI module not loaded). Run via runtests.jl for full coverage."
            @test true  # placeholder so testset isn't empty
        end
    end

end
