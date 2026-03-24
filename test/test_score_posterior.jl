"""
Tests for score-based posterior estimation.

Updated to use the new ScoreNetwork (FiLM-conditioned Lux layers) API.
"""

using Test, Random, Statistics, LinearAlgebra
using Lux, Zygote, Optimisers

include("../src/models/ball_stick.jl")
include("../src/pipeline/acquisition.jl")
include("../src/noise.jl")
include("../src/diffusion/schedule.jl")
include("../src/diffusion/score_net.jl")
include("../src/diffusion/sample.jl")
include("../src/validation/metrics.jl")

@testset "Score Posterior" begin

    @testset "VP schedule properties" begin
        s = VPSchedule(0.01, 5.0)

        # alpha_bar(0) ≈ 1 (no noise at t=0)
        @test alpha_bar(s, 0.0) ≈ 1.0 atol=1e-6

        # alpha_bar(1) < alpha_bar(0) (noise increases with t)
        @test alpha_bar(s, 1.0) < alpha_bar(s, 0.0)

        # alpha_bar is monotonically decreasing
        ts = range(0, 1, length=100)
        abs_vals = [alpha_bar(s, t) for t in ts]
        for i in 2:length(abs_vals)
            @test abs_vals[i] <= abs_vals[i-1] + 1e-10
        end

        # noise_and_signal: signal^2 + noise^2 = 1
        for t in [0.0, 0.1, 0.5, 0.9, 1.0]
            sig, noi = noise_and_signal(s, t)
            @test sig^2 + noi^2 ≈ 1.0 atol=1e-6
        end
    end

    @testset "Posterior samples have correct shape" begin
        # Build a small ScoreNetwork and run sample_posterior
        rng = MersenneTwister(42)
        param_dim = 10
        signal_dim = 8
        n_samples = 5

        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = 16,
            depth = 3,
            cond_dim = 16,
        )
        ps, st = Lux.setup(rng, model)

        # Fake observed signal (must be a vector, not matrix)
        signal = randn(rng, Float32, signal_dim)

        samples = sample_posterior(
            model, ps, st, signal;
            n_samples = n_samples,
            n_steps = 5,       # minimal steps for speed
            n_scalars = 4,
            n_vectors = 2,
        )

        @test size(samples) == (param_dim, n_samples)
        @test all(isfinite, samples)
    end

    @testset "Posterior concentrates around truth (easy case)" begin
        # SPEC: For high SNR (SNR=100), the posterior median should be
        # within 10% of the true parameters for scalar params.

        @test_skip begin
            # Requires trained network -- skipped for CI speed
            # sim = build_simulator(snr=100)
            # net = train(sim, n_steps=50_000)
            # theta_true, signal = sample_and_simulate(sim, 1)
            # samples = sample_posterior(net, signal; n_samples=500)
            # median_pred = median(samples, dims=2)
            # for i in 1:4  # scalar params
            #     rel_err = abs(median_pred[i] - theta_true[i]) / theta_true[i]
            #     @test rel_err < 0.1
            # end
            true
        end
    end

    @testset "Orientation posterior on unit sphere" begin
        # After sample_posterior, orientation vectors should be unit vectors
        rng = MersenneTwister(99)
        param_dim = 10
        signal_dim = 8
        n_samples = 5

        model = build_score_net(;
            param_dim = param_dim,
            signal_dim = signal_dim,
            hidden_dim = 16,
            depth = 3,
            cond_dim = 16,
        )
        ps, st = Lux.setup(rng, model)
        signal = randn(rng, Float32, signal_dim)

        samples = sample_posterior(
            model, ps, st, signal;
            n_samples = n_samples,
            n_steps = 5,
            n_scalars = 4,
            n_vectors = 2,
        )

        # Orientation vectors (indices 5:7 and 8:10) should be unit norm
        for j in 1:n_samples
            mu1 = samples[5:7, j]
            @test norm(mu1) ≈ 1.0 atol=1e-4
            mu2 = samples[8:10, j]
            @test norm(mu2) ≈ 1.0 atol=1e-4
        end
    end

    @testset "Metrics computation" begin
        # Test that evaluation metrics work correctly
        rng = MersenneTwister(42)

        # Perfect predictions -> zero error
        theta = randn(rng, 10, 50)
        # Normalize orientations
        for j in 1:50
            theta[5:7, j] ./= norm(theta[5:7, j])
            theta[8:10, j] ./= norm(theta[8:10, j])
        end

        result = evaluate_ball2stick(theta, theta)
        @test result.fiber1_median ≈ 0.0 atol=1e-6
        @test result.fiber2_median ≈ 0.0 atol=1e-6
        for (name, r) in result.correlations
            @test r ≈ 1.0 atol=1e-6
        end
    end

    @testset "Angular error properties" begin
        # Parallel vectors -> 0 degrees
        @test angular_error_deg([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) ≈ 0.0 atol=1e-10

        # Antipodal vectors -> 0 degrees (we use abs(dot))
        @test angular_error_deg([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) ≈ 0.0 atol=1e-10

        # Perpendicular -> 90 degrees
        @test angular_error_deg([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) ≈ 90.0 atol=1e-10

        # 45 degree angle
        v = [1.0, 1.0, 0.0] / sqrt(2)
        @test angular_error_deg([1.0, 0.0, 0.0], v) ≈ 45.0 atol=1e-6
    end

    @testset "Pearson r properties" begin
        x = collect(1.0:100.0)
        @test pearson_r(x, x) ≈ 1.0 atol=1e-10
        @test pearson_r(x, -x) ≈ -1.0 atol=1e-10
        @test abs(pearson_r(x, randn(100))) < 0.3  # uncorrelated
    end
end
