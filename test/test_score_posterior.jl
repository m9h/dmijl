"""
Tests for score-based posterior estimation.

TDD: define what "working" means before implementing.
"""

using Test, Random, Statistics, LinearAlgebra

include("../src/models/ball_stick.jl")
include("../src/pipeline/acquisition.jl")
include("../src/noise.jl")
include("../src/diffusion/schedule.jl")
include("../src/validation/metrics.jl")

@testset "Score Posterior" begin

    @testset "VP schedule properties" begin
        s = VPSchedule(0.01, 5.0)

        # ᾱ(0) ≈ 1 (no noise at t=0)
        @test alpha_bar(s, 0.0) ≈ 1.0 atol=1e-6

        # ᾱ(1) < ᾱ(0) (noise increases with t)
        @test alpha_bar(s, 1.0) < alpha_bar(s, 0.0)

        # ᾱ is monotonically decreasing
        ts = range(0, 1, length=100)
        abs = [alpha_bar(s, t) for t in ts]
        for i in 2:length(abs)
            @test abs[i] <= abs[i-1] + 1e-10
        end

        # noise_and_signal: signal² + noise² = 1
        for t in [0.0, 0.1, 0.5, 0.9, 1.0]
            sig, noi = noise_and_signal(s, t)
            @test sig^2 + noi^2 ≈ 1.0 atol=1e-6
        end
    end

    @testset "Posterior samples have correct shape" begin
        @test_skip begin
            # After training, sample_posterior should return (param_dim, n_samples)
            # net, ps, st = train_score!(...)
            # samples = sample_posterior(net, ps, st, signal; n_samples=100)
            # @test size(samples) == (10, 100)
            true
        end
    end

    @testset "Posterior concentrates around truth (easy case)" begin
        # SPEC: For high SNR (SNR=100), the posterior median should be
        # within 10% of the true parameters for scalar params.

        @test_skip begin
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
        # SPEC: Orientation samples should be unit vectors
        @test_skip begin
            # samples = sample_posterior(...)
            # for j in 1:n_samples
            #     mu1 = samples[5:7, j]
            #     @test norm(mu1) ≈ 1.0 atol=1e-4
            #     mu2 = samples[8:10, j]
            #     @test norm(mu2) ≈ 1.0 atol=1e-4
            # end
            true
        end
    end

    @testset "Metrics computation" begin
        # Test that evaluation metrics work correctly
        rng = MersenneTwister(42)

        # Perfect predictions → zero error
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
        # Parallel vectors → 0°
        @test angular_error_deg([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) ≈ 0.0 atol=1e-10

        # Antipodal vectors → 0° (we use abs(dot))
        @test angular_error_deg([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) ≈ 0.0 atol=1e-10

        # Perpendicular → 90°
        @test angular_error_deg([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) ≈ 90.0 atol=1e-10

        # 45° angle
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
