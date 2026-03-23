"""
Physics invariant tests for Bloch-Torrey surrogate.

These are property-based tests: they don't require a reference solver,
just that the surrogate respects physical laws.
"""

using Test, LinearAlgebra, Random

include("../src/models/ball_stick.jl")
include("../src/pipeline/acquisition.jl")
include("../src/noise.jl")

@testset "Physics Invariants" begin

    @testset "Rotation equivariance" begin
        # If we rotate both the fiber orientation AND the gradient directions
        # by the same rotation R, the signal should be identical.
        # S(R·g, R·μ) = S(g, μ)

        acq = hcp_like_acquisition()

        # Random rotation matrix (Rodrigues)
        rng = MersenneTwister(42)
        axis = randn(rng, 3); axis ./= norm(axis)
        angle = rand(rng) * 2π
        K = [0 -axis[3] axis[2]; axis[3] 0 -axis[1]; -axis[2] axis[1] 0]
        R = I + sin(angle) * K + (1 - cos(angle)) * K^2

        # Original
        mu1 = [0.0, 0.0, 1.0]
        mu2 = [1.0, 0.0, 0.0]
        params_orig = [2e-9, 1.5e-9, 0.4, 0.3, mu1..., mu2...]
        model_orig = BallStickModel(acq.bvalues, acq.gradient_directions)
        sig_orig = simulate(model_orig, params_orig)

        # Rotated: rotate both gradients and orientations
        g_rot = acq.gradient_directions * R'
        mu1_rot = R * mu1
        mu2_rot = R * mu2
        params_rot = [2e-9, 1.5e-9, 0.4, 0.3, mu1_rot..., mu2_rot...]
        model_rot = BallStickModel(acq.bvalues, g_rot)
        sig_rot = simulate(model_rot, params_rot)

        @test sig_orig ≈ sig_rot atol=1e-8
    end

    @testset "Antipodal symmetry" begin
        # S(g, μ) = S(g, -μ) because cos²θ is even
        acq = hcp_like_acquisition()
        model = BallStickModel(acq.bvalues, acq.gradient_directions)

        mu = [0.3, 0.5, sqrt(1 - 0.3^2 - 0.5^2)]
        params_pos = [2e-9, 1.5e-9, 0.5, 0.2, mu..., 1.0, 0.0, 0.0]
        params_neg = [2e-9, 1.5e-9, 0.5, 0.2, (-mu)..., 1.0, 0.0, 0.0]

        @test simulate(model, params_pos) ≈ simulate(model, params_neg) atol=1e-10
    end

    @testset "Signal bounded in [0, 1]" begin
        # Physical signal (before noise) must be in [0, 1]
        acq = hcp_like_acquisition()
        model = BallStickModel(acq.bvalues, acq.gradient_directions)
        rng = MersenneTwister(123)

        for _ in 1:100
            d_ball = 0.5e-9 + rand(rng) * 3e-9
            d_stick = 0.5e-9 + rand(rng) * 2e-9
            f1 = rand(rng) * 0.8
            f2 = rand(rng) * min(0.5, 0.95 - f1)
            mu1 = randn(rng, 3); mu1 ./= norm(mu1)
            mu2 = randn(rng, 3); mu2 ./= norm(mu2)

            params = [d_ball, d_stick, f1, f2, mu1..., mu2...]
            signal = simulate(model, params)

            @test all(signal .>= -1e-10)
            @test all(signal .<= 1.0 + 1e-10)
        end
    end

    @testset "Higher b-value → lower signal (on average)" begin
        # For any fixed microstructure, mean signal should decrease with b
        acq = hcp_like_acquisition()
        model = BallStickModel(acq.bvalues, acq.gradient_directions)
        rng = MersenneTwister(456)

        for _ in 1:20
            d_ball = 1e-9 + rand(rng) * 2e-9
            d_stick = 0.5e-9 + rand(rng) * 2e-9
            f1 = 0.2 + rand(rng) * 0.5
            f2 = 0.05 + rand(rng) * 0.2
            mu1 = randn(rng, 3); mu1 ./= norm(mu1)
            mu2 = randn(rng, 3); mu2 ./= norm(mu2)

            params = [d_ball, d_stick, f1, f2, mu1..., mu2...]
            signal = simulate(model, params)

            # Average signal per shell
            shells = [0.0, 1e9, 2e9, 3e9]
            shell_means = Float64[]
            for s in shells
                mask = abs.(acq.bvalues .- s) .< 1e6
                if any(mask)
                    push!(shell_means, mean(signal[mask]))
                end
            end

            # Each shell mean should be ≤ previous (with tolerance for angular effects)
            for i in 2:length(shell_means)
                @test shell_means[i] <= shell_means[i-1] + 0.05
            end
        end
    end

    @testset "Rician noise is positive" begin
        rng = MersenneTwister(789)
        signal = rand(rng, Float32, 90, 100) .* 0.8f0 .+ 0.1f0  # (90, 100)
        noisy = add_rician_noise(rng, signal'; snr_range=(10.0, 50.0))'
        @test all(noisy .>= 0)
    end

    @testset "Rician noise has positive bias" begin
        # Rician noise biases the signal upward (√(S² + σ²) > S on average)
        rng = MersenneTwister(101)
        n = 10_000
        signal = fill(0.5f0, 1, n)
        noisy = add_rician_noise(rng, signal, 1.0f0 / 30.0f0)
        @test mean(noisy) > 0.5  # positive bias
    end
end
