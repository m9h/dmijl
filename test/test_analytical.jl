"""
Test Bloch-Torrey solutions against known analytical cases.

These are the unit tests: if the surrogate can't match these exactly,
it can't be trusted for anything harder.
"""

using Test, LinearAlgebra, Random, Statistics

# Include source directly for now (before package is registered)
include("../src/models/ball_stick.jl")
include("../src/pipeline/acquisition.jl")

@testset "Analytical Solutions" begin

    @testset "Free isotropic diffusion (Ball model)" begin
        # For free diffusion with coefficient D:
        # S(b) = S₀ exp(-b D)
        # This is the exact Bloch-Torrey solution with no barriers.

        D = 2.0e-9  # m²/s (typical CSF)
        bvals = [0.0, 500e6, 1000e6, 2000e6, 3000e6]
        expected = [exp(-b * D) for b in bvals]

        # Our forward model should reproduce this exactly
        bvecs = repeat([1.0 0.0 0.0], length(bvals), 1)
        model = BallStickModel(bvals, bvecs)

        # Ball-only: f1=0, f2=0, d_ball=D, d_stick=D (irrelevant)
        params = [D, D, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        signal = simulate(model, params)

        for i in eachindex(bvals)
            @test signal[i] ≈ expected[i] atol=1e-10
        end
    end

    @testset "Single stick along gradient" begin
        # Stick along z with gradient along z:
        # S(b) = exp(-b D cos²θ) where θ=0 → S = exp(-b D)
        # Stick along z with gradient along x:
        # cos²θ = 0 → S = 1 (no attenuation)

        D = 1.7e-9
        b = 1000e6

        # Gradient along z, stick along z
        model_z = BallStickModel([0.0, b], [1.0 0.0 0.0; 0.0 0.0 1.0])
        params_z = [D, D, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        sig_z = simulate(model_z, params_z)
        @test sig_z[1] ≈ 1.0 atol=1e-10  # b=0
        @test sig_z[2] ≈ exp(-b * D) atol=1e-10  # parallel

        # Gradient along x, stick along z → cos²θ = 0
        model_x = BallStickModel([b], reshape([1.0 0.0 0.0], 1, 3))
        params_x = [D, D, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        sig_x = simulate(model_x, params_x)
        @test sig_x[1] ≈ 1.0 atol=1e-6  # perpendicular → no decay
    end

    @testset "Signal at b=0 is always 1" begin
        # Regardless of parameters, S(b=0) = f1 + f2 + f_ball = 1
        acq = hcp_like_acquisition()
        model = BallStickModel(acq.bvalues, acq.gradient_directions)

        for _ in 1:10
            d_ball = 1e-9 + rand() * 2.5e-9
            d_stick = 0.5e-9 + rand() * 2e-9
            f1 = 0.1 + rand() * 0.5
            f2 = 0.05 + rand() * min(0.4, 0.95 - f1)
            mu = randn(3); mu ./= norm(mu)
            mu2 = randn(3); mu2 ./= norm(mu2)

            params = [d_ball, d_stick, f1, f2, mu..., mu2...]
            signal = simulate(model, params)

            b0_idx = findall(acq.bvalues .== 0.0)
            for i in b0_idx
                @test signal[i] ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "Signal monotonically decreases with b-value (isotropic)" begin
        # For isotropic diffusion (ball only), signal strictly decreases with b
        D = 2.0e-9
        bvals = collect(range(0, 3000e6, length=20))
        bvecs = repeat([1.0 0.0 0.0], length(bvals), 1)
        model = BallStickModel(bvals, bvecs)
        params = [D, D, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        signal = simulate(model, params)

        for i in 2:length(signal)
            @test signal[i] <= signal[i-1] + 1e-12
        end
    end

    @testset "Volume fractions sum to 1" begin
        # f1 + f2 + f_ball = 1 is enforced by the model
        # Signal at b=0 tests this implicitly
        # But also: weighted sum of compartment signals at any b should be bounded [0, 1]
        acq = hcp_like_acquisition()
        model = BallStickModel(acq.bvalues, acq.gradient_directions)

        params = [2e-9, 1.5e-9, 0.4, 0.3, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        signal = simulate(model, params)

        @test all(0.0 .<= signal .<= 1.0 .+ 1e-10)
    end
end
