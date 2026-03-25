"""
Tests for AxCaliber PINN — restricted diffusion in cylinders.
"""

using Test, LinearAlgebra, Random, Statistics, Lux

include("../src/pinn/axcaliber_pinn.jl")

@testset "AxCaliber PINN" begin

    @testset "Van Gelderen: free diffusion limit" begin
        # When R → ∞, restricted signal → free diffusion: exp(-bD)
        D = 2.0e-9
        R = 1e-3  # 1mm — effectively infinite for diffusion
        delta = 10e-3
        Delta = 40e-3
        b = 1000e6  # s/m²

        S_restricted = van_gelderen_cylinder(b, delta, Delta, D, R)
        S_free = exp(-b * D)

        @test S_restricted ≈ S_free atol=0.01
    end

    @testset "Van Gelderen: small radius increases signal" begin
        # Restriction increases signal (less attenuation) compared to free
        D = 2.0e-9
        delta = 10e-3
        Delta = 40e-3
        b = 2000e6

        S_free = van_gelderen_cylinder(b, delta, Delta, D, 100e-6)  # large R
        S_restricted = van_gelderen_cylinder(b, delta, Delta, D, 2e-6)  # 2μm cylinder

        @test S_restricted > S_free  # restriction reduces attenuation
    end

    @testset "Van Gelderen: Δ-dependence" begin
        # At fixed b, longer Δ → more restriction effects → higher signal
        D = 1.7e-9
        R = 3e-6  # 3 μm
        delta = 10e-3
        b = 3000e6

        S_short = van_gelderen_cylinder(b, delta, 18e-3, D, R)
        S_long = van_gelderen_cylinder(b, delta, 55e-3, D, R)

        # With restriction, long Δ sees more restriction → different signal
        @test S_short != S_long  # Δ matters for restricted diffusion
    end

    @testset "Van Gelderen: b=0 gives 1" begin
        @test van_gelderen_cylinder(0.0, 10e-3, 40e-3, 2e-9, 3e-6) ≈ 1.0
    end

    @testset "Van Gelderen: signal in [0, 1]" begin
        rng = MersenneTwister(42)
        for _ in 1:100
            b = rand(rng) * 20000e6
            delta = 5e-3 + rand(rng) * 15e-3
            Delta = delta + rand(rng) * 50e-3
            D = 0.5e-9 + rand(rng) * 2.5e-9
            R = 0.5e-6 + rand(rng) * 10e-6

            S = van_gelderen_cylinder(b, delta, Delta, D, R)
            @test 0.0 <= S <= 1.0 + 1e-10
        end
    end

    @testset "axcaliber_signal: multi-compartment" begin
        b = 3000e6
        delta = 10e-3
        Delta = 40e-3
        D_intra = 1.7e-9
        D_extra = 0.8e-9
        R = 3e-6
        f_intra = 0.5
        g = [1.0, 0.0, 0.0]
        mu = [0.0, 0.0, 1.0]  # fiber along z

        S = axcaliber_signal(b, delta, Delta, D_intra, D_extra, R, f_intra, g, mu)
        @test 0.0 <= S <= 1.0
        @test S > 0.01  # should have some signal
    end

    @testset "build_axcaliber_pinn constructs" begin
        rng = MersenneTwister(42)
        model = build_axcaliber_pinn(; signal_dim=264, hidden_dim=32, depth=3)
        ps, st = Lux.setup(rng, model)

        # Forward pass
        x = randn(rng, Float32, 264, 1)
        out, _ = model(x, ps, st)
        @test size(out) == (7, 1)

        # Decode
        geom = decode_geometry(out[:, 1])
        @test geom.R > 0
        @test geom.D_intra > 0
        @test geom.D_extra > 0
        @test 0 < geom.f_intra < 1
        @test norm(geom.mu) ≈ 1.0 atol=1e-4
    end
end
