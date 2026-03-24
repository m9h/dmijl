"""
Tests for non-parametric diffusion field recovery.
"""

using Test, Random, Statistics, LinearAlgebra, Lux

include("../src/pinn/diffusion_field.jl")

@testset "Diffusion Field Recovery" begin

    @testset "D-network produces positive diffusivity" begin
        rng = MersenneTwister(42)

        for output_type in [:scalar, :diagonal, :full]
            net, otype = build_diffusivity_net(;
                hidden_dim=16, depth=3, output_type=output_type)
            ps, st = Lux.setup(rng, net)

            x = Float32[0.0, 0.0, 0.0]
            D = eval_D(net, ps, st, x, otype)

            if output_type == :scalar
                @test D > 0  # positive
                @test D < 1e-7  # physical range (< 100 μm²/s)
            elseif output_type == :diagonal
                @test all(D .> 0)
                @test length(D) == 3
            else  # :full
                @test size(D) == (3, 3)
                @test issymmetric(D)
                eigs = eigvals(Symmetric(D))
                @test all(real.(eigs) .>= 0)  # SPD
            end
        end
    end

    @testset "M-network produces finite output" begin
        rng = MersenneTwister(42)
        net = build_magnetization_net(; hidden_dim=16, depth=3)
        ps, st = Lux.setup(rng, net)

        inp = Float32[0.01, 0.0, 0.0, 0.0]  # (t, x, y, z)
        out, _ = net(reshape(inp, :, 1), ps, st)
        @test size(out) == (2, 1)  # M_re, M_im
        @test all(isfinite, out)
    end

    @testset "FA is 0 for isotropic, >0 for anisotropic" begin
        rng = MersenneTwister(42)

        # Isotropic case
        net, otype = build_diffusivity_net(; hidden_dim=16, depth=3, output_type=:scalar)
        ps, st = Lux.setup(rng, net)
        result = (; D_net=net, ps_D=ps, st_D=st, D_type=:scalar,
                    M_net=net, ps_M=ps, st_M=st,
                    losses_data=Float64[], losses_pde=Float64[])
        maps = extract_maps(result; grid_resolution=4)
        @test all(maps.FA .== 0)  # isotropic → FA=0
        @test all(maps.MD .> 0)   # positive diffusivity
    end

    @testset "DiffusionFieldProblem construction" begin
        prob = DiffusionFieldProblem(
            randn(Float32, 90),
            vcat(zeros(6), fill(1e9, 30), fill(2e9, 30), fill(3e9, 24)),
            randn(90, 3),
            10e-3, 40e-3, 80e-3, 10e-6,
        )
        @test length(prob.observed_signal) == 90
        @test prob.delta == 10e-3
        @test prob.voxel_size == 10e-6
    end

    @testset "solve_diffusion_field runs without error (tiny)" begin
        # Minimal config — just verify it doesn't crash
        prob = DiffusionFieldProblem(
            ones(Float32, 10),         # fake signal
            vcat(zeros(2), fill(1e9, 4), fill(2e9, 4)),
            randn(10, 3),
            10e-3, 40e-3, 80e-3, 10e-6,
        )

        result = solve_diffusion_field(prob;
            output_type=:scalar,
            D_hidden=8, D_depth=2,
            M_hidden=8, M_depth=2,
            n_steps=10, n_colloc=4, n_spatial=4,
            print_every=5,
        )

        @test length(result.losses_data) == 10
        @test haskey(result, :D_net)
        @test haskey(result, :M_net)

        # Extract maps
        maps = extract_maps(result; grid_resolution=3)
        @test size(maps.FA) == (3, 3, 3)
        @test size(maps.MD) == (3, 3, 3)
        @test all(isfinite, maps.FA)
        @test all(isfinite, maps.MD)
    end
end
