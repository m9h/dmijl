# Tests for the Bloch-Torrey neural surrogate, PINN components,
# and MCMRSimulator integration.

using Test, LinearAlgebra, Random, Statistics
using Lux, Zygote

@testset "Bloch-Torrey Surrogate" begin

    @testset "build_surrogate returns a Lux model" begin
        model = build_surrogate(; param_dim=4, signal_dim=8, hidden_dim=16, depth=3)
        rng = MersenneTwister(42)
        ps, st = Lux.setup(rng, model)

        x = randn(rng, Float32, 4, 2)
        out, st_new = model(x, ps, st)
        @test size(out) == (8, 2)
        @test all(0.0f0 .<= out .<= 1.0f0)
    end

    @testset "BlochTorreyResidual construction" begin
        G_fn = t -> Float32[0.0, 0.0, 0.0]
        res = BlochTorreyResidual(; gradient_fn=G_fn)
        @test res.gamma ≈ 2.675e8
        @test res.gradient_fn === G_fn
    end

    @testset "pde_loss computes finite loss" begin
        model = Chain(
            Dense(4 => 16, gelu),
            Dense(16 => 16, gelu),
            Dense(16 => 2),
        )
        rng = MersenneTwister(42)
        ps, st = Lux.setup(rng, model)

        G_fn = t -> Float32[0.0, 0.0, 0.0]
        res = BlochTorreyResidual(; gradient_fn=G_fn)

        n_colloc = 2
        t_c = rand(rng, Float32, n_colloc)
        x_c = randn(rng, Float32, 3, n_colloc)
        D_c = fill(2.0f-9, n_colloc)
        T2_c = fill(80.0f-3, n_colloc)

        loss = pde_loss(res, model, ps, st, t_c, x_c, D_c, T2_c)
        @test isfinite(loss)
        @test loss >= 0.0
    end
end

# ================================================================
# MCMRSimulator Integration Tests
# ================================================================

@testset "MCMRSimulator Integration" begin

    mcmr_available = try
        using MCMRSimulator
        using MRIBuilder: DWI
        true
    catch
        false
    end

    if !mcmr_available
        @info "MCMRSimulator not available — skipping integration tests"
        @test_skip "MCMRSimulator not available"
    else
        @testset "Geometry construction" begin
            @testset "cylinder geometry builds without error" begin
                geo = MCMRGeometry(
                    geometry_type=:cylinders,
                    mean_radius=2.0,
                    radius_variance=0.1,
                    volume_fraction=0.5,
                    box_size=15.0,
                )
                mcmr_geo = DMI.build_mcmr_geometry(geo)
                @test mcmr_geo isa MCMRSimulator.Cylinders
            end

            @testset "sphere geometry builds without error" begin
                geo = MCMRGeometry(
                    geometry_type=:spheres,
                    mean_radius=5.0,
                    radius_variance=0.5,
                    volume_fraction=0.4,
                    box_size=20.0,
                )
                mcmr_geo = DMI.build_mcmr_geometry(geo)
                @test mcmr_geo isa MCMRSimulator.Spheres
            end
        end

        @testset "Free diffusion: signal matches exp(-bD)" begin
            # With no geometry (empty box), signal should match free diffusion
            seq = DWI(bval=1.0, TE=80, TR=1000)
            D = 2.0  # um²/ms

            # Create a simulation with no obstacles (free diffusion)
            sim = MCMRSimulator.Simulation(
                seq;
                diffusivity=D,
                R1=0.0,  # no T1 relaxation
                R2=0.0,  # no T2 relaxation
                verbose=false,
            )

            n_spins = 50_000
            result = MCMRSimulator.readout(n_spins, sim; skip_TR=2, bounding_box=500)
            sig = abs(MCMRSimulator.transverse(result) / n_spins)

            # For free diffusion: S = exp(-b*D)
            # b=1 ms/um², D=2 um²/ms → S = exp(-2) ≈ 0.135
            expected = exp(-1.0 * D)
            @test sig ≈ expected atol=0.02  # MC noise tolerance
        end

        @testset "Cylinder restricted diffusion: signal > free diffusion" begin
            # Inside cylinders, diffusion is restricted → signal is HIGHER
            # than free diffusion at the same b-value (less attenuation)
            seq = DWI(bval=2.0, TE=80, TR=1000)
            D = 2.0

            geo = MCMRGeometry(
                geometry_type=:cylinders,
                mean_radius=2.0,
                radius_variance=0.1,
                volume_fraction=0.6,
                box_size=15.0,
            )
            mcmr_geo = DMI.build_mcmr_geometry(geo)

            sim = MCMRSimulator.Simulation(
                seq;
                geometry=mcmr_geo,
                diffusivity=D,
                R1=0.0,
                R2=0.0,
                verbose=false,
            )

            n_spins = 20_000
            result = MCMRSimulator.readout(n_spins, sim; skip_TR=2, bounding_box=500)
            sig_restricted = abs(MCMRSimulator.transverse(result) / n_spins)

            sig_free = exp(-2.0 * D)

            # Restricted diffusion should give higher signal (less attenuation)
            @test sig_restricted > sig_free
            # But still in valid range
            @test 0.0 < sig_restricted < 1.0
        end

        @testset "Sphere restricted diffusion: signal > free diffusion" begin
            seq = DWI(bval=2.0, TE=80, TR=1000)
            D = 2.0

            geo = MCMRGeometry(
                geometry_type=:spheres,
                mean_radius=5.0,
                radius_variance=0.5,
                volume_fraction=0.4,
                box_size=20.0,
            )
            mcmr_geo = DMI.build_mcmr_geometry(geo)

            sim = MCMRSimulator.Simulation(
                seq;
                geometry=mcmr_geo,
                diffusivity=D,
                R1=0.0,
                R2=0.0,
                verbose=false,
            )

            n_spins = 20_000
            result = MCMRSimulator.readout(n_spins, sim; skip_TR=2, bounding_box=500)
            sig_restricted = abs(MCMRSimulator.transverse(result) / n_spins)

            sig_free = exp(-2.0 * D)

            @test sig_restricted > sig_free
            @test 0.0 < sig_restricted < 1.0
        end

        @testset "generate_mcmr_training_data produces valid data" begin
            seq = DWI(bval=1.0, TE=80, TR=1000)
            n_samples = 3

            params, signals = generate_mcmr_training_data(
                sample_cylinder_geometry,
                seq,
                n_samples;
                n_spins=5_000,
                verbose=false,
            )

            @test size(params, 2) == n_samples
            @test size(signals, 2) == n_samples
            @test size(params, 1) == 3  # mean_radius, radius_variance, volume_fraction

            # Signals should be in valid range
            @test all(signals .>= 0.0)
            @test all(signals .<= 1.5)  # can slightly exceed 1 due to MC noise

            # Parameters should be in prior ranges
            @test all(params[1, :] .>= 0.3)   # mean_radius
            @test all(params[1, :] .<= 5.0)
            @test all(params[3, :] .>= 0.3)   # volume_fraction
            @test all(params[3, :] .<= 0.8)
        end

        @testset "mcmr_data_fn closure works" begin
            seq = DWI(bval=1.0, TE=80, TR=1000)
            data_fn = mcmr_data_fn(seq, :cylinders; n_spins=5_000)

            rng = MersenneTwister(42)
            params, signals = data_fn(rng, 2)

            @test size(params) == (3, 2)
            @test size(signals, 2) == 2
            @test all(isfinite.(signals))
        end
    end
end
