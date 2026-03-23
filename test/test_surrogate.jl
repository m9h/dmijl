"""
Tests for the Bloch-Torrey neural surrogate.

The surrogate must:
1. Reproduce analytical solutions after training (unit tests)
2. Match MCMRSimulator.jl for simple geometries (integration tests)
3. Generalize to unseen parameters within the training range

These tests define the SPEC before implementation.
"""

using Test, LinearAlgebra, Random, Statistics

include("../src/models/ball_stick.jl")
include("../src/pipeline/acquisition.jl")

@testset "Bloch-Torrey Surrogate" begin

    @testset "Surrogate reproduces free diffusion after training" begin
        # SPEC: After training on Bloch-Torrey solutions, the surrogate
        # should match exp(-bD) for free diffusion to within 1% relative error.
        #
        # This test will SKIP until the surrogate is implemented.
        # It defines the acceptance criterion.

        @test_skip begin
            # surrogate = train_bt_surrogate(training_data)
            # D_test = 2.0e-9
            # b_test = [0, 500e6, 1000e6, 2000e6, 3000e6]
            # pred = surrogate(D_test, b_test)
            # exact = exp.(-b_test .* D_test)
            # @test maximum(abs.(pred .- exact) ./ max.(exact, 1e-10)) < 0.01
            true
        end
    end

    @testset "Surrogate reproduces stick model" begin
        # SPEC: For a single infinite cylinder (stick), the surrogate
        # should match the analytical solution: S = exp(-b D cos²θ)
        # to within 2% for all gradient directions.

        @test_skip begin
            # D = 1.7e-9
            # mu = [0, 0, 1]  # stick along z
            # acq = hcp_like_acquisition()
            # pred = surrogate(D, mu, acq)
            # exact = exp.(-acq.bvalues .* D .* (acq.gradient_directions * mu).^2)
            # @test maximum(abs.(pred .- exact) ./ max.(exact, 1e-10)) < 0.02
            true
        end
    end

    @testset "Surrogate generalizes within training range" begin
        # SPEC: For parameters NOT in the training set but within the
        # training range, the surrogate matches the forward model
        # to within 5% relative error on average.

        @test_skip begin
            # acq = hcp_like_acquisition()
            # model = BallStickModel(acq.bvalues, acq.gradient_directions)
            # rng = MersenneTwister(999)
            #
            # errors = Float64[]
            # for _ in 1:100
            #     params = random_params(rng)
            #     exact = simulate(model, params)
            #     pred = surrogate(params, acq)
            #     rel_err = mean(abs.(pred .- exact) ./ max.(exact, 1e-10))
            #     push!(errors, rel_err)
            # end
            # @test mean(errors) < 0.05
            true
        end
    end

    @testset "Surrogate is faster than Monte Carlo" begin
        # SPEC: The surrogate should be at least 100x faster than
        # MCMRSimulator.jl for a single forward evaluation.

        @test_skip begin
            # t_mc = @elapsed mc_signal = mcmr_simulate(geometry, sequence)
            # t_nn = @elapsed nn_signal = surrogate(params, acq)
            # @test t_mc / t_nn > 100
            true
        end
    end

    @testset "Surrogate preserves physics invariants" begin
        # SPEC: All physics invariants from test_physics.jl must hold
        # for surrogate predictions, not just the analytical model.

        @test_skip begin
            # Test rotation equivariance of surrogate
            # Test signal bounds [0, 1]
            # Test b=0 normalization
            # Test monotonic decay
            true
        end
    end
end

# ================================================================
# Validation against MCMRSimulator.jl (when available)
# ================================================================

@testset "MCMRSimulator Integration" begin

    @testset "Free diffusion: surrogate vs Monte Carlo" begin
        @test_skip begin
            # Requires MCMRSimulator.jl
            # geometry = empty_domain(size=10e-6)  # no barriers
            # sequence = pgse_sequence(b=1000e6, delta=10e-3, Delta=30e-3)
            # D = 2.0e-9
            #
            # mc_signal = mcmr_simulate(geometry, sequence, D=D, n_spins=100_000)
            # analytical = exp(-1000e6 * D)
            # surrogate_pred = surrogate(D, 1000e6)
            #
            # @test abs(mc_signal - analytical) / analytical < 0.02
            # @test abs(surrogate_pred - analytical) / analytical < 0.01
            true
        end
    end

    @testset "Cylinder: restricted diffusion" begin
        @test_skip begin
            # Requires MCMRSimulator.jl
            # This is the key test: restricted diffusion inside a cylinder
            # has no closed-form solution. The surrogate must learn it.
            #
            # geometry = cylinder(radius=2e-6, orientation=[0,0,1])
            # sequence = pgse_sequence(b=2000e6, direction=[1,0,0])
            # D_intra = 1.7e-9
            #
            # mc_signal = mcmr_simulate(geometry, sequence, D=D_intra, n_spins=100_000)
            # surrogate_pred = surrogate(geometry_params, sequence_params)
            #
            # @test abs(surrogate_pred - mc_signal) / mc_signal < 0.05
            true
        end
    end

    @testset "Sphere: restricted diffusion" begin
        @test_skip begin
            # Sphere has a known GPD approximation but PINN should be exact
            # geometry = sphere(radius=5e-6)
            # Compare PINN vs MC for multiple b-values
            true
        end
    end
end
