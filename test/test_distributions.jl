# Phase 6: Orientation distribution models (Watson, DistributedModel)
using Test, LinearAlgebra, Random, SpecialFunctions

@testset "Distribution Models" begin

    # ---- Watson Distribution ----
    @testset "WatsonDistribution" begin
        @testset "construction" begin
            w = WatsonDistribution(n_grid=100)
            @test length(w.grid_vectors) == 100
            @test size(w.grid_vectors[1]) == (3,)
        end

        @testset "grid vectors are unit vectors" begin
            w = WatsonDistribution(n_grid=200)
            for v in w.grid_vectors
                @test norm(v) ≈ 1.0 atol=1e-12
            end
        end

        @testset "weights sum to 1 for any kappa and mu" begin
            w = WatsonDistribution(n_grid=300)
            for kappa in [0.0, 1.0, 5.0, 20.0, 100.0]
                mu = [0.0, 0.0, 1.0]
                weights = watson_weights(w, mu, kappa)
                @test sum(weights) ≈ 1.0 atol=1e-4
            end
        end

        @testset "high kappa concentrates near mu" begin
            w = WatsonDistribution(n_grid=500)
            mu = [0.0, 0.0, 1.0]
            weights = watson_weights(w, mu, 100.0)

            # Watson is antipodally symmetric, so mean direction cancels.
            # Instead check that weight is concentrated near the poles.
            polar_weight = sum(weights[i] for i in 1:length(weights)
                              if abs(dot(w.grid_vectors[i], mu)) > 0.9)
            @test polar_weight > 0.5  # >50% of weight near poles
        end

        @testset "kappa=0 gives uniform distribution" begin
            w = WatsonDistribution(n_grid=500)
            mu = [0.0, 0.0, 1.0]
            weights = watson_weights(w, mu, 0.0)
            # All weights should be approximately equal
            mean_w = mean(weights)
            @test all(abs.(weights .- mean_w) .< 0.01)
        end
    end

    # ---- DistributedModel ----
    @testset "DistributedModel" begin
        @testset "Watson-distributed Stick is a compartment" begin
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            watson = WatsonDistribution(n_grid=200)
            dm = DistributedModel(stick, watson, :mu, [0.0, 0.0, 1.0], 10.0)
            @test dm isa AbstractCompartment
        end

        @testset "signal bounded [0, 1]" begin
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            watson = WatsonDistribution(n_grid=200)
            dm = DistributedModel(stick, watson, :mu, [0.0, 0.0, 1.0], 10.0)
            acq = hcp_like_acquisition()
            sig = signal(dm, acq)
            @test all(sig .>= -1e-10)
            @test all(sig .<= 1.0 + 1e-10)
        end

        @testset "b=0 signal is 1.0" begin
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            watson = WatsonDistribution(n_grid=200)
            dm = DistributedModel(stick, watson, :mu, [0.0, 0.0, 1.0], 10.0)
            acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
            @test signal(dm, acq)[1] ≈ 1.0 atol=1e-6
        end

        @testset "high kappa recovers single-orientation Stick" begin
            mu = [0.0, 0.0, 1.0]
            d_par = 1.7e-9
            stick = C1Stick(mu=mu, lambda_par=d_par)
            watson = WatsonDistribution(n_grid=2000)
            dm = DistributedModel(stick, watson, :mu, mu, 500.0)
            acq = hcp_like_acquisition()

            sig_dm = signal(dm, acq)
            sig_stick = signal(stick, acq)
            @test maximum(abs.(sig_dm .- sig_stick)) < 0.05
        end

        @testset "kappa=0 matches analytical powder average" begin
            # Powder average of stick: ∫ exp(-b*D*cos²θ) dΩ / 4π
            # = √(π/(4bD)) * erf(√(bD))   for bD > 0
            d_par = 1.7e-9
            b = 2000e6

            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=d_par)
            watson = WatsonDistribution(n_grid=1000)
            dm = DistributedModel(stick, watson, :mu, [0.0, 0.0, 1.0], 0.0)  # uniform

            acq = Acquisition([0.0, b], [1.0 0.0 0.0; 1.0 0.0 0.0])
            sig = signal(dm, acq)

            # Analytical powder average
            bD = b * d_par
            powder_avg = sqrt(pi / (4 * bD)) * erf(sqrt(bD))

            @test sig[1] ≈ 1.0 atol=1e-4
            @test sig[2] ≈ powder_avg atol=0.02
        end

        @testset "Watson-dispersed Zeppelin signal" begin
            mu = [0.0, 0.0, 1.0]
            zep = G2Zeppelin(mu=mu, lambda_par=1.7e-9, lambda_perp=0.5e-9)
            watson = WatsonDistribution(n_grid=300)
            dm = DistributedModel(zep, watson, :mu, mu, 5.0)
            acq = hcp_like_acquisition()
            sig = signal(dm, acq)

            @test all(sig .>= -1e-10)
            @test all(sig .<= 1.0 + 1e-10)

            # Dispersed signal should be more isotropic than undispersed
            # (lower variance across directions at same b-value)
            sig_undispersed = signal(zep, acq)
            b2_mask = acq.bvalues .≈ 2e9
            var_dispersed = var(sig[b2_mask])
            var_undispersed = var(sig_undispersed[b2_mask])
            @test var_dispersed < var_undispersed
        end

        @testset "composable with MultiCompartmentModel" begin
            mu = [0.0, 0.0, 1.0]
            stick = C1Stick(mu=mu, lambda_par=1.7e-9)
            watson = WatsonDistribution(n_grid=200)
            dm_stick = DistributedModel(stick, watson, :mu, mu, 10.0)
            ball = G1Ball(lambda_iso=3.0e-9)

            # DistributedModel should work in MCM
            mcm = MultiCompartmentModel(dm_stick, ball)
            acq = hcp_like_acquisition()

            # dm_stick params: lambda_par(1), mu(3), kappa(1) = 5
            # ball params: lambda_iso(1) = 1
            # fractions: 2
            # total: 8
            @test nparams(mcm) == 8
            params = [1.7e-9, mu..., 10.0, 3.0e-9, 0.6, 0.4]
            sig = signal(mcm, acq, params)
            @test all(sig .>= -1e-10)
            @test all(sig .<= 1.0 + 1e-10)
        end
    end
end
