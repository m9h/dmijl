using Test, LinearAlgebra, Random

@testset "Fitting" begin

    @testset "fit_mcm returns fitted parameters" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)
        acq = hcp_like_acquisition()

        true_params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]
        data = signal(mcm, acq, true_params)

        result = fit_mcm(mcm, acq, data)
        @test haskey(result, :parameters)
        @test haskey(result, :residuals)
        @test length(result[:parameters]) == nparams(mcm)
    end

    @testset "recover Ball+Stick params from noise-free data" begin
        acq = hcp_like_acquisition()
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        true_params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]
        data = signal(mcm, acq, true_params)

        init = [1.5e-9, 0.1, 0.1, 0.9, 1.5e-9, 0.5, 0.5]
        result = fit_mcm(mcm, acq, data; init=init)
        fitted = result[:parameters]

        @test fitted[1] ≈ 2.0e-9 rtol=0.05   # lambda_iso
        @test fitted[5] ≈ 1.7e-9 rtol=0.05   # lambda_par
        @test fitted[6] ≈ 0.3 atol=0.05       # f_ball
        @test fitted[7] ≈ 0.7 atol=0.05       # f_stick
    end

    @testset "fit with constrained model" begin
        acq = hcp_like_acquisition()
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        ball = G1Ball(lambda_iso=3.0e-9)
        mcm = MultiCompartmentModel(stick, ball)
        constrained = set_fixed_parameter(mcm, :lambda_par, 1.7e-9)

        full_params = [0.0, 0.0, 1.0, 1.7e-9, 3.0e-9, 0.6, 0.4]
        data = signal(mcm, acq, full_params)

        result = fit_mcm(constrained, acq, data)
        @test length(result[:parameters]) == nparams(constrained)
    end

    @testset "batched voxel-wise fitting" begin
        acq = hcp_like_acquisition()
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        rng = MersenneTwister(42)
        n_voxels = 5
        data = zeros(n_voxels, length(acq.bvalues))
        true_fracs = zeros(n_voxels)

        for i in 1:n_voxels
            d_iso = 1.5e-9 + rand(rng) * 1.5e-9
            d_par = 1.0e-9 + rand(rng) * 1.5e-9
            mu_v = randn(rng, 3); mu_v ./= norm(mu_v)
            f = 0.2 + rand(rng) * 0.6
            p = [d_iso, mu_v..., d_par, f, 1.0 - f]
            true_fracs[i] = f
            data[i, :] = signal(mcm, acq, p)
        end

        results = fit_mcm_batch(mcm, acq, data)
        @test size(results[:parameters]) == (n_voxels, nparams(mcm))
    end
end
