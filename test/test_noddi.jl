using Test, LinearAlgebra, Random

@testset "NODDI-Watson" begin
    @testset "noddi_watson construction" begin
        model = noddi_watson()
        @test model isa ConstrainedModel
        # Free parameters: mu(3), kappa(1), f_intra(1), f_iso(1) = 6
        # Fixed: lambda_par=1.7e-9, lambda_iso=3.0e-9
        # Derived: lambda_perp via tortuosity, f_extra via fraction unity
        @test nparams(model) == 6
        names = parameter_names(model)
        @test :mu in names
        @test :kappa in names
    end

    @testset "signal bounded [0, 1]" begin
        model = noddi_watson()
        acq = hcp_like_acquisition()
        # Free params: [mu(3), kappa(1), f_intra(1), f_iso(1)]
        params = [0.0, 0.0, 1.0, 10.0, 0.5, 0.1]
        sig = signal(model, acq, params)
        @test all(sig .>= -1e-10)
        @test all(sig .<= 1.0 + 1e-10)
    end

    @testset "b=0 signal is 1.0" begin
        model = noddi_watson()
        acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
        params = [0.0, 0.0, 1.0, 10.0, 0.5, 0.1]
        @test signal(model, acq, params)[1] ≈ 1.0 atol=1e-6
    end

    @testset "f_iso=1 gives pure ball signal" begin
        model = noddi_watson()
        acq = hcp_like_acquisition()
        # f_intra=0, f_iso=1 -> all CSF
        params = [0.0, 0.0, 1.0, 10.0, 0.0, 1.0]
        sig = signal(model, acq, params)
        ball_sig = signal(G1Ball(lambda_iso=3.0e-9), acq)
        @test maximum(abs.(sig .- ball_sig)) < 0.01
    end

    @testset "high kappa, f_iso=0 gives stick-like signal" begin
        model = noddi_watson()
        acq = hcp_like_acquisition()
        # f_intra=0.7, f_iso=0, high kappa (concentrated)
        params = [0.0, 0.0, 1.0, 100.0, 0.7, 0.0]
        sig = signal(model, acq, params)
        # Should be anisotropic - different signals for different directions at same b
        b2_mask = acq.bvalues .≈ 2e9
        @test var(sig[b2_mask]) > 0.001
    end

    @testset "noise-free parameter recovery" begin
        model = noddi_watson()
        acq = hcp_like_acquisition()
        true_params = [0.0, 0.0, 1.0, 10.0, 0.5, 0.1]
        data = signal(model, acq, true_params)
        init = [0.1, 0.1, 0.9, 5.0, 0.4, 0.2]
        result = fit_mcm(model, acq, data; init=init)
        fitted = result[:parameters]
        @test fitted[4] ≈ 10.0 atol=5.0    # kappa (concentration)
        @test fitted[5] ≈ 0.5 atol=0.1     # f_intra
        @test fitted[6] ≈ 0.1 atol=0.1     # f_iso
    end

    @testset "custom parameters" begin
        model = noddi_watson(; d_par=1.5e-9, d_iso=2.5e-9, n_grid=100)
        @test model isa ConstrainedModel
    end
end
