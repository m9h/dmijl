using Test, LinearAlgebra

@testset "Parameter Constraints" begin

    @testset "FixedParameter removes parameter from optimization" begin
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        ball = G1Ball(lambda_iso=3.0e-9)
        mcm = MultiCompartmentModel(stick, ball)

        constrained = set_fixed_parameter(mcm, :lambda_par, 1.7e-9)

        @test nparams(constrained) == nparams(mcm) - 1
        @test :lambda_par ∉ parameter_names(constrained)
    end

    @testset "FixedParameter signal matches manual computation" begin
        mu = [0.0, 0.0, 1.0]
        d_par = 1.7e-9
        d_iso = 3.0e-9

        stick = C1Stick(mu=mu, lambda_par=d_par)
        ball = G1Ball(lambda_iso=d_iso)
        mcm = MultiCompartmentModel(stick, ball)
        constrained = set_fixed_parameter(mcm, :lambda_par, d_par)

        acq = hcp_like_acquisition()

        # Full MCM params: [mu..., lambda_par, lambda_iso, f_stick, f_ball]
        full_params = [mu..., d_par, d_iso, 0.6, 0.4]
        expected = signal(mcm, acq, full_params)

        # Constrained params: lambda_par removed → [mu..., lambda_iso, f_stick, f_ball]
        free_params = [mu..., d_iso, 0.6, 0.4]
        result = signal(constrained, acq, free_params)

        @test result ≈ expected atol=1e-12
    end

    @testset "Volume fraction unity" begin
        ball = G1Ball(lambda_iso=3.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        dot = S1Dot()
        mcm = MultiCompartmentModel(stick, ball, dot)

        constrained = set_volume_fraction_unity(mcm)

        # Last fraction derived: f_dot = 1 - f_stick - f_ball
        @test nparams(constrained) == nparams(mcm) - 1
        @test Symbol(:partial_volume_, length(mcm.compartments)) ∉ parameter_names(constrained)

        # At b=0, all compartment signals are 1, so total = sum(fractions) = 1.0
        acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
        # Free params: [mu..., lambda_par, lambda_iso, f_stick, f_ball] (f_dot derived)
        free_params = [0.0, 0.0, 1.0, 1.7e-9, 3.0e-9, 0.3, 0.5]
        sig = signal(constrained, acq, free_params)
        @test sig[1] ≈ 1.0 atol=1e-15
    end

    @testset "Tortuosity constraint" begin
        mu = [0.0, 0.0, 1.0]
        d_par = 1.7e-9
        f_intra = 0.6

        stick = C1Stick(mu=mu, lambda_par=d_par)
        zep = G2Zeppelin(mu=mu, lambda_par=d_par, lambda_perp=d_par * (1 - f_intra))
        mcm = MultiCompartmentModel(stick, zep)

        constrained = set_tortuosity(mcm,
            target=:lambda_perp,
            lambda_par_name=:lambda_par_2,
            volume_fraction_name=:partial_volume_1
        )

        @test :lambda_perp ∉ parameter_names(constrained)
        @test nparams(constrained) == nparams(mcm) - 1

        # Verify signal matches manual computation
        acq = hcp_like_acquisition()

        # Full MCM params: [mu_stick..., lambda_par_stick, mu_zep..., lambda_par_zep, lambda_perp, f_stick, f_zep]
        d_perp = d_par * (1 - f_intra)
        full_params = [mu..., d_par, mu..., d_par, d_perp, f_intra, 1.0 - f_intra]
        expected = signal(mcm, acq, full_params)

        # Constrained: lambda_perp removed
        free_params = [mu..., d_par, mu..., d_par, f_intra, 1.0 - f_intra]
        result = signal(constrained, acq, free_params)
        @test result ≈ expected atol=1e-12
    end
end
