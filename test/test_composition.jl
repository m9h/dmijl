using Test, LinearAlgebra, Random

@testset "Model Composition" begin

    @testset "MultiCompartmentModel construction" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)
        @test mcm isa MultiCompartmentModel
        @test length(mcm.compartments) == 2
    end

    @testset "combined parameter names with volume fractions" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        names = parameter_names(mcm)
        @test :lambda_iso in names
        @test :mu in names
        @test :lambda_par in names
        @test :partial_volume_1 in names
        @test :partial_volume_2 in names
        @test length(names) == 5
    end

    @testset "collision resolution appends _N suffix" begin
        stick1 = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        stick2 = C1Stick(mu=[1.0, 0.0, 0.0], lambda_par=1.5e-9)
        mcm = MultiCompartmentModel(stick1, stick2)

        names = parameter_names(mcm)
        @test :mu in names
        @test :lambda_par in names
        @test :mu_2 in names
        @test :lambda_par_2 in names
    end

    @testset "nparams includes volume fractions" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)
        # Ball: 1, Stick: 4, Fractions: 2 = 7
        @test nparams(mcm) == 7
    end

    @testset "MCM signal is weighted sum" begin
        D_ball = 2.0e-9
        D_stick = 1.7e-9
        f_ball = 0.3
        f_stick = 0.7
        mu = [0.0, 0.0, 1.0]

        ball = G1Ball(lambda_iso=D_ball)
        stick = C1Stick(mu=mu, lambda_par=D_stick)
        mcm = MultiCompartmentModel(ball, stick)
        acq = hcp_like_acquisition()

        sig_ball = signal(ball, acq)
        sig_stick = signal(stick, acq)
        expected = f_ball .* sig_ball .+ f_stick .* sig_stick

        # Layout: [lambda_iso, mu_x, mu_y, mu_z, lambda_par, f_ball, f_stick]
        params = [D_ball, mu..., D_stick, f_ball, f_stick]
        sig = signal(mcm, acq, params)
        @test sig ≈ expected atol=1e-12
    end

    @testset "b=0 signal equals sum of fractions" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)
        acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))

        # Fractions sum to 1
        params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.4, 0.6]
        @test signal(mcm, acq, params)[1] ≈ 1.0 atol=1e-15

        # Fractions sum to 0.8
        params2 = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.5]
        @test signal(mcm, acq, params2)[1] ≈ 0.8 atol=1e-15
    end

    @testset "Dict <-> array conversion" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        dict = Dict(
            :lambda_iso => [2.0e-9],
            :mu => [0.0, 0.0, 1.0],
            :lambda_par => [1.7e-9],
            :partial_volume_1 => [0.3],
            :partial_volume_2 => [0.7],
        )
        arr = parameter_dictionary_to_array(mcm, dict)
        @test length(arr) == 7
        @test arr ≈ [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]

        # Round-trip
        dict2 = parameter_array_to_dictionary(mcm, arr)
        @test dict2[:lambda_iso] ≈ [2.0e-9]
        @test dict2[:mu] ≈ [0.0, 0.0, 1.0]
        @test dict2[:partial_volume_1] ≈ [0.3]
    end

    @testset "Three-compartment Ball+Stick+Stick matches BallStickModel" begin
        acq = hcp_like_acquisition()

        d_ball = 2.0e-9
        d_stick = 1.5e-9
        f1 = 0.4
        f2 = 0.3
        mu1 = [0.0, 0.0, 1.0]
        mu2 = [1.0, 0.0, 0.0]
        f_ball = 1.0 - f1 - f2

        # Old model: params = [d_ball, d_stick, f1, f2, mu1..., mu2...]
        old_model = BallStickModel(acq.bvalues, acq.gradient_directions)
        old_sig = simulate(old_model, [d_ball, d_stick, f1, f2, mu1..., mu2...])

        # New composable model
        ball = G1Ball(lambda_iso=d_ball)
        stick1 = C1Stick(mu=mu1, lambda_par=d_stick)
        stick2 = C1Stick(mu=mu2, lambda_par=d_stick)
        mcm = MultiCompartmentModel(ball, stick1, stick2)

        # Layout: [d_ball, mu1..., d_stick, mu2..., d_stick, f_ball, f1, f2]
        params = [d_ball, mu1..., d_stick, mu2..., d_stick, f_ball, f1, f2]
        new_sig = signal(mcm, acq, params)
        @test old_sig ≈ new_sig atol=1e-10
    end

    @testset "get_flat_bounds" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        lower, upper = get_flat_bounds(mcm)
        @test length(lower) == nparams(mcm)
        @test length(upper) == nparams(mcm)
        # Volume fractions bounded [0, 1]
        @test lower[end] == 0.0
        @test upper[end] == 1.0
        @test lower[end-1] == 0.0
        @test upper[end-1] == 1.0
        # lambda_iso bounded [0, 3e-9]
        @test lower[1] == 0.0
        @test upper[1] == 3.0e-9
    end
end
