using Test
using DMI
using LinearAlgebra
using Random

@testset "Optimal Experimental Design" begin

    @testset "FIM — Ball+Stick analytical" begin
        ball = G1Ball(lambda_iso=1.7e-9)
        stick = C1Stick(lambda_par=1.7e-9, mu=[0.0, 0.0, 1.0])
        model = MultiCompartmentModel(ball, stick)
        sigma = 0.02

        bvals = vcat(zeros(3), fill(1e9, 10), fill(2e9, 10), fill(3e9, 10))
        dirs = vcat(repeat([1.0 0.0 0.0], 3), electrostatic_directions(30))
        acq = Acquisition(bvals, dirs)

        params = [0.5, 0.5, 1.7e-9, 1.7e-9, 0.0, 0.0, 1.0]
        F = fisher_information(model, acq, params; sigma=sigma)

        np = nparams(model)
        @test size(F) == (np, np)
        @test F ≈ F' atol=1e-10  # symmetric
        @test all(eigvals(Symmetric(F)) .>= -1e-10)  # positive semi-definite
    end

    @testset "Jacobian correctness — finite difference check" begin
        ball = G1Ball(lambda_iso=1.7e-9)
        stick = C1Stick(lambda_par=1.7e-9, mu=[0.0, 0.0, 1.0])
        model = MultiCompartmentModel(ball, stick)
        bvals = Float64[0, 1e9, 2e9, 3e9]
        dirs = repeat([1.0 0.0 0.0], 4)
        acq = Acquisition(bvals, dirs)
        params = [0.5, 0.5, 1.7e-9, 1.7e-9, 0.0, 0.0, 1.0]

        J_ad = jacobian_signal(model, acq, params)

        # Finite difference per parameter
        eps = 1e-7
        np = length(params)
        for j in 1:np
            p_plus = copy(params); p_plus[j] += eps
            p_minus = copy(params); p_minus[j] -= eps
            J_fd_j = (signal(model, acq, p_plus) .- signal(model, acq, p_minus)) ./ (2 * eps)
            @test J_ad[:, j] ≈ J_fd_j atol=1e-3
        end
    end

    @testset "CRLB" begin
        ball = G1Ball(lambda_iso=1.7e-9)
        stick = C1Stick(lambda_par=1.7e-9, mu=[0.0, 0.0, 1.0])
        model = MultiCompartmentModel(ball, stick)
        bvals = vcat(zeros(3), fill(1e9, 10), fill(2e9, 10), fill(3e9, 10))
        dirs = vcat(repeat([1.0 0.0 0.0], 3), electrostatic_directions(30))
        acq = Acquisition(bvals, dirs)
        params = [0.5, 0.5, 1.7e-9, 1.7e-9, 0.0, 0.0, 1.0]

        cr = crlb(model, acq, params; sigma=0.02)
        @test length(cr) == nparams(model)
        @test all(cr .> 0)  # variances must be positive
    end

    @testset "Rician FIM correction" begin
        # High SNR → correction ≈ 1
        @test rician_fim_correction(1.0, 0.001) ≈ 1.0 atol=1e-3
        # Zero signal → no information
        @test rician_fim_correction(0.0, 0.02) == 0.0
        # Moderate SNR → 0 < L < 1
        L_mod = rician_fim_correction(0.1, 0.02)  # SNR=5
        @test 0 < L_mod < 1
        # Monotonic: higher SNR → higher correction
        @test rician_fim_correction(1.0, 0.02) > rician_fim_correction(0.1, 0.02)
    end

    @testset "Optimality criteria" begin
        F = [4.0 1.0; 1.0 3.0]

        d = d_optimality(F)
        a = a_optimality(F)
        e = e_optimality(F)

        @test d ≈ log(det(F)) rtol=1e-6
        @test a ≈ -tr(inv(F)) rtol=1e-6
        @test e ≈ minimum(eigvals(F)) rtol=1e-6

        # Unified interface
        @test optimality_criterion(F; criterion=:D) ≈ d
        @test optimality_criterion(F; criterion=:A) ≈ a
        @test optimality_criterion(F; criterion=:E) ≈ e
    end

    @testset "Expected FIM (Bayesian)" begin
        ball = G1Ball(lambda_iso=1.7e-9)
        stick = C1Stick(lambda_par=1.7e-9, mu=[0.0, 0.0, 1.0])
        model = MultiCompartmentModel(ball, stick)
        bvals = vcat(zeros(2), fill(1e9, 5), fill(2e9, 5))
        dirs = vcat(repeat([1.0 0.0 0.0], 2), electrostatic_directions(10))
        acq = Acquisition(bvals, dirs)

        np = nparams(model)
        # Sample prior: slight variation around nominal
        rng = Random.MersenneTwister(42)
        nominal = [0.5, 0.5, 1.7e-9, 1.7e-9, 0.0, 0.0, 1.0]
        prior = hcat([nominal .+ 0.01 .* randn(rng, np) .* nominal for _ in 1:20]...)
        F_exp = expected_fim(model, acq, prior; sigma=0.02)

        @test size(F_exp) == (np, np)
        @test all(eigvals(Symmetric(F_exp)) .>= -1e-10)
    end

    @testset "Hardware constraints" begin
        G_max = 0.08  # 80 mT/m clinical
        delta = 0.01
        Delta = 0.03
        b_max = max_bvalue(G_max, delta, Delta)
        @test b_max > 0

        G_needed = required_gradient(b_max, delta, Delta)
        @test G_needed ≈ G_max rtol=1e-6

        # Feasibility check
        bvals = [0.0, b_max * 0.5, b_max]
        dirs = repeat([1.0 0.0 0.0], 3)
        acq = Acquisition(bvals, dirs, delta, Delta)
        @test is_feasible(acq, G_max)
        @test !is_feasible(Acquisition([b_max * 2], dirs[1:1, :], delta, Delta), G_max)
    end

    @testset "Electrostatic directions" begin
        dirs = electrostatic_directions(30)
        @test size(dirs) == (30, 3)
        # All unit vectors
        norms = sqrt.(sum(dirs.^2, dims=2))
        @test all(isapprox.(norms, 1.0, atol=1e-10))
        # All in upper hemisphere
        @test all(dirs[:, 3] .>= -1e-10)
    end

    @testset "Compare protocols" begin
        ball = G1Ball(lambda_iso=1.7e-9)
        stick = C1Stick(lambda_par=1.7e-9, mu=[0.0, 0.0, 1.0])
        model = MultiCompartmentModel(ball, stick)
        params = [0.5, 0.5, 1.7e-9, 1.7e-9, 0.0, 0.0, 1.0]
        dirs = electrostatic_directions(5)

        dirs6 = vcat([1.0 0.0 0.0], dirs)
        acq_low = Acquisition(vcat([0.0], fill(1e9, 5)), dirs6)
        acq_high = Acquisition(vcat([0.0], fill(3e9, 5)), dirs6)

        results = compare_protocols(model, [acq_low, acq_high], params;
                                    labels=["low-b", "high-b"])
        @test length(results) == 2
        @test results[1].label == "low-b"
        @test results[2].label == "high-b"
    end
end
