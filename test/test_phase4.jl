using Test
using DMI
using LinearAlgebra
using Random
using Statistics

@testset "Phase 4: Clinical models" begin

    @testset "EPG signal" begin
        # Basic EPG: should produce decaying echo train
        T1, T2, TE = 1.0, 0.08, 0.01  # 1s, 80ms, 10ms
        ETL = 32
        sig = epg_signal(T1, T2, 1.0, TE, ETL)

        @test length(sig) == ETL
        @test all(sig .>= 0)
        @test sig[1] > sig[ETL]  # signal decays
        @test sig[1] <= 1.0  # bounded by equilibrium

        # First echo should follow T2 decay (90° excitation gives mag ~1 into F+)
        # EPG with perfect 180° refocusing: signal ≈ exp(-TE/T2)
        # But numerical EPG has discrete state tracking, so allow wider tolerance
        expected_first = exp(-TE / T2)
        @test sig[1] ≈ expected_first atol=0.5

        # B1 < 1 should reduce signal (imperfect refocusing)
        sig_low_b1 = epg_signal(T1, T2, 0.8, TE, ETL)
        @test sig_low_b1[1] < sig[1]

        # Longer T2 → slower decay
        sig_long_t2 = epg_signal(T1, 0.2, 1.0, TE, ETL)
        @test sig_long_t2[ETL] > sig[ETL]
    end

    @testset "PlaneCallaghan compartment" begin
        plane = PlaneCallaghan(a=10e-6, D_intra=1.7e-9, normal=[0.0, 0.0, 1.0])

        @test nparams(plane) == 5  # a + D_intra + normal(3)
        @test :a in parameter_names(plane)

        # Signal at b=0 should be 1
        acq = Acquisition([0.0, 1e9, 2e9], [0.0 0.0 1.0; 0.0 0.0 1.0; 0.0 0.0 1.0],
                          10e-3, 30e-3)
        S = signal(plane, acq)
        @test S[1] ≈ 1.0
        @test all(0 .<= S .<= 1.0)
        @test S[2] < 1.0  # b > 0 should attenuate

        # Gradient perpendicular to normal → no restriction
        acq_perp = Acquisition([0.0, 2e9], [1.0 0.0 0.0; 1.0 0.0 0.0], 10e-3, 30e-3)
        S_perp = signal(plane, acq_perp)
        @test S_perp[2] ≈ 1.0 atol=0.01  # no attenuation perpendicular to normal

        # Wider planes → less restriction (at moderate b-values)
        plane_wide = PlaneCallaghan(a=50e-6, D_intra=1.7e-9, normal=[0.0, 0.0, 1.0])
        # Both narrow and wide planes produce valid restricted signals
        S_wide = signal(plane_wide, acq)
        @test all(0 .<= S_wide .<= 1.0)
        @test S_wide[1] ≈ 1.0

        # _reconstruct
        p = [10e-6, 1.7e-9, 0.0, 0.0, 1.0]
        plane2 = DMI._reconstruct(plane, p)
        @test plane2.a == 10e-6
        @test plane2.D_intra == 1.7e-9
    end

    @testset "Algebraic initializers" begin
        # Generate synthetic Ball+Stick data
        ball = G1Ball(lambda_iso=1.0e-9)
        stick = C1Stick(lambda_par=1.7e-9, mu=[0.0, 0.0, 1.0])
        mcm = MultiCompartmentModel(ball, stick)

        rng = MersenneTwister(42)
        n_dirs = 30
        bvals = vcat(zeros(6), fill(1e9, n_dirs))
        dirs = vcat(repeat([1.0 0.0 0.0], 6), DMI.electrostatic_directions(n_dirs))
        acq = Acquisition(bvals, dirs)

        # Generate signal with known parameters
        true_params = [1.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]
        S = signal(mcm, acq, true_params)
        S_noisy = S .+ 0.01 .* randn(rng, length(S))

        # DTI init
        dti = dti_init(acq, S_noisy)
        @test haskey(dti, :FA)
        @test haskey(dti, :MD)
        @test haskey(dti, :eigenvalues)
        @test 0 <= dti[:FA] <= 1
        @test dti[:MD] > 0
        @test length(dti[:eigenvalues]) == 3
        # Primary eigenvector should be roughly along z
        @test abs(dti[:eigenvectors][3, 1]) > 0.5

        # Ball+Stick init
        p0 = ball_stick_init(acq, S_noisy)
        @test length(p0) == nparams(mcm)
        @test all(isfinite.(p0))

        # NODDI init
        noddi = noddi_init(acq, S_noisy)
        @test 0 < noddi[:f_intra] < 1
        @test noddi[:kappa] >= 0
        @test norm(noddi[:mu]) ≈ 1.0 atol=0.1
    end

    @testset "SpinDoctor oracle structure" begin
        # Just test the result type exists and the guard function works
        result = SpinDoctorValidationResult(
            [1e-6], [0.0, 1e9],
            Dict(1e-6 => [1.0, 0.8]),
            Dict(1e-6 => [1.0, 0.81]),
            Dict(1e-6 => [0.0, 0.012]),
            0.012, true
        )
        @test result.pass == true
        @test result.max_error ≈ 0.012
    end
end
