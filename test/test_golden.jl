"""
Golden reference tests for DMI.jl — pre-computed datasets with known answers.

These tests lock in current numerical behaviour. If a code change causes outputs
to drift beyond the specified tolerances the test fails, flagging a potential
regression.
"""

using Test, LinearAlgebra

@testset "Golden reference datasets" begin

    # ================================================================
    # 1. Single-compartment reference signals
    # ================================================================
    @testset "Single-compartment golden signals" begin
        bvals = [0.0, 500e6, 1000e6, 2000e6, 3000e6]
        bvecs = repeat([0.0 0.0 1.0], 5, 1)
        acq = Acquisition(bvals, bvecs)

        # --- G1Ball (lambda_iso = 1.5e-9) ---
        ball = G1Ball(lambda_iso=1.5e-9)
        sig_ball = signal(ball, acq)
        expected_ball = [
            1.0000000000,
            0.4723665527,
            0.2231301601,
            0.0497870684,
            0.0111089965,
        ]
        @testset "G1Ball" begin
            for i in 1:5
                @test sig_ball[i] ≈ expected_ball[i] atol=1e-10
            end
        end

        # --- C1Stick (mu = [0,0,1], lambda_par = 1.7e-9) ---
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        sig_stick = signal(stick, acq)
        expected_stick = [
            1.0000000000,
            0.4274149319,
            0.1826835241,
            0.0333732700,
            0.0060967466,
        ]
        @testset "C1Stick" begin
            for i in 1:5
                @test sig_stick[i] ≈ expected_stick[i] atol=1e-10
            end
        end

        # --- G2Zeppelin (mu = [0,0,1], lambda_par = 1.7e-9, lambda_perp = 0.5e-9) ---
        # With gradient along mu, the perpendicular diffusivity does not
        # contribute, so this equals the Stick signal at these directions.
        zep = G2Zeppelin(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9, lambda_perp=0.5e-9)
        sig_zep = signal(zep, acq)
        expected_zep = [
            1.0000000000,
            0.4274149319,
            0.1826835241,
            0.0333732700,
            0.0060967466,
        ]
        @testset "G2Zeppelin" begin
            for i in 1:5
                @test sig_zep[i] ≈ expected_zep[i] atol=1e-10
            end
        end

        # --- S1Dot (no diffusion) ---
        sig_dot = signal(S1Dot(), acq)
        @testset "S1Dot" begin
            for i in 1:5
                @test sig_dot[i] ≈ 1.0 atol=1e-15
            end
        end
    end

    # ================================================================
    # 2. Multi-compartment golden (Ball + Stick)
    # ================================================================
    @testset "Multi-compartment Ball+Stick golden" begin
        ball = G1Ball(lambda_iso=3e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        # params layout: [lambda_iso, mu_x, mu_y, mu_z, lambda_par, f_ball, f_stick]
        params = [3e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]

        bvals = [0.0, 500e6, 1000e6, 1500e6, 2000e6,
                 500e6, 1000e6, 2000e6, 3000e6, 1000e6]
        dirs = [0.0 0.0 1.0;    # parallel
                0.0 0.0 1.0;    # parallel
                0.0 0.0 1.0;    # parallel
                0.0 0.0 1.0;    # parallel
                0.0 0.0 1.0;    # parallel
                1.0 0.0 0.0;    # perpendicular
                1.0 0.0 0.0;    # perpendicular
                1.0 0.0 0.0;    # perpendicular
                1.0 0.0 0.0;    # perpendicular
                sqrt(0.5) 0.0 sqrt(0.5)]  # oblique 45 deg
        acq = Acquisition(bvals, dirs)
        sig = signal(mcm, acq, params)

        expected = [
            1.0000000000,
            0.3661295004,
            0.1428145873,
            0.0579898652,
            0.0241049146,
            0.7669390480,
            0.7149361205,
            0.7007436257,
            0.7000370229,
            0.3141265729,
        ]
        for i in 1:10
            @test sig[i] ≈ expected[i] atol=1e-10
        end
    end

    # ================================================================
    # 3. NODDI-Watson golden
    # ================================================================
    @testset "NODDI-Watson golden" begin
        model = noddi_watson()

        # Free params: [mu_x, mu_y, mu_z, kappa, f_intra, f_iso]
        free_params = [0.0, 0.0, 1.0, 10.0, 0.5, 0.1]

        bvals = [0.0, 1000e6, 2000e6, 3000e6, 1000e6]
        bvecs = [0.0 0.0 1.0;
                 0.0 0.0 1.0;
                 0.0 0.0 1.0;
                 0.0 0.0 1.0;
                 1.0 0.0 0.0]
        acq = Acquisition(bvals, bvecs)

        sig = signal(model, acq, free_params)

        expected = [
            1.0000000000,
            0.1972276678,
            0.0429821001,
            0.0101915617,
            0.6285550521,
        ]
        # Watson uses numerical integration on a sphere grid, so allow
        # slightly larger tolerance.
        for i in 1:5
            @test sig[i] ≈ expected[i] atol=0.01
        end
    end

    # ================================================================
    # 4. RestrictedCylinder golden
    # ================================================================
    @testset "RestrictedCylinder golden" begin
        cyl = RestrictedCylinder(
            mu=[0.0, 0.0, 1.0],
            lambda_par=1.7e-9,
            diameter=4e-6,
        )
        bvals = [0.0, 1000e6, 3000e6]
        bvecs = [1.0 0.0 0.0;
                 1.0 0.0 0.0;
                 1.0 0.0 0.0]
        acq = Acquisition(bvals, bvecs, 10e-3, 30e-3)

        sig = signal(cyl, acq)

        expected = [
            1.0000000000,
            0.9630808393,
            0.8926373184,
        ]
        for i in 1:3
            @test sig[i] ≈ expected[i] atol=1e-10
        end
    end

    # ================================================================
    # 5. SphereGPD golden
    # ================================================================
    @testset "SphereGPD golden" begin
        sphere = SphereGPD(diameter=10e-6, D_intra=1e-9)
        bvals = [0.0, 1000e6, 3000e6]
        bvecs = [0.0 0.0 1.0;
                 0.0 0.0 1.0;
                 0.0 0.0 1.0]
        acq = Acquisition(bvals, bvecs, 10e-3, 30e-3)

        sig = signal(sphere, acq)

        expected = [
            1.0000000000,
            0.9999999999,
            0.9999999997,
        ]
        for i in 1:3
            @test sig[i] ≈ expected[i] atol=1e-9
        end
    end

    # ================================================================
    # 6. fit_mcm regression — fit Ball+Stick, verify reconstruction
    # ================================================================
    @testset "fit_mcm regression" begin
        ball = G1Ball(lambda_iso=3e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        params_true = [3e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]

        bvals = [0.0, 500e6, 1000e6, 1500e6, 2000e6,
                 500e6, 1000e6, 2000e6, 3000e6, 1000e6]
        dirs = [0.0 0.0 1.0;
                0.0 0.0 1.0;
                0.0 0.0 1.0;
                0.0 0.0 1.0;
                0.0 0.0 1.0;
                1.0 0.0 0.0;
                1.0 0.0 0.0;
                1.0 0.0 0.0;
                1.0 0.0 0.0;
                sqrt(0.5) 0.0 sqrt(0.5)]
        acq = Acquisition(bvals, dirs)

        data = signal(mcm, acq, params_true)

        result = fit_mcm(mcm, acq, data)
        p = result[:parameters]

        # Volume fractions should be recovered accurately
        @test p[6] ≈ 0.3 rtol=0.05   # f_ball
        @test p[7] ≈ 0.7 rtol=0.05   # f_stick

        # lambda_iso should be recovered accurately
        @test p[1] ≈ 3e-9 rtol=0.05

        # The reconstructed signal from fitted params must match the data
        sig_fit = signal(mcm, acq, p)
        for i in 1:length(data)
            @test sig_fit[i] ≈ data[i] atol=1e-6
        end
    end
end
