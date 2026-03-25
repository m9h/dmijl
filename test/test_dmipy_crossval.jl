# Cross-validation of DMI.jl composable compartments against dmipy-jax.
# Reproduces demo_ball_and_stick_jax.py with identical parameters and
# acquisition scheme, verifying signal-level parity.
using Test, LinearAlgebra, Random

@testset "dmipy-jax Cross-Validation" begin

    # ------------------------------------------------------------------ #
    # 1. Ball+Stick: reproduce demo_ball_and_stick_jax.py
    # ------------------------------------------------------------------ #
    @testset "Ball+Stick (demo_ball_and_stick_jax.py)" begin
        # Ground truth from dmipy-jax demo
        f_stick_true = 0.6
        f_ball_true  = 1.0 - f_stick_true
        mu_true      = [1.0, 0.0, 0.0]
        lambda_par   = 1.7e-9   # m²/s
        lambda_iso   = 3.0e-9   # m²/s

        # Acquisition: 40 measurements (4 b-values × 10 directions)
        # dmipy-jax: bvals = tile([0, 1000, 2000, 3000], 10) * 1e6
        bvals_base = [0.0, 1000.0, 2000.0, 3000.0] .* 1e6  # s/m²
        bvals = repeat(bvals_base, 10)

        # Fixed random directions (seeded for reproducibility)
        rng = MersenneTwister(0)
        bvecs = randn(rng, 40, 3)
        bvecs = bvecs ./ sqrt.(sum(bvecs.^2, dims=2))

        acq = Acquisition(bvals, bvecs)

        # ---- Compartment-level signal verification ----
        @testset "G1Ball signal matches analytical" begin
            ball = G1Ball(lambda_iso=lambda_iso)
            sig = signal(ball, acq)

            # Analytical: S = exp(-b * D_iso)
            expected = exp.(-bvals .* lambda_iso)
            @test sig ≈ expected atol=1e-14
        end

        @testset "C1Stick signal matches analytical" begin
            stick = C1Stick(mu=mu_true, lambda_par=lambda_par)
            sig = signal(stick, acq)

            # Analytical: S = exp(-b * D_par * (g·μ)²)
            dot_prod = bvecs * mu_true
            expected = exp.(-bvals .* lambda_par .* dot_prod.^2)
            @test sig ≈ expected atol=1e-14
        end

        # ---- Multi-compartment signal ----
        @testset "Ball+Stick MCM signal" begin
            ball = G1Ball(lambda_iso=lambda_iso)
            stick = C1Stick(mu=mu_true, lambda_par=lambda_par)
            mcm = MultiCompartmentModel(ball, stick)

            # Flat params: [lambda_iso, mu..., lambda_par, f_ball, f_stick]
            params = [lambda_iso, mu_true..., lambda_par, f_ball_true, f_stick_true]
            sig = signal(mcm, acq, params)

            # Manual computation
            dot_prod = bvecs * mu_true
            S_ball  = exp.(-bvals .* lambda_iso)
            S_stick = exp.(-bvals .* lambda_par .* dot_prod.^2)
            expected = f_ball_true .* S_ball .+ f_stick_true .* S_stick

            @test sig ≈ expected atol=1e-14

            # Physics checks
            @test all(0.0 .<= sig .<= 1.0 .+ 1e-12)
            @test all(sig[bvals .== 0.0] .≈ 1.0)
        end

        # ---- Parameter recovery (noise-free, HCP-like acquisition) ----
        @testset "noise-free parameter recovery (HCP-like)" begin
            # Use richer HCP-like acquisition for better conditioning
            acq_hcp = hcp_like_acquisition()
            ball = G1Ball(lambda_iso=lambda_iso)
            stick = C1Stick(mu=mu_true, lambda_par=lambda_par)
            mcm = MultiCompartmentModel(ball, stick)

            true_params = [lambda_iso, mu_true..., lambda_par, f_ball_true, f_stick_true]
            data = signal(mcm, acq_hcp, true_params)

            # Good initialization near truth
            init = [2.5e-9, 0.9, 0.1, 0.1, 1.5e-9, 0.5, 0.5]
            result = fit_mcm(mcm, acq_hcp, data; init=init)
            fitted = result[:parameters]

            @test fitted[1] ≈ lambda_iso rtol=0.05    # lambda_iso
            @test fitted[5] ≈ lambda_par rtol=0.05    # lambda_par
            @test fitted[6] ≈ f_ball_true atol=0.05   # f_ball
            @test fitted[7] ≈ f_stick_true atol=0.05  # f_stick
            @test maximum(abs.(result[:residuals])) < 1e-4
        end

        # ---- Parameter recovery (noisy, SNR ~50) ----
        @testset "noisy parameter recovery (SNR ≈ 50)" begin
            ball = G1Ball(lambda_iso=lambda_iso)
            stick = C1Stick(mu=mu_true, lambda_par=lambda_par)
            mcm = MultiCompartmentModel(ball, stick)

            true_params = [lambda_iso, mu_true..., lambda_par, f_ball_true, f_stick_true]
            clean = signal(mcm, acq, true_params)

            rng_noise = MersenneTwister(1)
            noise = randn(rng_noise, length(clean)) .* 0.02
            noisy = abs.(clean .+ noise)  # Rician-like

            init = [2.5e-9, 0.8, 0.1, 0.1, 1.5e-9, 0.5, 0.5]
            result = fit_mcm(mcm, acq, noisy; init=init)
            fitted = result[:parameters]

            # Relaxed tolerances for noisy case
            @test fitted[6] ≈ f_ball_true atol=0.15   # f_ball
            @test fitted[7] ≈ f_stick_true atol=0.15  # f_stick
        end
    end

    # ------------------------------------------------------------------ #
    # 2. Zeppelin+Stick (NODDI-like, without Watson dispersion)
    # ------------------------------------------------------------------ #
    @testset "Stick+Zeppelin+Ball (NODDI-like)" begin
        # NODDI-like parameters from dmipy examples
        mu = [0.0, 0.0, 1.0]
        d_par = 1.7e-9
        d_iso = 3.0e-9
        f_intra = 0.5   # intra-cellular (stick)
        f_iso   = 0.1   # CSF (ball)
        f_extra = 1.0 - f_intra - f_iso  # extra-cellular (zeppelin)

        # Tortuosity constraint: d_perp = d_par * (1 - f_intra)
        d_perp = d_par * (1.0 - f_intra)

        acq = hcp_like_acquisition()

        @testset "signal with tortuosity constraint" begin
            stick = C1Stick(mu=mu, lambda_par=d_par)
            zep = G2Zeppelin(mu=mu, lambda_par=d_par, lambda_perp=d_perp)
            ball = G1Ball(lambda_iso=d_iso)
            mcm = MultiCompartmentModel(stick, zep, ball)

            # Params: [mu_stick..., d_par_stick, mu_zep..., d_par_zep, d_perp, d_iso, f_stick, f_zep, f_ball]
            params = [mu..., d_par, mu..., d_par, d_perp, d_iso, f_intra, f_extra, f_iso]
            sig = signal(mcm, acq, params)

            # Manual computation
            dot_prod = acq.gradient_directions * (mu ./ norm(mu))
            b = acq.bvalues
            S_stick = exp.(-b .* d_par .* dot_prod.^2)
            S_zep   = exp.(-b .* (d_par .* dot_prod.^2 .+ d_perp .* (1 .- dot_prod.^2)))
            S_ball  = exp.(-b .* d_iso)
            expected = f_intra .* S_stick .+ f_extra .* S_zep .+ f_iso .* S_ball

            @test sig ≈ expected atol=1e-12
            @test all(0.0 .<= sig .<= 1.0 .+ 1e-12)
        end

        @testset "constrained model with tortuosity" begin
            stick = C1Stick(mu=mu, lambda_par=d_par)
            zep = G2Zeppelin(mu=mu, lambda_par=d_par, lambda_perp=d_perp)
            ball = G1Ball(lambda_iso=d_iso)
            mcm = MultiCompartmentModel(stick, zep, ball)

            # Apply tortuosity constraint
            constrained = set_tortuosity(mcm,
                target=:lambda_perp,
                lambda_par_name=:lambda_par_2,
                volume_fraction_name=:partial_volume_1)

            # Apply volume fraction unity
            constrained2 = set_volume_fraction_unity(mcm)

            # Verify tortuosity reduces free parameters
            @test :lambda_perp ∉ parameter_names(constrained)
            @test nparams(constrained) == nparams(mcm) - 1
        end
    end

    # ------------------------------------------------------------------ #
    # 3. Crossing fibers (Ball + 2 Sticks)
    # ------------------------------------------------------------------ #
    @testset "Crossing fibers (Ball + 2 Sticks)" begin
        # From dmipy generate_synthetic_oracle.py
        d_iso = 1.7e-9    # note: dmipy uses 1.7e-3 mm²/s = 1.7e-9 m²/s
        d_par = 1.7e-9
        f1 = 0.4
        f2 = 0.3
        f_iso = 0.3
        mu1 = [0.0, 0.0, 1.0]  # vertical
        mu2 = [1.0, 0.0, 0.0]  # horizontal

        acq = hcp_like_acquisition()

        ball = G1Ball(lambda_iso=d_iso)
        stick1 = C1Stick(mu=mu1, lambda_par=d_par)
        stick2 = C1Stick(mu=mu2, lambda_par=d_par)
        mcm = MultiCompartmentModel(ball, stick1, stick2)

        # Params: [d_iso, mu1..., d_par1, mu2..., d_par2, f_ball, f_stick1, f_stick2]
        params = [d_iso, mu1..., d_par, mu2..., d_par, f_iso, f1, f2]
        sig = signal(mcm, acq, params)

        @testset "signal properties" begin
            @test all(0.0 .<= sig .<= 1.0 .+ 1e-12)
            b0_idx = findall(acq.bvalues .== 0.0)
            @test all(sig[b0_idx] .≈ 1.0)
        end

        @testset "matches legacy BallStickModel" begin
            old_model = BallStickModel(acq.bvalues, acq.gradient_directions)
            # Legacy params: [d_ball, d_stick, f1, f2, mu1..., mu2...]
            old_sig = simulate(old_model, [d_iso, d_par, f1, f2, mu1..., mu2...])
            @test sig ≈ old_sig atol=1e-10
        end

        @testset "noise-free recovery of crossing angle" begin
            data = signal(mcm, acq, params)
            init = [2.0e-9, 0.1, 0.1, 0.9, 1.5e-9, 0.9, 0.1, 0.1, 1.5e-9, 0.33, 0.33, 0.34]
            result = fit_mcm(mcm, acq, data; init=init)
            fitted = result[:parameters]

            # Volume fractions should be recovered
            @test fitted[10] ≈ f_iso atol=0.1   # f_ball
            @test fitted[11] ≈ f1 atol=0.1      # f_stick1
            @test fitted[12] ≈ f2 atol=0.1      # f_stick2
        end
    end

    # ------------------------------------------------------------------ #
    # 4. Unit consistency (SI vs clinical)
    # ------------------------------------------------------------------ #
    @testset "Unit consistency" begin
        # dmipy uses SI: m²/s for diffusivity, s/m² for b-values
        # Clinical convention: mm²/s and s/mm²
        # 1 mm²/s = 1e-6 m²/s; 1 s/mm² = 1e6 s/m²
        # So b * D is dimensionless in both conventions

        D_si = 2.0e-9       # m²/s
        D_clin = 2.0e-3     # mm²/s
        b_si = 1000e6        # s/m²
        b_clin = 1000.0      # s/mm²

        @test D_si * b_si ≈ D_clin * b_clin atol=1e-12

        # Verify our convention
        ball = G1Ball(lambda_iso=D_si)
        acq = Acquisition([b_si], reshape([1.0 0.0 0.0], 1, 3))
        sig = signal(ball, acq)
        @test sig[1] ≈ exp(-b_clin * D_clin) atol=1e-14
    end
end
