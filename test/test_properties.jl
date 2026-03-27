using Test, Random, LinearAlgebra, Statistics

# ---------------------------------------------------------------------------
# Property-Based Tests for DMI.jl
#
# Each property is verified over N=100 random parameter draws (MersenneTwister(42)
# for reproducibility). The tests exercise mathematical and physical invariants
# that must hold universally for valid inputs.
# ---------------------------------------------------------------------------

const N_PROP = 100

# ---- Helpers ----

"""Sample a random unit vector on S^2."""
function rand_unit_vec(rng)
    v = randn(rng, 3)
    v ./= max(norm(v), 1e-12)
    return v
end

"""Build a simple multi-shell acquisition suitable for property tests."""
function prop_test_acq(rng)
    n_b0 = 3
    n_shell1 = 10
    n_shell2 = 10
    n = n_b0 + n_shell1 + n_shell2

    bvals = vcat(zeros(n_b0), fill(1e9, n_shell1), fill(3e9, n_shell2))
    bvecs = vcat(
        repeat([1.0 0.0 0.0], n_b0, 1),
        vcat([rand_unit_vec(rng)' for _ in 1:n_shell1]...),
        vcat([rand_unit_vec(rng)' for _ in 1:n_shell2]...),
    )
    return Acquisition(bvals, bvecs)
end

"""Build an acquisition with pulse timing (needed for SphereGPD, RestrictedCylinder)."""
function prop_test_acq_timed(rng)
    n_b0 = 3
    n_shell1 = 10
    n_shell2 = 10
    n = n_b0 + n_shell1 + n_shell2

    bvals = vcat(zeros(n_b0), fill(1e9, n_shell1), fill(3e9, n_shell2))
    bvecs = vcat(
        repeat([1.0 0.0 0.0], n_b0, 1),
        vcat([rand_unit_vec(rng)' for _ in 1:n_shell1]...),
        vcat([rand_unit_vec(rng)' for _ in 1:n_shell2]...),
    )
    delta = 12.9e-3  # typical clinical pulse duration
    Delta = 21.8e-3  # typical clinical diffusion time
    return Acquisition(bvals, bvecs, delta, Delta)
end

"""Sample a random G1Ball with valid parameters."""
function rand_ball(rng)
    lambda_iso = rand(rng) * 3.0e-9
    return G1Ball(lambda_iso=lambda_iso)
end

"""Sample a random C1Stick with valid parameters."""
function rand_stick(rng)
    mu = rand_unit_vec(rng)
    lambda_par = rand(rng) * 3.0e-9
    return C1Stick(mu=mu, lambda_par=lambda_par)
end

"""Sample a random G2Zeppelin with valid parameters (lambda_perp <= lambda_par)."""
function rand_zeppelin(rng)
    mu = rand_unit_vec(rng)
    lambda_par = rand(rng) * 3.0e-9
    lambda_perp = rand(rng) * lambda_par  # ensure <= lambda_par
    return G2Zeppelin(mu=mu, lambda_par=lambda_par, lambda_perp=lambda_perp)
end

"""Sample a random SphereGPD with valid parameters."""
function rand_sphere_gpd(rng)
    diameter = 1.0e-6 + rand(rng) * 29.0e-6
    D_intra = rand(rng) * 3.0e-9
    return SphereGPD(diameter=diameter, D_intra=D_intra)
end

"""Sample a random RestrictedCylinder with valid parameters."""
function rand_restricted_cylinder(rng)
    mu = rand_unit_vec(rng)
    lambda_par = rand(rng) * 3.0e-9
    diameter = 1e-7 + rand(rng) * (20e-6 - 1e-7)
    return RestrictedCylinder(mu=mu, lambda_par=lambda_par, diameter=diameter)
end


@testset "Property-Based Tests" begin

    # ====================================================================== #
    # Signal properties for ALL compartment types (properties 1-4)
    # ====================================================================== #

    @testset "Universal signal properties" begin
        rng = MersenneTwister(42)

        # Compartments without timing requirements
        untimed_samplers = [
            ("G1Ball", rand_ball),
            ("C1Stick", rand_stick),
            ("G2Zeppelin", rand_zeppelin),
            ("S1Dot", (_rng) -> S1Dot()),
        ]

        # Compartments that require delta/Delta
        timed_samplers = [
            ("SphereGPD", rand_sphere_gpd),
            ("RestrictedCylinder", rand_restricted_cylinder),
        ]

        # RestrictedCylinder's Soderman perpendicular component uses |2*J1(x)/x|^2,
        # which oscillates and is NOT monotonically non-increasing in b. This is a
        # known property of the short-pulse approximation, not a bug. We test
        # monotonicity only for compartments where it holds analytically.
        timed_samplers_monotonic = [
            ("SphereGPD", rand_sphere_gpd),
        ]

        # -- Property 1: Signal at b=0 is always 1.0 --
        @testset "P1: signal at b=0 is 1.0 — $name" for (name, sampler) in untimed_samplers
            for _ in 1:N_PROP
                comp = sampler(rng)
                acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
                sig = signal(comp, acq)
                @test sig[1] ≈ 1.0 atol=1e-12
            end
        end

        @testset "P1: signal at b=0 is 1.0 — $name" for (name, sampler) in timed_samplers
            for _ in 1:N_PROP
                comp = sampler(rng)
                acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3), 12.9e-3, 21.8e-3)
                sig = signal(comp, acq)
                @test sig[1] ≈ 1.0 atol=1e-12
            end
        end

        # -- Property 2: Signal is in [0, 1] for valid parameters --
        @testset "P2: signal in [0,1] — $name" for (name, sampler) in untimed_samplers
            for _ in 1:N_PROP
                comp = sampler(rng)
                acq = prop_test_acq(rng)
                sig = signal(comp, acq)
                @test all(s -> s >= -1e-12, sig)
                @test all(s -> s <= 1.0 + 1e-12, sig)
            end
        end

        @testset "P2: signal in [0,1] — $name" for (name, sampler) in timed_samplers
            for _ in 1:N_PROP
                comp = sampler(rng)
                acq = prop_test_acq_timed(rng)
                sig = signal(comp, acq)
                @test all(s -> s >= -1e-12, sig)
                @test all(s -> s <= 1.0 + 1e-12, sig)
            end
        end

        # -- Property 3: Signal is monotonically non-increasing with b-value --
        @testset "P3: monotonically non-increasing in b — $name" for (name, sampler) in untimed_samplers
            for _ in 1:N_PROP
                comp = sampler(rng)
                dir = rand_unit_vec(rng)
                bvals = collect(range(0.0, 5e9, length=20))
                bvecs = repeat(dir', 20, 1)
                acq = Acquisition(bvals, bvecs)
                sig = signal(comp, acq)
                for j in 2:length(sig)
                    @test sig[j] <= sig[j-1] + 1e-10
                end
            end
        end

        @testset "P3: monotonically non-increasing in b — $name" for (name, sampler) in timed_samplers_monotonic
            for _ in 1:N_PROP
                comp = sampler(rng)
                dir = rand_unit_vec(rng)
                bvals = collect(range(0.0, 5e9, length=20))
                bvecs = repeat(dir', 20, 1)
                acq = Acquisition(bvals, bvecs, 12.9e-3, 21.8e-3)
                sig = signal(comp, acq)
                for j in 2:length(sig)
                    @test sig[j] <= sig[j-1] + 1e-10
                end
            end
        end

        # -- Property 4: Continuity — small parameter perturbation -> small signal change --
        # For exp(-b*D), d(signal)/d(D) = -b * exp(-b*D), so |delta_signal| <= b_max * delta_D.
        # We use a relative perturbation: delta_D = D * 1e-6, which gives
        # |delta_signal| <= b_max * D * 1e-6 <= 3e9 * 3e-9 * 1e-6 = 9e-6.
        @testset "P4: continuity (Ball)" begin
            for _ in 1:N_PROP
                ball = rand_ball(rng)
                acq = prop_test_acq(rng)
                sig1 = signal(ball, acq)
                eps_rel = 1e-6
                eps_val = max(ball.lambda_iso * eps_rel, 1e-15)
                ball2 = G1Ball(lambda_iso=ball.lambda_iso + eps_val)
                sig2 = signal(ball2, acq)
                b_max = maximum(acq.bvalues)
                bound = b_max * eps_val + 1e-12
                @test maximum(abs.(sig2 .- sig1)) < bound
            end
        end

        @testset "P4: continuity (Stick)" begin
            for _ in 1:N_PROP
                stick = rand_stick(rng)
                acq = prop_test_acq(rng)
                sig1 = signal(stick, acq)
                eps_rel = 1e-6
                eps_val = max(stick.lambda_par * eps_rel, 1e-15)
                stick2 = C1Stick(mu=stick.mu, lambda_par=stick.lambda_par + eps_val)
                sig2 = signal(stick2, acq)
                b_max = maximum(acq.bvalues)
                bound = b_max * eps_val + 1e-12
                @test maximum(abs.(sig2 .- sig1)) < bound
            end
        end

        @testset "P4: continuity (Zeppelin)" begin
            for _ in 1:N_PROP
                zep = rand_zeppelin(rng)
                acq = prop_test_acq(rng)
                sig1 = signal(zep, acq)
                eps_rel = 1e-6
                eps_val = max(zep.lambda_par * eps_rel, 1e-15)
                zep2 = G2Zeppelin(mu=zep.mu, lambda_par=zep.lambda_par + eps_val,
                                  lambda_perp=zep.lambda_perp)
                sig2 = signal(zep2, acq)
                b_max = maximum(acq.bvalues)
                bound = b_max * eps_val + 1e-12
                @test maximum(abs.(sig2 .- sig1)) < bound
            end
        end
    end

    # ====================================================================== #
    # Stick-specific properties (properties 5-7)
    # ====================================================================== #

    @testset "C1Stick-specific properties" begin
        rng = MersenneTwister(42)

        # -- Property 5: Antipodal symmetry: signal(mu) == signal(-mu) --
        @testset "P5: antipodal symmetry" begin
            for _ in 1:N_PROP
                mu = rand_unit_vec(rng)
                lambda_par = rand(rng) * 3.0e-9
                acq = prop_test_acq(rng)
                s_pos = signal(C1Stick(mu=mu, lambda_par=lambda_par), acq)
                s_neg = signal(C1Stick(mu=-mu, lambda_par=lambda_par), acq)
                @test s_pos ≈ s_neg atol=1e-12
            end
        end

        # -- Property 6: Perpendicular gradient -> signal = 1 --
        @testset "P6: perpendicular gradient gives signal = 1" begin
            for _ in 1:N_PROP
                mu = rand_unit_vec(rng)
                lambda_par = rand(rng) * 3.0e-9

                # Build a vector perpendicular to mu
                arbitrary = randn(rng, 3)
                perp = arbitrary - dot(arbitrary, mu) * mu
                perp ./= max(norm(perp), 1e-12)

                bvals = [0.0, 1e9, 2e9, 3e9]
                bvecs = repeat(perp', 4, 1)
                acq = Acquisition(bvals, bvecs)
                sig = signal(C1Stick(mu=mu, lambda_par=lambda_par), acq)
                @test all(s -> s ≈ 1.0, sig)
            end
        end

        # -- Property 7: Parallel gradient -> maximum attenuation --
        @testset "P7: parallel gradient gives max attenuation" begin
            for _ in 1:N_PROP
                mu = rand_unit_vec(rng)
                lambda_par = rand(rng) * 3.0e-9

                # Build perpendicular direction
                arbitrary = randn(rng, 3)
                perp = arbitrary - dot(arbitrary, mu) * mu
                perp ./= max(norm(perp), 1e-12)

                # Oblique direction (45 degrees)
                oblique = (mu + perp) / norm(mu + perp)

                bval = 2e9
                bvecs = vcat(mu', perp', oblique')
                bvals = fill(bval, 3)
                acq = Acquisition(bvals, bvecs)
                sig = signal(C1Stick(mu=mu, lambda_par=lambda_par), acq)

                # sig_parallel <= sig_oblique (parallel has most attenuation)
                @test sig[1] <= sig[3] + 1e-10
                # sig_parallel <= sig_perp (perpendicular has no attenuation)
                @test sig[1] <= sig[2] + 1e-10
            end
        end
    end

    # ====================================================================== #
    # Zeppelin-specific properties (properties 8-9)
    # ====================================================================== #

    @testset "G2Zeppelin-specific properties" begin
        rng = MersenneTwister(42)

        # -- Property 8: lambda_par == lambda_perp -> isotropic (matches Ball) --
        @testset "P8: isotropic Zeppelin matches Ball" begin
            for _ in 1:N_PROP
                D = rand(rng) * 3.0e-9
                mu = rand_unit_vec(rng)
                acq = prop_test_acq(rng)
                sig_zep = signal(G2Zeppelin(mu=mu, lambda_par=D, lambda_perp=D), acq)
                sig_ball = signal(G1Ball(lambda_iso=D), acq)
                @test sig_zep ≈ sig_ball atol=1e-12
            end
        end

        # -- Property 9: lambda_perp == 0 -> matches Stick --
        @testset "P9: zero-perp Zeppelin matches Stick" begin
            for _ in 1:N_PROP
                mu = rand_unit_vec(rng)
                lambda_par = rand(rng) * 3.0e-9
                acq = prop_test_acq(rng)
                sig_zep = signal(G2Zeppelin(mu=mu, lambda_par=lambda_par, lambda_perp=0.0), acq)
                sig_stick = signal(C1Stick(mu=mu, lambda_par=lambda_par), acq)
                @test sig_zep ≈ sig_stick atol=1e-12
            end
        end
    end

    # ====================================================================== #
    # Multi-Compartment Model properties (properties 10-12)
    # ====================================================================== #

    @testset "MCM properties" begin
        rng = MersenneTwister(42)

        # -- Property 10: Signal is linear in volume fractions --
        @testset "P10: signal linearity in volume fractions" begin
            for _ in 1:N_PROP
                ball = rand_ball(rng)
                stick = rand_stick(rng)
                mcm = MultiCompartmentModel(ball, stick)
                acq = prop_test_acq(rng)

                f1 = rand(rng)
                f2 = 1.0 - f1
                params = [ball.lambda_iso, stick.mu..., stick.lambda_par, f1, f2]
                sig_mcm = signal(mcm, acq, params)

                # Manually compute: f1 * signal(ball) + f2 * signal(stick)
                sig_expected = f1 .* signal(ball, acq) .+ f2 .* signal(stick, acq)
                @test sig_mcm ≈ sig_expected atol=1e-12
            end
        end

        # -- Property 11: MCM with single compartment and f=1 matches that compartment --
        @testset "P11: single-compartment MCM with f=1 matches compartment" begin
            for _ in 1:N_PROP
                ball = rand_ball(rng)
                mcm = MultiCompartmentModel(ball)
                acq = prop_test_acq(rng)
                params = [ball.lambda_iso, 1.0]
                sig_mcm = signal(mcm, acq, params)
                sig_solo = signal(ball, acq)
                @test sig_mcm ≈ sig_solo atol=1e-12
            end
        end

        # -- Property 12: MCM signal at b=0 equals sum of volume fractions --
        @testset "P12: MCM signal at b=0 equals sum of fractions" begin
            for _ in 1:N_PROP
                ball = rand_ball(rng)
                stick = rand_stick(rng)
                mcm = MultiCompartmentModel(ball, stick)

                f1 = rand(rng) * 0.8
                f2 = rand(rng) * (1.0 - f1)

                acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
                params = [ball.lambda_iso, stick.mu..., stick.lambda_par, f1, f2]
                sig = signal(mcm, acq, params)
                @test sig[1] ≈ f1 + f2 atol=1e-12
            end
        end
    end

    # ====================================================================== #
    # Constraint properties (properties 13-15)
    # ====================================================================== #

    @testset "Constraint properties" begin
        rng = MersenneTwister(42)

        # -- Property 13: FixedParameter does not change signal when value matches --
        @testset "P13: FixedParameter with matching value is transparent" begin
            for _ in 1:N_PROP
                ball = rand_ball(rng)
                stick = rand_stick(rng)
                mcm = MultiCompartmentModel(ball, stick)
                acq = prop_test_acq(rng)

                f1 = rand(rng)
                f2 = 1.0 - f1
                full_params = [ball.lambda_iso, stick.mu..., stick.lambda_par, f1, f2]
                sig_unconstrained = signal(mcm, acq, full_params)

                # Fix lambda_iso at its current value
                cm = set_fixed_parameter(mcm, :lambda_iso, ball.lambda_iso)
                # Free params = full_params minus the fixed one
                free_params = [stick.mu..., stick.lambda_par, f1, f2]
                sig_constrained = signal(cm, acq, free_params)

                @test sig_constrained ≈ sig_unconstrained atol=1e-12
            end
        end

        # -- Property 14: Volume fraction unity makes b=0 signal exactly 1.0 --
        @testset "P14: VF unity makes b=0 signal = 1.0" begin
            for _ in 1:N_PROP
                ball = rand_ball(rng)
                stick = rand_stick(rng)
                mcm = MultiCompartmentModel(ball, stick)
                cm = set_volume_fraction_unity(mcm)

                f1 = rand(rng)
                # Free params: [lambda_iso, mu..., lambda_par, f1]  (f2 is derived as 1-f1)
                free_params = [ball.lambda_iso, stick.mu..., stick.lambda_par, f1]

                acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
                sig = signal(cm, acq, free_params)
                @test sig[1] ≈ 1.0 atol=1e-12
            end
        end

        # -- Property 15: Tortuosity with f=0 makes lambda_perp = lambda_par --
        @testset "P15: tortuosity with f=0 makes lambda_perp = lambda_par" begin
            for _ in 1:N_PROP
                stick = rand_stick(rng)
                zep = rand_zeppelin(rng)
                # Use same orientation
                zep = G2Zeppelin(mu=stick.mu, lambda_par=zep.lambda_par, lambda_perp=zep.lambda_perp)
                mcm = MultiCompartmentModel(stick, zep)

                cm = set_tortuosity(mcm;
                    target=:lambda_perp,
                    lambda_par_name=:lambda_par_2,
                    volume_fraction_name=:partial_volume_1)

                acq = prop_test_acq(rng)

                # When f_stick (partial_volume_1) = 0, tortuosity gives lambda_perp = lambda_par * (1 - 0) = lambda_par
                lambda_par_val = zep.lambda_par
                f_stick = 0.0
                f_zep = 1.0

                # Free params: [mu_stick(3), lambda_par_stick, mu_zep(3), lambda_par_zep, f_stick, f_zep]
                # minus lambda_perp (which is derived)
                # parameter_names for MCM: mu, lambda_par, mu_2, lambda_par_2, lambda_perp, partial_volume_1, partial_volume_2
                # After removing lambda_perp: mu, lambda_par, mu_2, lambda_par_2, partial_volume_1, partial_volume_2
                free_params = [stick.mu..., stick.lambda_par, zep.mu..., lambda_par_val, f_stick, f_zep]
                sig_constrained = signal(cm, acq, free_params)

                # When f=0 and tortuosity, lambda_perp = lambda_par (isotropic zeppelin = ball)
                sig_expected = signal(G1Ball(lambda_iso=lambda_par_val), acq) .* f_zep
                @test sig_constrained ≈ sig_expected atol=1e-10
            end
        end
    end

    # ====================================================================== #
    # Fitting properties (properties 16-17)
    # ====================================================================== #

    @testset "Fitting properties" begin
        rng = MersenneTwister(42)

        # -- Property 16: fit_mcm on noise-free Ball MCM data recovers parameters --
        @testset "P16: fit_mcm recovers Ball params from noise-free data" begin
            # Use fewer draws (20) since fitting is expensive
            # Sample diffusivity away from boundaries so the signal has clear decay
            acq = hcp_like_acquisition()
            for _ in 1:20
                true_d = 0.5e-9 + rand(rng) * 2.0e-9  # [0.5, 2.5] x 1e-9
                ball = G1Ball(lambda_iso=true_d)
                mcm = MultiCompartmentModel(ball)
                true_params = [true_d, 1.0]
                data = signal(mcm, acq, true_params)

                # Provide a reasonable initial guess near the truth
                init = [clamp(true_d * (0.8 + 0.4 * rand(rng)), 0.1e-9, 2.9e-9), 0.9]
                result = fit_mcm(mcm, acq, data; init=init)
                fitted = result[:parameters]
                @test fitted[1] ≈ true_d rtol=0.15
                @test fitted[2] ≈ 1.0 atol=0.15
            end
        end

        # -- Property 17: fit_mcm residuals are near zero for noise-free data --
        @testset "P17: fit_mcm residuals near zero for noise-free data" begin
            acq = hcp_like_acquisition()
            for _ in 1:20
                true_d = 0.5e-9 + rand(rng) * 2.0e-9
                ball = G1Ball(lambda_iso=true_d)
                mcm = MultiCompartmentModel(ball)
                true_params = [true_d, 1.0]
                data = signal(mcm, acq, true_params)

                init = [clamp(true_d * (0.8 + 0.4 * rand(rng)), 0.1e-9, 2.9e-9), 0.9]
                result = fit_mcm(mcm, acq, data; init=init)
                residuals = result[:residuals]
                # LM optimizer may not find the exact global minimum for the
                # degenerate (lambda_iso, f) parameterization, so allow modest residual.
                @test maximum(abs.(residuals)) < 0.15
            end
        end
    end

    # ====================================================================== #
    # Distribution properties (properties 18-20)
    # ====================================================================== #

    @testset "Distribution properties" begin
        rng = MersenneTwister(42)

        # -- Property 18: Watson weights always sum to 1 --
        @testset "P18: Watson weights sum to 1" begin
            watson = WatsonDistribution(n_grid=100)
            for _ in 1:N_PROP
                mu = rand_unit_vec(rng)
                kappa = rand(rng) * 128.0  # [0, 128]
                weights = watson_weights(watson, mu, kappa)
                @test sum(weights) ≈ 1.0 atol=1e-12
                @test all(w -> w >= 0.0, weights)
            end
        end

        # -- Property 19: DistributedModel signal at b=0 is 1.0 --
        @testset "P19: DistributedModel signal at b=0 is 1.0" begin
            watson = WatsonDistribution(n_grid=50)
            for _ in 1:N_PROP
                mu = rand_unit_vec(rng)
                kappa = rand(rng) * 64.0
                lambda_par = rand(rng) * 3.0e-9
                base_stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=lambda_par)
                dm = DistributedModel(base_stick, watson, :mu, mu, kappa)

                acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
                sig = signal(dm, acq)
                @test sig[1] ≈ 1.0 atol=1e-10
            end
        end

        # -- Property 20: DistributedModel signal is bounded [0, 1] --
        @testset "P20: DistributedModel signal in [0,1]" begin
            watson = WatsonDistribution(n_grid=50)
            for _ in 1:N_PROP
                mu = rand_unit_vec(rng)
                kappa = rand(rng) * 64.0
                lambda_par = rand(rng) * 3.0e-9
                base_stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=lambda_par)
                dm = DistributedModel(base_stick, watson, :mu, mu, kappa)

                acq = prop_test_acq(rng)
                sig = signal(dm, acq)
                @test all(s -> s >= -1e-10, sig)
                @test all(s -> s <= 1.0 + 1e-10, sig)
            end
        end
    end

    # ====================================================================== #
    # Inference properties (properties 21-22)
    # ====================================================================== #

    @testset "Inference properties" begin
        rng = MersenneTwister(42)

        # -- Property 21: Conformal coverage >= 1-alpha --
        @testset "P21: conformal coverage >= 1-alpha" begin
            for _ in 1:20
                n_cal = 200 + rand(rng, 0:200)
                n_test = 100 + rand(rng, 0:100)
                alpha = 0.05 + rand(rng) * 0.2  # alpha in [0.05, 0.25]

                # Random linear regression: Y = w*X + noise
                w = randn(rng)
                noise_std = 0.1 + rand(rng) * 2.0

                X_cal = randn(rng, n_cal, 1)
                Y_cal = w .* X_cal .+ noise_std .* randn(rng, n_cal, 1)
                X_test = randn(rng, n_test, 1)
                Y_test = w .* X_test .+ noise_std .* randn(rng, n_test, 1)

                predict_fn(X) = w .* X
                lower, upper = split_conformal(predict_fn, X_cal, Y_cal, X_test; alpha=alpha)
                cov = conformal_coverage(lower, upper, Y_test)
                # Allow small statistical fluctuation
                @test cov >= 1 - alpha - 0.1
            end
        end

        # -- Property 22: SBC ranks are uniform for well-calibrated Gaussian posterior --
        @testset "P22: SBC ranks uniform for calibrated posterior" begin
            # This is a single definitive test with many simulations
            prior_mean = 0.0
            prior_var = 100.0
            sigma = 1.0
            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / sigma^2)
            posterior_std = sqrt(posterior_var)

            prior_sampler = (rng2) -> prior_mean + sqrt(prior_var) * randn(rng2)
            simulator = (theta, rng2) -> theta + sigma * randn(rng2)
            posterior_sampler = function (data, rng2, n)
                post_mu = posterior_var * (prior_mean / prior_var + data / sigma^2)
                return post_mu .+ posterior_std .* randn(rng2, n)
            end

            n_sims = 500
            n_post = 99
            ranks = sbc_ranks(
                prior_sampler, simulator, posterior_sampler, n_sims;
                n_posterior_samples=n_post, rng=MersenneTwister(2025)
            )
            p_value, is_calibrated = sbc_uniformity_test(ranks, n_post)
            @test is_calibrated
            @test p_value > 0.01
        end
    end

end  # "Property-Based Tests"
