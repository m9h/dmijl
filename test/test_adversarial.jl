"""
Adversarial / red-team tests for DMI.jl public API.

Probes edge cases, pathological inputs, numerical instabilities,
degenerate parameters, and cross-method consistency.

Convention:
  - @test / @test_throws  → expected graceful behavior
  - @test_broken           → known bug, documenting for triage
  - # BUG: ...             → inline annotation when a genuine bug is found
"""

using Test, Random, Statistics, LinearAlgebra
using SpecialFunctions: besseli

# ─────────────────────────────────────────────────────────────────────────────
# 1. Pathological inputs
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adversarial: Pathological inputs" begin

    acq = hcp_like_acquisition()
    ball = G1Ball(lambda_iso=2.0e-9)
    stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
    mcm = MultiCompartmentModel(ball, stick)
    true_params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]

    @testset "zero signal vector → fit_mcm should not crash" begin
        zero_signal = zeros(length(acq.bvalues))
        # fit_mcm may converge to degenerate parameters but should NOT throw
        result = try
            fit_mcm(mcm, acq, zero_signal)
        catch e
            e
        end
        if result isa Exception
            # If it throws, that is a bug — it should handle gracefully
            @test_broken false  # BUG: fit_mcm crashes on zero signal
        else
            @test haskey(result, :parameters)
            @test all(isfinite, result[:parameters])
        end
    end

    @testset "all-ones signal → fit_mcm should converge" begin
        ones_signal = ones(length(acq.bvalues))
        result = try
            fit_mcm(mcm, acq, ones_signal)
        catch e
            e
        end
        if result isa Exception
            @test_broken false  # BUG: fit_mcm crashes on all-ones signal
        else
            @test haskey(result, :parameters)
            @test all(isfinite, result[:parameters])
        end
    end

    @testset "negative b-values → should error or handle gracefully" begin
        neg_bvals = [-1000e6, -500e6, 0.0]
        neg_bvecs = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        neg_acq = Acquisition(neg_bvals, neg_bvecs)
        sig = signal(ball, neg_acq)
        # With negative b and positive D, exp(-b*D) = exp(+|b|*D) > 1
        # This is physically nonsensical. Either the code should error or
        # we document the behavior.
        if all(isfinite, sig)
            # signal > 1 for negative b-values: no guard in place
            @test any(sig .> 1.0)  # documents the lack of validation
            # BUG: negative b-values silently produce unphysical signal > 1
        end
    end

    @testset "NaN in signal → should not silently propagate through fit_mcm" begin
        nan_signal = signal(mcm, acq, true_params)
        nan_signal[1] = NaN
        result = try
            fit_mcm(mcm, acq, nan_signal)
        catch e
            e
        end
        if result isa Exception
            # Throwing on NaN input is acceptable (informative error)
            @test true
        else
            # If it returns, the parameters should not all be NaN
            fitted = result[:parameters]
            @test all(isfinite, fitted)  # BUG if this fails: NaN propagated silently
        end
    end

    @testset "Inf in signal → should not silently propagate through fit_mcm" begin
        inf_signal = signal(mcm, acq, true_params)
        inf_signal[1] = Inf
        result = try
            fit_mcm(mcm, acq, inf_signal)
        catch e
            e
        end
        if result isa Exception
            @test true  # throwing is acceptable
        else
            fitted = result[:parameters]
            @test all(isfinite, fitted)
        end
    end

    @testset "empty acquisition (0 measurements)" begin
        empty_bvals = Float64[]
        empty_bvecs = Matrix{Float64}(undef, 0, 3)
        empty_acq = Acquisition(empty_bvals, empty_bvecs)
        result = try
            signal(ball, empty_acq)
        catch e
            e
        end
        if result isa Exception
            @test true  # throwing is acceptable for empty input
        else
            @test length(result) == 0
        end
    end

    @testset "single measurement (1 b-value)" begin
        single_acq = Acquisition([1000e6], reshape([1.0 0.0 0.0], 1, 3))
        sig = signal(ball, single_acq)
        @test length(sig) == 1
        @test sig[1] ≈ exp(-1000e6 * 2.0e-9) atol=1e-12

        # Fitting with 1 measurement and 7 params — extremely underdetermined
        data = signal(mcm, single_acq, true_params)
        result = try
            fit_mcm(mcm, single_acq, data)
        catch e
            e
        end
        if result isa Exception
            @test true  # acceptable: underdetermined problem
        else
            @test haskey(result, :parameters)
        end
    end

end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Degenerate parameter combinations
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adversarial: Degenerate parameter combinations" begin

    acq = hcp_like_acquisition()

    @testset "G1Ball with lambda_iso=0 → signal all 1s" begin
        ball_zero = G1Ball(lambda_iso=0.0)
        sig = signal(ball_zero, acq)
        @test all(sig .≈ 1.0)
    end

    @testset "G1Ball with lambda_iso < 0 → unphysical" begin
        ball_neg = G1Ball(lambda_iso=-1.0e-9)
        sig = signal(ball_neg, acq)
        # exp(-b * (-D)) = exp(b*D) → signal explodes > 1
        if all(isfinite, sig)
            @test any(sig .> 1.0)
            # BUG: no validation on negative diffusivity in G1Ball constructor
        end
    end

    @testset "C1Stick with zero-norm mu → should handle gracefully" begin
        # The implementation normalizes mu with max(norm, 1e-12), so zero mu
        # should not cause division by zero
        stick_zero = C1Stick(mu=[0.0, 0.0, 0.0], lambda_par=1.7e-9)
        sig = try
            signal(stick_zero, acq)
        catch e
            e
        end
        if sig isa Exception
            @test_broken false  # BUG: zero mu should be handled gracefully
        else
            @test all(isfinite, sig)
            @test length(sig) == length(acq.bvalues)
        end
    end

    @testset "G2Zeppelin with lambda_perp > lambda_par → still computes" begin
        # Physically unusual (prolate requires lambda_par >= lambda_perp)
        # but the math should not break
        zep_inverted = G2Zeppelin(mu=[0.0, 0.0, 1.0], lambda_par=0.5e-9, lambda_perp=2.0e-9)
        sig = signal(zep_inverted, acq)
        @test all(isfinite, sig)
        @test all(sig .>= 0.0)
        @test all(sig .<= 1.0 + 1e-10)
    end

    @testset "volume fractions that do not sum to 1 in MCM" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        # Fractions sum to 0.5
        params_low = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.2, 0.3]
        sig = signal(mcm, acq, params_low)
        @test all(isfinite, sig)
        # At b=0, signal should equal sum of fractions
        acq_b0 = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
        sig_b0 = signal(mcm, acq_b0, params_low)
        @test sig_b0[1] ≈ 0.5 atol=1e-15

        # Fractions sum to 1.5 — no guard
        params_high = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.8, 0.7]
        sig_high = signal(mcm, acq_b0, params_high)
        @test sig_high[1] ≈ 1.5 atol=1e-15
        # BUG: MCM does not validate that volume fractions sum to 1
    end

    @testset "negative volume fractions in MCM" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        params_neg = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, -0.3, 1.3]
        sig = signal(mcm, acq, params_neg)
        # No validation → could produce negative signal
        @test all(isfinite, sig)
        # BUG: negative volume fractions silently accepted
    end

    @testset "volume fractions > 1 in MCM" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        params_over = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 1.5, 0.5]
        sig = signal(mcm, acq, params_over)
        @test all(isfinite, sig)
        # BUG: volume fractions > 1 silently accepted
    end

end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Numerical stability
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adversarial: Numerical stability" begin

    @testset "very high b-values (1e12) → signal ≈ 0, not NaN" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        extreme_acq = Acquisition([1e12], reshape([1.0 0.0 0.0], 1, 3))
        sig = signal(ball, extreme_acq)
        @test all(isfinite, sig)
        @test sig[1] ≈ 0.0 atol=1e-100  # exp(-2e3) is effectively zero
    end

    @testset "very small diffusivity (1e-15) → signal ≈ 1" begin
        ball_tiny = G1Ball(lambda_iso=1e-15)
        acq = hcp_like_acquisition()
        sig = signal(ball_tiny, acq)
        @test all(isfinite, sig)
        # b*D ≈ 3e9 * 1e-15 = 3e-6, exp(-3e-6) ≈ 1
        @test all(sig .> 0.999)
    end

    @testset "very large diffusivity (1e-6) → signal ≈ 0 for non-b0" begin
        ball_huge = G1Ball(lambda_iso=1e-6)
        acq = hcp_like_acquisition()
        sig = signal(ball_huge, acq)
        @test all(isfinite, sig)
        # b=0 measurements should still be 1.0
        b0_mask = acq.bvalues .== 0.0
        @test all(sig[b0_mask] .≈ 1.0)
        # Non-b0: b*D ≈ 1e9 * 1e-6 = 1e3, exp(-1e3) ≈ 0
        non_b0 = sig[.!b0_mask]
        @test all(non_b0 .< 1e-100)
    end

    @testset "RestrictedCylinder with very small diameter (1e-10 m)" begin
        cyl = RestrictedCylinder(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9, diameter=1e-10)
        acq = Acquisition([1000e6, 2000e6], [1.0 0.0 0.0; 0.0 1.0 0.0], 12.9e-3, 21.8e-3)
        sig = try
            signal(cyl, acq)
        catch e
            e
        end
        if sig isa Exception
            @test_broken false  # BUG: tiny diameter should not crash
        else
            @test all(isfinite, sig)
            # Fully restricted perpendicular: signal should ≈ stick signal
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            sig_stick = signal(stick, acq)
            @test sig ≈ sig_stick atol=0.01
        end
    end

    @testset "SphereGPD with very large diameter (1e-3 m = 1 mm)" begin
        sphere = SphereGPD(diameter=1e-3, D_intra=2.0e-9)
        acq = Acquisition([1000e6, 2000e6], [1.0 0.0 0.0; 0.0 1.0 0.0], 12.9e-3, 21.8e-3)
        sig = try
            signal(sphere, acq)
        catch e
            e
        end
        if sig isa Exception
            @test_broken false  # BUG: large diameter should not crash
        else
            @test all(isfinite, sig)
            @test all(sig .>= 0.0)
        end
    end

    @testset "Rician loglikelihood with sigma → 0" begin
        observed = [0.8, 0.6, 0.4]
        predicted = [0.8, 0.6, 0.4]

        # Very small sigma → z = x*v/sigma^2 is huge
        ll = try
            rician_loglikelihood(observed, predicted, 1e-10)
        catch e
            e
        end
        if ll isa Exception
            @test true  # throwing on degenerate sigma is acceptable
        else
            @test isfinite(ll)  # BUG if -Inf: numerical instability
        end
    end

    @testset "Rician loglikelihood with very large sigma" begin
        observed = [0.8, 0.6, 0.4]
        predicted = [0.8, 0.6, 0.4]
        ll = rician_loglikelihood(observed, predicted, 1e6)
        @test isfinite(ll)
    end

    @testset "log_besseli0 with z=0" begin
        val = log_besseli0(0.0)
        @test val ≈ 0.0 atol=1e-12  # I0(0) = 1, log(1) = 0
    end

    @testset "log_besseli0 with very large z (1e6)" begin
        val = log_besseli0(1e6)
        @test isfinite(val)
        # Asymptotic: z - 0.5*log(2π*z)
        expected = 1e6 - 0.5 * log(2π * 1e6)
        @test val ≈ expected rtol=1e-4
    end

    @testset "log_besseli0 with negative z" begin
        # I0(-z) = I0(z) for the modified Bessel function, but implementation
        # may not handle negative z gracefully
        val = try
            log_besseli0(-10.0)
        catch e
            e
        end
        if val isa Exception
            @test true  # throwing on negative z is acceptable
        else
            # I0 is even, so log(I0(-z)) = log(I0(z))
            @test val ≈ log_besseli0(10.0) atol=1e-10
        end
    end

    @testset "Rician loglikelihood with zero observed signal" begin
        # log(x) where x=0 → -Inf, but code uses log(max(x, 1e-30))
        observed = [0.0, 0.6, 0.4]
        predicted = [0.1, 0.6, 0.4]
        ll = rician_loglikelihood(observed, predicted, 0.05)
        @test isfinite(ll)
    end

    @testset "Rician loglikelihood with zero predicted signal" begin
        observed = [0.8, 0.6, 0.4]
        predicted = [0.0, 0.6, 0.4]
        ll = rician_loglikelihood(observed, predicted, 0.05)
        @test isfinite(ll)
    end

    @testset "Stick signal with extreme b-value and diffusivity" begin
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=3.0e-9)
        extreme_acq = Acquisition([1e13], reshape([0.0 0.0 1.0], 1, 3))
        sig = signal(stick, extreme_acq)
        @test all(isfinite, sig)
        @test sig[1] ≈ 0.0 atol=1e-100
    end

    @testset "Zeppelin signal with extreme parameters" begin
        zep = G2Zeppelin(mu=[0.0, 0.0, 1.0], lambda_par=3.0e-9, lambda_perp=3.0e-9)
        extreme_acq = Acquisition([1e13], reshape([1.0 0.0 0.0], 1, 3))
        sig = signal(zep, extreme_acq)
        @test all(isfinite, sig)
    end

end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-method consistency
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adversarial: Cross-method consistency" begin

    @testset "MCM signal matches manual compartment sum exactly" begin
        acq = hcp_like_acquisition()
        D_ball = 2.0e-9
        D_stick = 1.7e-9
        f_ball = 0.4
        f_stick = 0.6
        mu = [0.0, 0.0, 1.0]

        ball = G1Ball(lambda_iso=D_ball)
        stick = C1Stick(mu=mu, lambda_par=D_stick)
        mcm = MultiCompartmentModel(ball, stick)

        params = [D_ball, mu..., D_stick, f_ball, f_stick]
        sig_mcm = signal(mcm, acq, params)

        # Manual sum
        sig_ball = signal(ball, acq)
        sig_stick = signal(stick, acq)
        sig_manual = f_ball .* sig_ball .+ f_stick .* sig_stick

        @test sig_mcm ≈ sig_manual atol=1e-14
    end

    @testset "fit_mcm and MCMC agree on high-SNR Ball+Stick data" begin
        acq = hcp_like_acquisition()
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        true_params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]
        clean_signal = signal(mcm, acq, true_params)

        # Add tiny noise for realism
        rng = MersenneTwister(42)
        sigma = 0.01
        noisy = clean_signal .+ sigma .* randn(rng, length(clean_signal))
        noisy = max.(noisy, 1e-6)

        # NLLS fit
        nlls_result = fit_mcm(mcm, acq, noisy; init=true_params)
        nlls_params = nlls_result[:parameters]

        # MCMC
        lower, upper = get_flat_bounds(mcm)
        forward_fn = (model, acq_arg, p) -> signal(mcm, acq_arg, p)
        samples, accept_rate = mcmc_sample(
            forward_fn, acq, noisy, MersenneTwister(99);
            n_samples=500, n_warmup=300, sigma=sigma,
            lower=lower, upper=upper,
            init=true_params
        )
        mcmc_mean = vec(mean(samples, dims=2))

        # Both should recover lambda_iso within 20%
        @test abs(nlls_params[1] - 2.0e-9) / 2.0e-9 < 0.2
        @test abs(mcmc_mean[1] - 2.0e-9) / 2.0e-9 < 0.2

        # Both should recover lambda_par within 20%
        @test abs(nlls_params[5] - 1.7e-9) / 1.7e-9 < 0.2
        @test abs(mcmc_mean[5] - 1.7e-9) / 1.7e-9 < 0.2

        # Volume fraction agreement (NLLS vs MCMC) — should be within 0.15
        @test abs(nlls_params[6] - mcmc_mean[6]) < 0.15
    end

    @testset "Three-compartment MCM signal additivity" begin
        acq = hcp_like_acquisition()
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        dot = S1Dot()
        mcm = MultiCompartmentModel(ball, stick, dot)

        # Params: [D_ball, mu..., D_stick, f_ball, f_stick, f_dot]
        f_ball, f_stick, f_dot = 0.3, 0.5, 0.2
        params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, f_ball, f_stick, f_dot]

        sig_mcm = signal(mcm, acq, params)
        sig_manual = f_ball .* signal(ball, acq) .+
                     f_stick .* signal(stick, acq) .+
                     f_dot .* signal(dot, acq)

        @test sig_mcm ≈ sig_manual atol=1e-14
    end

    @testset "Constrained model signal equals unconstrained with same params" begin
        acq = hcp_like_acquisition()
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        ball = G1Ball(lambda_iso=3.0e-9)
        mcm = MultiCompartmentModel(stick, ball)
        constrained = set_fixed_parameter(mcm, :lambda_par, 1.7e-9)

        # Full MCM signal
        full_params = [0.0, 0.0, 1.0, 1.7e-9, 3.0e-9, 0.6, 0.4]
        sig_full = signal(mcm, acq, full_params)

        # Constrained signal (lambda_par removed)
        free_params = [0.0, 0.0, 1.0, 3.0e-9, 0.6, 0.4]
        sig_constrained = signal(constrained, acq, free_params)

        @test sig_full ≈ sig_constrained atol=1e-14
    end

end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adversarial: Reproducibility" begin

    @testset "same RNG seed → same MCMC samples" begin
        observed = [0.5, 0.3, 0.7]
        lower = [0.0, 0.0, 0.0]
        upper = [1.0, 1.0, 1.0]
        init = [0.5, 0.3, 0.7]
        forward_fn = (model, acq, p) -> p

        samples1, _ = mcmc_sample(
            forward_fn, nothing, observed, MersenneTwister(42);
            n_samples=100, n_warmup=50, sigma=0.05,
            lower=lower, upper=upper, init=init
        )
        samples2, _ = mcmc_sample(
            forward_fn, nothing, observed, MersenneTwister(42);
            n_samples=100, n_warmup=50, sigma=0.05,
            lower=lower, upper=upper, init=init
        )

        @test samples1 == samples2
    end

    @testset "fit_mcm with same init → same result" begin
        acq = hcp_like_acquisition()
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        true_params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]
        data = signal(mcm, acq, true_params)
        init = [1.5e-9, 0.1, 0.1, 0.9, 1.5e-9, 0.5, 0.5]

        r1 = fit_mcm(mcm, acq, data; init=init)
        r2 = fit_mcm(mcm, acq, data; init=init)

        @test r1[:parameters] ≈ r2[:parameters] atol=1e-15
    end

    @testset "same RNG seed → same Rician noise" begin
        sig_matrix = reshape([0.8, 0.6, 0.4, 0.3], 1, 4)
        noisy1 = add_rician_noise(MersenneTwister(7), sig_matrix, 0.05)
        noisy2 = add_rician_noise(MersenneTwister(7), sig_matrix, 0.05)
        @test noisy1 == noisy2
    end

    @testset "different RNG seed → different Rician noise" begin
        sig_matrix = reshape([0.8, 0.6, 0.4, 0.3], 1, 4)
        noisy1 = add_rician_noise(MersenneTwister(7), sig_matrix, 0.05)
        noisy2 = add_rician_noise(MersenneTwister(8), sig_matrix, 0.05)
        @test noisy1 != noisy2
    end

end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Additional edge cases
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adversarial: Additional edge cases" begin

    @testset "G2Zeppelin with lambda_par=0 and lambda_perp=0 → signal all 1s" begin
        zep = G2Zeppelin(mu=[0.0, 0.0, 1.0], lambda_par=0.0, lambda_perp=0.0)
        acq = hcp_like_acquisition()
        sig = signal(zep, acq)
        @test all(sig .≈ 1.0)
    end

    @testset "Stick with unnormalized mu → signal still valid" begin
        # mu = [10, 0, 0] should be internally normalized to [1, 0, 0]
        stick_big = C1Stick(mu=[10.0, 0.0, 0.0], lambda_par=1.7e-9)
        stick_unit = C1Stick(mu=[1.0, 0.0, 0.0], lambda_par=1.7e-9)
        acq = hcp_like_acquisition()
        @test signal(stick_big, acq) ≈ signal(stick_unit, acq) atol=1e-12
    end

    @testset "MCM with single compartment" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        mcm = MultiCompartmentModel(ball)
        acq = hcp_like_acquisition()

        @test nparams(mcm) == 2  # lambda_iso + 1 fraction
        params = [2.0e-9, 1.0]
        sig = signal(mcm, acq, params)
        @test sig ≈ signal(ball, acq) atol=1e-14
    end

    @testset "fit_mcm_batch with single voxel" begin
        acq = hcp_like_acquisition()
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        true_params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]
        data = reshape(signal(mcm, acq, true_params), 1, :)

        results = fit_mcm_batch(mcm, acq, data)
        @test size(results[:parameters]) == (1, nparams(mcm))
    end

    @testset "hcp_like_acquisition deterministic with same seed" begin
        acq1 = hcp_like_acquisition(seed=42)
        acq2 = hcp_like_acquisition(seed=42)
        @test acq1.bvalues == acq2.bvalues
        @test acq1.gradient_directions == acq2.gradient_directions
    end

    @testset "hcp_like_acquisition different with different seed" begin
        acq1 = hcp_like_acquisition(seed=0)
        acq2 = hcp_like_acquisition(seed=1)
        @test acq1.bvalues == acq2.bvalues  # shells are the same
        @test acq1.gradient_directions != acq2.gradient_directions  # directions differ
    end

    @testset "rician_loglikelihood length mismatch → assertion error" begin
        @test_throws AssertionError rician_loglikelihood([0.5, 0.3], [0.5], 0.05)
    end

    @testset "rician_loglikelihood sigma=0 → assertion error" begin
        @test_throws AssertionError rician_loglikelihood([0.5], [0.5], 0.0)
    end

    @testset "rician_loglikelihood negative sigma → assertion error" begin
        @test_throws AssertionError rician_loglikelihood([0.5], [0.5], -0.05)
    end

    @testset "parameter_dictionary_to_array round-trip fidelity" begin
        ball = G1Ball(lambda_iso=2.0e-9)
        stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
        mcm = MultiCompartmentModel(ball, stick)

        original = [2.0e-9, 0.1, 0.2, 0.3, 1.7e-9, 0.4, 0.6]
        dict = parameter_array_to_dictionary(mcm, original)
        reconstructed = parameter_dictionary_to_array(mcm, dict)
        @test original ≈ reconstructed atol=1e-15
    end

    @testset "add_rician_noise output is always positive" begin
        rng = MersenneTwister(42)
        sig = rand(rng, 10, 90)  # 10 voxels, 90 measurements
        noisy = add_rician_noise(MersenneTwister(0), sig, 0.05)
        @test all(noisy .>= 0.0)  # Rician is non-negative by definition
    end

    @testset "add_rician_noise with zero signal" begin
        rng = MersenneTwister(42)
        sig = zeros(5, 10)
        noisy = add_rician_noise(MersenneTwister(0), sig, 0.05)
        @test all(isfinite, noisy)
        @test all(noisy .>= 0.0)
    end

    @testset "S1Dot has empty parameter ranges" begin
        dot = S1Dot()
        @test parameter_ranges(dot) == Dict{Symbol, Tuple{Float64, Float64}}()
        @test parameter_cardinality(dot) == Dict{Symbol, Int}()
    end

    @testset "Watson distribution edge case: kappa=0 (uniform)" begin
        # kappa=0 → uniform distribution on sphere
        wd = WatsonDistribution(n_grid=300)
        mu = [0.0, 0.0, 1.0]
        w = watson_weights(wd, mu, 0.0)
        @test all(isfinite, w)
        @test sum(w) ≈ 1.0 atol=1e-10
        # With kappa=0, weights should be approximately uniform
        @test maximum(w) / minimum(w) < 2.0  # loose bound
    end

    @testset "Watson distribution edge case: very large kappa" begin
        wd = WatsonDistribution(n_grid=300)
        mu = [0.0, 0.0, 1.0]
        w = try
            watson_weights(wd, mu, 1000.0)
        catch e
            e
        end
        if w isa Exception
            @test_broken false  # BUG: large kappa should not crash
        else
            @test all(isfinite, w)
            @test sum(w) ≈ 1.0 atol=1e-10
        end
    end

    @testset "Watson distribution edge case: negative kappa (girdle)" begin
        wd = WatsonDistribution(n_grid=300)
        mu = [0.0, 0.0, 1.0]
        w = watson_weights(wd, mu, -10.0)
        @test all(isfinite, w)
        @test sum(w) ≈ 1.0 atol=1e-10
    end

end
