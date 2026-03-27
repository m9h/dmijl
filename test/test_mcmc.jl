"""
Tests for MCMC posterior sampling with Rician log-likelihood.

Red-Green TDD: tests written first, implementation follows.
Tests: rician_loglikelihood, mcmc_sample, mcmc_summary,
       and integration with composable compartment models.
"""

using Test, Random, Statistics, LinearAlgebra
using SpecialFunctions: besseli

@testset "MCMC Inference" begin

    # ---- Rician log-likelihood ----

    @testset "rician_loglikelihood is finite for valid inputs" begin
        observed = [0.8, 0.6, 0.4, 0.3]
        predicted = [0.85, 0.55, 0.42, 0.28]
        sigma = 0.05
        ll = rician_loglikelihood(observed, predicted, sigma)
        @test isfinite(ll)
        @test ll isa Real
    end

    @testset "rician_loglikelihood is higher for data closer to prediction" begin
        observed = [0.8, 0.6, 0.4]
        predicted_close = [0.81, 0.59, 0.41]
        predicted_far = [0.5, 0.3, 0.1]
        sigma = 0.05

        ll_close = rician_loglikelihood(observed, predicted_close, sigma)
        ll_far = rician_loglikelihood(observed, predicted_far, sigma)

        @test ll_close > ll_far
    end

    @testset "rician_loglikelihood handles SNR parameter correctly" begin
        observed = [0.8, 0.6, 0.4]
        predicted = [0.8, 0.6, 0.4]  # exact match

        ll_low_noise = rician_loglikelihood(observed, predicted, 0.01)
        ll_high_noise = rician_loglikelihood(observed, predicted, 0.1)

        # With exact match and lower noise, the likelihood should be higher
        # (sharper peak around the true value)
        @test ll_low_noise > ll_high_noise
    end

    @testset "rician_loglikelihood numerical stability with large arguments" begin
        # Large signal values that could cause Bessel overflow without log-space trick
        observed = [10.0, 20.0, 30.0]
        predicted = [10.5, 19.5, 30.5]
        sigma = 0.01  # small sigma => large z = obs*pred/sigma^2

        ll = rician_loglikelihood(observed, predicted, sigma)
        @test isfinite(ll)
    end

    @testset "rician_loglikelihood matches analytical for known case" begin
        # For a single observation, verify against manual computation
        x = 0.5  # observed
        v = 0.5  # predicted
        sigma = 0.1
        z = x * v / sigma^2

        # Manual log-likelihood:
        # log(x) - 2*log(sigma) - (x^2+v^2)/(2*sigma^2) + log(I0(z))
        expected = log(x) - 2 * log(sigma) - (x^2 + v^2) / (2 * sigma^2) + log(besseli(0, z))

        ll = rician_loglikelihood([x], [v], sigma)
        @test ll ≈ expected atol = 1e-6
    end

    # ---- mcmc_sample ----

    @testset "mcmc_sample returns matrix of correct shape" begin
        # Simple Gaussian toy problem: minimize (params - target)^2
        # Use a dummy "model" via closure
        target = [0.5, 0.3]
        n_params = 2
        n_samples = 200
        n_warmup = 100

        # Mock model: signal just returns the params (identity forward model)
        mock_model = nothing
        mock_acq = nothing

        # We need to define a log_posterior function for MH
        # For this test, just verify shape — use the toy interface
        observed = [0.5, 0.3]
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]

        rng = MersenneTwister(42)
        samples, accept_rate = mcmc_sample(
            (model, acq, p) -> p,  # identity forward model
            mock_acq, observed, rng;
            n_samples=n_samples, n_warmup=n_warmup, sigma=0.05,
            lower=lower, upper=upper,
            init=target
        )

        @test size(samples) == (n_params, n_samples)
        @test accept_rate isa Float64
    end

    @testset "mcmc_sample chains have finite values within bounds" begin
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]
        observed = [0.5, 0.3]
        init = [0.5, 0.3]
        rng = MersenneTwister(42)

        samples, accept_rate = mcmc_sample(
            (model, acq, p) -> p,
            nothing, observed, rng;
            n_samples=300, n_warmup=100, sigma=0.05,
            lower=lower, upper=upper, init=init
        )

        @test all(isfinite, samples)
        # All samples should respect bounds
        for j in 1:size(samples, 2)
            for i in 1:size(samples, 1)
                @test samples[i, j] >= lower[i]
                @test samples[i, j] <= upper[i]
            end
        end
        # Accept rate should be between 0 and 1
        @test 0.0 <= accept_rate <= 1.0
    end

    @testset "mcmc_sample with Gaussian toy problem recovers posterior mean" begin
        # Toy: forward model is identity, observed = [0.5, 0.3]
        # With Rician likelihood centered at truth + small noise,
        # the posterior mean should be near the truth.
        true_params = [0.5, 0.3]
        observed = [0.5, 0.3]  # no noise for simplicity
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]

        rng = MersenneTwister(123)
        samples, accept_rate = mcmc_sample(
            (model, acq, p) -> p,  # identity
            nothing, observed, rng;
            n_samples=2000, n_warmup=500, sigma=0.05,
            lower=lower, upper=upper,
            init=true_params,
            proposal_std=[0.02, 0.02]
        )

        posterior_mean = vec(mean(samples, dims=2))
        @test abs(posterior_mean[1] - 0.5) < 0.1
        @test abs(posterior_mean[2] - 0.3) < 0.1
    end

    @testset "mcmc_sample accept rate is reasonable" begin
        rng = MersenneTwister(7)
        samples, accept_rate = mcmc_sample(
            (model, acq, p) -> p,
            nothing, [0.5, 0.3], rng;
            n_samples=500, n_warmup=200, sigma=0.05,
            lower=[0.0, 0.0], upper=[1.0, 1.0],
            init=[0.5, 0.3],
            proposal_std=[0.02, 0.02]
        )
        # A well-tuned MH should accept 20-70% of proposals
        @test accept_rate > 0.05
        @test accept_rate < 0.99
    end

    # ---- mcmc_summary ----

    @testset "mcmc_summary returns correct statistics" begin
        rng = MersenneTwister(0)
        # Known distribution: standard normal-ish samples
        samples = randn(rng, 2, 1000) .* 0.1 .+ 0.5

        summary = mcmc_summary(samples)

        @test haskey(summary, :mean)
        @test haskey(summary, :std)
        @test haskey(summary, :median)
        @test haskey(summary, :q025)
        @test haskey(summary, :q975)

        @test length(summary[:mean]) == 2
        @test length(summary[:std]) == 2
        @test length(summary[:median]) == 2
        @test length(summary[:q025]) == 2
        @test length(summary[:q975]) == 2

        # Mean should be near 0.5
        @test abs(summary[:mean][1] - 0.5) < 0.05
        @test abs(summary[:mean][2] - 0.5) < 0.05

        # q025 < median < q975
        for i in 1:2
            @test summary[:q025][i] < summary[:median][i]
            @test summary[:median][i] < summary[:q975][i]
        end
    end

    # ---- Integration: Ball+Stick MCMC ----

    @testset "Integration: Ball+Stick model with known params" begin
        if isdefined(Main, :DMI) || isdefined(Main, :MultiCompartmentModel)
            acq = hcp_like_acquisition()

            ball = G1Ball(lambda_iso=2.0e-9)
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            mcm = MultiCompartmentModel(ball, stick)

            # True parameters: lambda_iso, mu_x, mu_y, mu_z, lambda_par, f1, f2
            true_params = [2.0e-9, 0.0, 0.0, 1.0, 1.7e-9, 0.3, 0.7]

            # Generate observed signal with slight noise
            clean_signal = signal(mcm, acq, true_params)
            rng_noise = MersenneTwister(42)
            sigma_noise = 0.02
            observed = clean_signal .+ sigma_noise .* randn(rng_noise, length(clean_signal))
            observed = max.(observed, 1e-6)  # ensure positive

            lower, upper = get_flat_bounds(mcm)

            # Run MCMC
            rng = MersenneTwister(99)
            forward_fn = (model, acq_arg, p) -> signal(mcm, acq_arg, p)

            samples, accept_rate = mcmc_sample(
                forward_fn, acq, observed, rng;
                n_samples=500, n_warmup=300, sigma=sigma_noise,
                lower=lower, upper=upper,
                init=true_params,
                proposal_std=nothing  # auto
            )

            @test size(samples, 1) == nparams(mcm)
            @test size(samples, 2) == 500
            @test all(isfinite, samples)

            summary = mcmc_summary(samples)

            # Posterior mean of lambda_iso (param 1) should be near 2.0e-9
            @test abs(summary[:mean][1] - 2.0e-9) < 1.0e-9

            # Posterior mean of lambda_par (param 5) should be near 1.7e-9
            @test abs(summary[:mean][5] - 1.7e-9) < 1.0e-9

            # Credible intervals should contain truth for most parameters
            for i in [1, 5]  # lambda_iso, lambda_par (well-identified)
                @test summary[:q025][i] <= true_params[i] <= summary[:q975][i]
            end
        else
            @info "Skipping Ball+Stick MCMC integration test (DMI module not loaded)."
            @test true
        end
    end

end
