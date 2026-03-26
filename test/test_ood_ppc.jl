using Test, Random, LinearAlgebra, Statistics

@testset "OOD Detection & Posterior Predictive Checks" begin

    # ==================================================================
    # OOD Detection
    # ==================================================================

    @testset "OOD Detection" begin

        @testset "reconstruction_error: higher for OOD data" begin
            rng = MersenneTwister(42)
            # Simple predict_fn: identity (perfect reconstruction for in-dist)
            predict_fn(X) = X
            X_in  = randn(rng, 50, 4)            # in-distribution
            X_ood = randn(rng, 50, 4) .+ 10.0    # shifted → predict_fn returns X_ood but "true" signal differs

            # For a meaningful test: predict_fn learns in-dist mean.
            # Use a predict_fn that returns the training-set mean for every row.
            X_train = randn(MersenneTwister(1), 200, 4)
            mu_train = mean(X_train, dims=1)
            pred_mean(X) = repeat(mu_train, size(X, 1), 1)

            err_in  = reconstruction_error(pred_mean, X_train[1:50, :])
            err_ood = reconstruction_error(pred_mean, X_ood)
            @test mean(err_ood) > mean(err_in)
        end

        @testset "mahalanobis_distance: higher for OOD points" begin
            rng = MersenneTwister(42)
            X_train = randn(rng, 200, 3)
            X_in    = randn(MersenneTwister(7), 50, 3)
            X_ood   = randn(MersenneTwister(8), 50, 3) .+ 5.0

            md_in  = mahalanobis_distance(X_in, X_train)
            md_ood = mahalanobis_distance(X_ood, X_train)

            @test md_in  isa Vector{Float64}
            @test md_ood isa Vector{Float64}
            @test length(md_in)  == 50
            @test length(md_ood) == 50
            @test mean(md_ood) > mean(md_in)
        end

        @testset "ood_score combines metrics correctly" begin
            rng = MersenneTwister(42)
            X_train = randn(rng, 200, 4)
            X_test  = randn(MersenneTwister(99), 30, 4)
            mu_train = mean(X_train, dims=1)
            pred_mean(X) = repeat(mu_train, size(X, 1), 1)

            scores = ood_score(pred_mean, X_test, X_train)
            @test scores isa Vector{Float64}
            @test length(scores) == 30
            @test all(scores .>= 0)

            # With different weights
            scores_recon = ood_score(pred_mean, X_test, X_train; weights=(1.0, 0.0))
            scores_mahal = ood_score(pred_mean, X_test, X_train; weights=(0.0, 1.0))
            err = reconstruction_error(pred_mean, X_test)
            md  = mahalanobis_distance(X_test, X_train)

            @test scores_recon ≈ err
            @test scores_mahal ≈ md
        end

        @testset "ood_detect flags OOD points with threshold" begin
            scores = [0.1, 0.2, 0.5, 0.8, 2.0, 5.0]

            # Explicit threshold
            flags = ood_detect(scores; threshold=1.0)
            @test flags isa BitVector
            @test flags == BitVector([0, 0, 0, 0, 1, 1])

            # Percentile-based threshold (default percentile=95)
            flags_pct = ood_detect(scores; percentile=50)
            # 50th percentile of [0.1, 0.2, 0.5, 0.8, 2.0, 5.0] = 0.65
            # Points > 0.65 are flagged
            @test sum(flags_pct) >= 2  # at least 0.8, 2.0, 5.0
        end

        @testset "Integration: detects distribution shift" begin
            rng = MersenneTwister(42)
            X_train = randn(rng, 500, 5)               # N(0,1)
            X_in    = randn(MersenneTwister(7), 100, 5) # N(0,1) — same dist
            X_ood   = randn(MersenneTwister(8), 100, 5) .+ 5.0  # N(5,1) — shifted

            mu_train = mean(X_train, dims=1)
            pred_mean(X) = repeat(mu_train, size(X, 1), 1)

            scores_in  = ood_score(pred_mean, X_in, X_train)
            scores_ood = ood_score(pred_mean, X_ood, X_train)

            # Compute threshold from in-distribution at 95th percentile
            threshold = sort(scores_in)[ceil(Int, 0.95 * length(scores_in))]

            flags_in  = ood_detect(scores_in;  threshold=threshold)
            flags_ood = ood_detect(scores_ood; threshold=threshold)

            # Most OOD points should be flagged, most in-dist should not
            @test sum(flags_ood) / length(flags_ood) > 0.8
            @test sum(flags_in)  / length(flags_in)  < 0.2
        end
    end

    # ==================================================================
    # Posterior Predictive Checks
    # ==================================================================

    @testset "Posterior Predictive Checks" begin

        @testset "posterior_predictive_check returns p-values per observation" begin
            rng = MersenneTwister(42)
            n_obs = 20
            observed = randn(rng, n_obs)
            # Posterior samples: 10 draws of a mean parameter
            posterior_samples = randn(MersenneTwister(7), 10)
            # Forward fn: data = theta + noise(0, 0.1)
            forward_fn(theta, rng) = theta .+ 0.1 .* randn(rng, n_obs)

            pvals = posterior_predictive_check(observed, posterior_samples, forward_fn, MersenneTwister(99); n_ppc=50)
            @test pvals isa Vector{Float64}
            @test length(pvals) == n_obs
            @test all(0.0 .<= pvals .<= 1.0)
        end

        @testset "well-specified model gives p-values ~ Uniform(0,1)" begin
            # Generate data from known model, then check posterior predictive
            rng = MersenneTwister(42)
            true_theta = 2.0
            n_obs = 100
            sigma = 1.0
            observed = true_theta .+ sigma .* randn(rng, n_obs)

            # Posterior samples near the true value
            posterior_samples = true_theta .+ 0.1 .* randn(MersenneTwister(7), 50)
            forward_fn(theta, rng) = theta .+ sigma .* randn(rng, n_obs)

            pvals = posterior_predictive_check(observed, posterior_samples, forward_fn, MersenneTwister(99); n_ppc=200)

            # For well-specified model, p-values should be roughly uniform
            # Kolmogorov-Smirnov-like check: median should be near 0.5
            @test 0.15 < median(pvals) < 0.85
            # Not too many extreme p-values
            @test mean(pvals .< 0.05) < 0.30
        end

        @testset "misspecified model gives low p-values" begin
            rng = MersenneTwister(42)
            n_obs = 50
            # True data from theta=10
            observed = 10.0 .+ 0.5 .* randn(rng, n_obs)

            # Posterior is completely wrong: centred at 0
            posterior_samples = 0.1 .* randn(MersenneTwister(7), 30)
            forward_fn(theta, rng) = theta .+ 0.5 .* randn(rng, n_obs)

            pvals = posterior_predictive_check(observed, posterior_samples, forward_fn, MersenneTwister(99); n_ppc=200)

            # Misspecified model should produce mostly low p-values
            @test median(pvals) < 0.15
        end

        @testset "ppc_summary returns fraction of flagged observations" begin
            pvals = [0.01, 0.02, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

            summary = ppc_summary(pvals)
            @test summary isa NamedTuple
            @test haskey(summary, :n_flagged)
            @test haskey(summary, :fraction_flagged)
            @test haskey(summary, :median_pvalue)

            # With alpha=0.05, observations with p < 0.05 are flagged
            result = ppc_summary(pvals; alpha=0.05)
            @test result.n_flagged == 3         # 0.01, 0.02, 0.03
            @test result.fraction_flagged ≈ 0.3 # 3/10
            @test result.median_pvalue ≈ median(pvals)
        end
    end
end
