using Test, Random, Statistics

@testset "Conformal Prediction" begin

    # ------------------------------------------------------------------ #
    # split_conformal: perfect predictor gives tight intervals
    # ------------------------------------------------------------------ #
    @testset "split_conformal with perfect predictor gives tight intervals" begin
        rng = MersenneTwister(42)
        n_cal, n_test, d = 500, 100, 3
        X_cal = randn(rng, n_cal, d)
        Y_cal = X_cal[:, 1:1]                      # Y = X[:,1] exactly
        X_test = randn(rng, n_test, d)
        Y_test = X_test[:, 1:1]

        perfect_predict(X) = X[:, 1:1]              # zero residual

        lower, upper = split_conformal(perfect_predict, X_cal, Y_cal, X_test; alpha=0.1)
        widths = upper .- lower

        # Perfect predictor => all residuals are 0 => intervals are [Yhat, Yhat]
        @test all(widths .< 1e-10)
    end

    # ------------------------------------------------------------------ #
    # Coverage guarantee: coverage >= 1 - alpha (finite-sample property)
    # ------------------------------------------------------------------ #
    @testset "split_conformal coverage >= 1-alpha on test set" begin
        rng = MersenneTwister(123)
        n_cal, n_test, d = 1000, 500, 2

        X_cal = randn(rng, n_cal, d)
        Y_cal = 2.0 .* X_cal[:, 1:1] .+ 0.5 .* randn(rng, n_cal, 1)
        X_test = randn(rng, n_test, d)
        Y_test = 2.0 .* X_test[:, 1:1] .+ 0.5 .* randn(rng, n_test, 1)

        noisy_predict(X) = 2.0 .* X[:, 1:1]         # ignores noise term

        alpha = 0.1
        lower, upper = split_conformal(noisy_predict, X_cal, Y_cal, X_test; alpha=alpha)
        cov = conformal_coverage(lower, upper, Y_test)

        @test cov >= 1 - alpha - 0.05                # allow small stat fluctuation
    end

    # ------------------------------------------------------------------ #
    # alpha=0.1 gives ~90% coverage
    # ------------------------------------------------------------------ #
    @testset "split_conformal with alpha=0.1 gives ~90% coverage" begin
        rng = MersenneTwister(7)
        n_cal, n_test = 2000, 1000

        X_cal = randn(rng, n_cal, 1)
        Y_cal = 3.0 .* X_cal .+ randn(rng, n_cal, 1)
        X_test = randn(rng, n_test, 1)
        Y_test = 3.0 .* X_test .+ randn(rng, n_test, 1)

        predict_fn(X) = 3.0 .* X

        lower, upper = split_conformal(predict_fn, X_cal, Y_cal, X_test; alpha=0.1)
        cov = conformal_coverage(lower, upper, Y_test)

        @test cov >= 0.85     # should be ~90%, allow margin
        @test cov <= 0.98     # should not be wildly too wide
    end

    # ------------------------------------------------------------------ #
    # alpha=0.05 gives ~95% coverage
    # ------------------------------------------------------------------ #
    @testset "split_conformal with alpha=0.05 gives ~95% coverage" begin
        rng = MersenneTwister(99)
        n_cal, n_test = 2000, 1000

        X_cal = randn(rng, n_cal, 1)
        Y_cal = -1.5 .* X_cal .+ 0.3 .* randn(rng, n_cal, 1)
        X_test = randn(rng, n_test, 1)
        Y_test = -1.5 .* X_test .+ 0.3 .* randn(rng, n_test, 1)

        predict_fn(X) = -1.5 .* X

        lower, upper = split_conformal(predict_fn, X_cal, Y_cal, X_test; alpha=0.05)
        cov = conformal_coverage(lower, upper, Y_test)

        @test cov >= 0.90     # should be ~95%, allow margin
        @test cov <= 0.995
    end

    # ------------------------------------------------------------------ #
    # conformal_intervals returns correct shape (lower, upper)
    # ------------------------------------------------------------------ #
    @testset "conformal_intervals returns (lower, upper) of correct shape" begin
        rng = MersenneTwister(11)
        n_cal, n_test, d = 200, 50, 4

        X_cal = randn(rng, n_cal, d)
        Y_cal = randn(rng, n_cal, 2)                 # multi-output
        X_test = randn(rng, n_test, d)

        predict_fn(X) = randn(MersenneTwister(0), size(X, 1), 2)

        lower, upper = split_conformal(predict_fn, X_cal, Y_cal, X_test; alpha=0.1)

        @test size(lower) == (n_test, 2)
        @test size(upper) == (n_test, 2)
        @test all(upper .>= lower)
    end

    # ------------------------------------------------------------------ #
    # Intervals widen when predictor is worse
    # ------------------------------------------------------------------ #
    @testset "intervals widen when predictor is worse" begin
        rng = MersenneTwister(55)
        n_cal, n_test = 500, 100

        X_cal = randn(rng, n_cal, 1)
        Y_cal = 2.0 .* X_cal .+ 0.5 .* randn(rng, n_cal, 1)
        X_test = randn(rng, n_test, 1)

        good_predict(X) = 2.0 .* X                   # correct slope
        bad_predict(X) = 0.0 .* X                     # always predicts 0

        lo_good, up_good = split_conformal(good_predict, X_cal, Y_cal, X_test; alpha=0.1)
        lo_bad, up_bad   = split_conformal(bad_predict, X_cal, Y_cal, X_test; alpha=0.1)

        width_good = mean(up_good .- lo_good)
        width_bad  = mean(up_bad  .- lo_bad)

        @test width_bad > width_good
    end

    # ------------------------------------------------------------------ #
    # CQR conformal also achieves coverage
    # ------------------------------------------------------------------ #
    @testset "cqr_conformal achieves coverage >= 1-alpha" begin
        rng = MersenneTwister(77)
        n_cal, n_test = 1000, 500

        X_cal = randn(rng, n_cal, 1)
        noise_cal = randn(rng, n_cal, 1)
        Y_cal = 2.0 .* X_cal .+ noise_cal

        X_test = randn(rng, n_test, 1)
        noise_test = randn(rng, n_test, 1)
        Y_test = 2.0 .* X_test .+ noise_test

        # Quantile predictors: predict_lower ~ alpha/2, predict_upper ~ 1-alpha/2
        # Use true mean +/- a rough quantile of the noise
        alpha = 0.1
        predict_lower(X) = 2.0 .* X .- 1.3          # rough 5th percentile offset
        predict_upper(X) = 2.0 .* X .+ 1.3          # rough 95th percentile offset

        lower, upper = cqr_conformal(predict_lower, predict_upper,
                                      X_cal, Y_cal, X_test; alpha=alpha)
        cov = conformal_coverage(lower, upper, Y_test)

        @test cov >= 1 - alpha - 0.05                # coverage guarantee
    end

    # ------------------------------------------------------------------ #
    # Integration: linear regression with multiple random seeds
    # ------------------------------------------------------------------ #
    @testset "Integration: linear model with coverage across seeds" begin
        alpha = 0.1
        n_seeds = 10
        coverages = Float64[]

        for seed in 1:n_seeds
            rng = MersenneTwister(seed * 1000)
            n_cal, n_test = 500, 200

            X_cal = randn(rng, n_cal, 1)
            Y_cal = 2.0 .* X_cal .+ 1.0 .+ 0.5 .* randn(rng, n_cal, 1)
            X_test = randn(rng, n_test, 1)
            Y_test = 2.0 .* X_test .+ 1.0 .+ 0.5 .* randn(rng, n_test, 1)

            # Simple linear predictor (knows slope & intercept, ignores noise)
            predict_fn(X) = 2.0 .* X .+ 1.0

            lower, upper = split_conformal(predict_fn, X_cal, Y_cal, X_test; alpha=alpha)
            push!(coverages, conformal_coverage(lower, upper, Y_test))
        end

        # Every seed should achieve >= 1-alpha - small tolerance
        @test all(c -> c >= 1 - alpha - 0.07, coverages)
        # Mean coverage should be close to 90%
        @test mean(coverages) >= 0.88
        @test mean(coverages) <= 0.97
    end

end
