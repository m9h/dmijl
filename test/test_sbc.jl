using Test, Random, Statistics

@testset "Simulation-Based Calibration (SBC)" begin

    # ------------------------------------------------------------------ #
    # Unit tests: compute_rank
    # ------------------------------------------------------------------ #

    @testset "compute_rank" begin
        @testset "returns integer in [0, n_samples]" begin
            samples = randn(100)
            rank = DMI.compute_rank(0.0, samples)
            @test rank isa Integer
            @test 0 <= rank <= length(samples)
        end

        @testset "rank 0 when theta < all samples" begin
            samples = collect(1.0:10.0)
            @test DMI.compute_rank(-1.0, samples) == 0
        end

        @testset "rank n when theta > all samples" begin
            samples = collect(1.0:10.0)
            @test DMI.compute_rank(100.0, samples) == 10
        end

        @testset "correct rank for known position" begin
            samples = [1.0, 3.0, 5.0, 7.0, 9.0]
            # 4.0 is greater than 1.0 and 3.0 -> rank 2
            @test DMI.compute_rank(4.0, samples) == 2
        end

        @testset "uniform samples give roughly uniform ranks" begin
            rng = MersenneTwister(42)
            n_posterior = 99
            n_reps = 2000
            ranks = Int[]
            for _ in 1:n_reps
                theta_true = rand(rng)
                posterior_samples = rand(rng, n_posterior)
                push!(ranks, DMI.compute_rank(theta_true, posterior_samples))
            end
            # Ranks should be roughly Uniform(0, n_posterior)
            # Mean should be near n_posterior/2
            @test abs(mean(ranks) - n_posterior / 2) < 5.0
            # Std should be near sqrt((n_posterior+1)^2 - 1)/12 ~ n_posterior/sqrt(12)
            expected_std = sqrt((n_posterior + 1)^2 - 1) / sqrt(12)
            @test abs(std(ranks) - expected_std) < 3.0
        end
    end

    # ------------------------------------------------------------------ #
    # Unit tests: sbc_ranks
    # ------------------------------------------------------------------ #

    @testset "sbc_ranks" begin
        @testset "returns vector of length n_simulations" begin
            # Trivial setup: prior = N(0,1), simulator = identity, posterior = N(data, 1)
            prior_sampler = (rng) -> randn(rng)
            simulator = (theta, rng) -> theta  # data = theta
            posterior_sampler = (data, rng, n) -> data .+ randn(rng, n)

            n_sims = 50
            n_post = 30
            ranks = DMI.sbc_ranks(
                prior_sampler, simulator, posterior_sampler, n_sims;
                n_posterior_samples=n_post, rng=MersenneTwister(123)
            )
            @test length(ranks) == n_sims
            @test all(0 .<= ranks .<= n_post)
            @test eltype(ranks) <: Integer
        end
    end

    # ------------------------------------------------------------------ #
    # Unit tests: sbc_histogram
    # ------------------------------------------------------------------ #

    @testset "sbc_histogram" begin
        @testset "produces correct bin counts" begin
            ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            n_posterior = 9
            bin_edges, counts = DMI.sbc_histogram(ranks, n_posterior; n_bins=5)
            @test length(counts) == 5
            @test sum(counts) == length(ranks)
            @test length(bin_edges) == 6  # n_bins + 1 edges
        end

        @testset "uniform ranks produce roughly equal bin counts" begin
            rng = MersenneTwister(99)
            n_posterior = 99
            ranks = rand(rng, 0:n_posterior, 10000)
            _, counts = DMI.sbc_histogram(ranks, n_posterior; n_bins=10)
            # Each bin should have ~1000 counts
            @test all(counts .> 800)
            @test all(counts .< 1200)
        end
    end

    # ------------------------------------------------------------------ #
    # Unit tests: sbc_uniformity_test
    # ------------------------------------------------------------------ #

    @testset "sbc_uniformity_test" begin
        @testset "uniform ranks pass" begin
            rng = MersenneTwister(77)
            n_posterior = 99
            ranks = rand(rng, 0:n_posterior, 5000)
            p_value, is_calibrated = DMI.sbc_uniformity_test(ranks, n_posterior)
            @test p_value > 0.01  # should not reject uniformity
            @test is_calibrated == true
        end

        @testset "non-uniform ranks fail" begin
            # Highly concentrated ranks (all near 0 = U-shaped extremes)
            ranks = zeros(Int, 500)
            n_posterior = 99
            p_value, is_calibrated = DMI.sbc_uniformity_test(ranks, n_posterior)
            @test p_value < 0.01
            @test is_calibrated == false
        end
    end

    # ------------------------------------------------------------------ #
    # Integration: well-calibrated Gaussian posterior
    # ------------------------------------------------------------------ #

    @testset "Integration: well-calibrated Gaussian" begin
        # Setup: Gaussian likelihood with known sigma
        # Prior: theta ~ N(0, 10)
        # Likelihood: data | theta ~ N(theta, sigma^2)
        # Exact posterior: theta | data ~ N(posterior_mean, posterior_var)
        # where posterior_var = 1 / (1/prior_var + 1/sigma^2)
        #       posterior_mean = posterior_var * (prior_mean/prior_var + data/sigma^2)

        prior_mean = 0.0
        prior_var = 100.0  # wide prior
        sigma = 1.0        # observation noise

        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / sigma^2)
        posterior_std = sqrt(posterior_var)

        prior_sampler = (rng) -> prior_mean + sqrt(prior_var) * randn(rng)
        simulator = (theta, rng) -> theta + sigma * randn(rng)
        posterior_sampler = function (data, rng, n)
            post_mu = posterior_var * (prior_mean / prior_var + data / sigma^2)
            return post_mu .+ posterior_std .* randn(rng, n)
        end

        n_sims = 1000
        n_post = 99

        ranks = DMI.sbc_ranks(
            prior_sampler, simulator, posterior_sampler, n_sims;
            n_posterior_samples=n_post, rng=MersenneTwister(2024)
        )

        p_value, is_calibrated = DMI.sbc_uniformity_test(ranks, n_post)
        @test is_calibrated
        @test p_value > 0.01
    end

    # ------------------------------------------------------------------ #
    # Integration: deliberately miscalibrated posterior (too narrow)
    # ------------------------------------------------------------------ #

    @testset "Integration: miscalibrated Gaussian (too narrow)" begin
        # Use a posterior that is too confident (variance too small)
        # This produces a U-shaped rank histogram (ranks pile up at 0 and n_post)
        prior_mean = 0.0
        prior_var = 100.0
        sigma = 1.0

        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / sigma^2)

        prior_sampler = (rng) -> prior_mean + sqrt(prior_var) * randn(rng)
        simulator = (theta, rng) -> theta + sigma * randn(rng)

        # Deliberately WRONG: posterior variance 10x too small
        wrong_std = sqrt(posterior_var) / sqrt(10.0)
        posterior_sampler = function (data, rng, n)
            post_mu = posterior_var * (prior_mean / prior_var + data / sigma^2)
            return post_mu .+ wrong_std .* randn(rng, n)
        end

        n_sims = 1000
        n_post = 99

        ranks = DMI.sbc_ranks(
            prior_sampler, simulator, posterior_sampler, n_sims;
            n_posterior_samples=n_post, rng=MersenneTwister(2024)
        )

        p_value, is_calibrated = DMI.sbc_uniformity_test(ranks, n_post)
        @test !is_calibrated
        @test p_value < 0.01
    end

    # ------------------------------------------------------------------ #
    # Integration: miscalibrated posterior (biased mean)
    # ------------------------------------------------------------------ #

    @testset "Integration: miscalibrated Gaussian (biased)" begin
        prior_mean = 0.0
        prior_var = 100.0
        sigma = 1.0

        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / sigma^2)
        posterior_std = sqrt(posterior_var)

        prior_sampler = (rng) -> prior_mean + sqrt(prior_var) * randn(rng)
        simulator = (theta, rng) -> theta + sigma * randn(rng)

        # Deliberately WRONG: posterior mean is shifted by +3
        posterior_sampler = function (data, rng, n)
            post_mu = posterior_var * (prior_mean / prior_var + data / sigma^2) + 3.0
            return post_mu .+ posterior_std .* randn(rng, n)
        end

        n_sims = 1000
        n_post = 99

        ranks = DMI.sbc_ranks(
            prior_sampler, simulator, posterior_sampler, n_sims;
            n_posterior_samples=n_post, rng=MersenneTwister(2024)
        )

        p_value, is_calibrated = DMI.sbc_uniformity_test(ranks, n_post)
        @test !is_calibrated
        @test p_value < 0.01
    end
end
