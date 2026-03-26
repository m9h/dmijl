using Test, LinearAlgebra, Random

@testset "RestrictedCylinder" begin

    # ---- Construction and traits ----

    @testset "construction and traits" begin
        cyl = RestrictedCylinder(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9, diameter=4e-6)
        @test cyl isa AbstractCompartment
        @test parameter_names(cyl) == (:mu, :lambda_par, :diameter)
        @test parameter_cardinality(cyl) == Dict(:mu => 3, :lambda_par => 1, :diameter => 1)
        @test nparams(cyl) == 5
        @test parameter_ranges(cyl)[:diameter] == (1e-7, 20e-6)
        @test parameter_ranges(cyl)[:lambda_par] == (0.0, 3.0e-9)
    end

    # ---- _reconstruct ----

    @testset "_reconstruct round-trip" begin
        cyl = RestrictedCylinder(mu=[0.3, 0.5, 0.8], lambda_par=1.2e-9, diameter=6e-6)
        p = [0.3, 0.5, 0.8, 1.2e-9, 6e-6]
        cyl2 = DMI._reconstruct(cyl, p)
        @test cyl2 isa RestrictedCylinder
        @test cyl2.mu == [0.3, 0.5, 0.8]
        @test cyl2.lambda_par == 1.2e-9
        @test cyl2.diameter == 6e-6
    end

    # ---- Requires delta and Delta ----

    @testset "errors when delta/Delta missing" begin
        cyl = RestrictedCylinder(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9, diameter=4e-6)
        bvals = [1000e6]
        bvecs = reshape([0.0 0.0 1.0], 1, 3)
        acq_no_timing = Acquisition(bvals, bvecs)
        @test_throws ArgumentError signal(cyl, acq_no_timing)
    end

    # Helper: build an acquisition with timing parameters
    function make_acq(bvals, bvecs; delta=12.9e-3, Delta=21.8e-3)
        Acquisition(bvals, bvecs, delta, Delta)
    end

    # ---- b=0 gives signal=1 ----

    @testset "b=0 gives signal=1" begin
        cyl = RestrictedCylinder(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9, diameter=4e-6)
        acq = make_acq([0.0], reshape([1.0 0.0 0.0], 1, 3))
        @test signal(cyl, acq)[1] ≈ 1.0 atol=1e-14
    end

    # ---- Parallel gradient: matches Stick (no perpendicular restriction) ----

    @testset "parallel gradient matches Stick signal" begin
        mu = [0.0, 0.0, 1.0]
        D = 1.7e-9
        cyl = RestrictedCylinder(mu=mu, lambda_par=D, diameter=4e-6)
        stick = C1Stick(mu=mu, lambda_par=D)

        bvals = [0.0, 500e6, 1000e6, 2000e6, 3000e6]
        bvecs = repeat([0.0 0.0 1.0], 5, 1)  # parallel to mu
        acq = make_acq(bvals, bvecs)

        sig_cyl = signal(cyl, acq)
        sig_stick = signal(stick, acq)
        @test sig_cyl ≈ sig_stick atol=1e-10
    end

    # ---- Smaller radius → less perpendicular attenuation (monotonic in small-R regime) ----

    @testset "smaller radius gives less perpendicular attenuation" begin
        mu = [0.0, 0.0, 1.0]
        D = 1.7e-9
        # Purely perpendicular gradient: S_par = 1, so signal = S_perp
        bvals = [1000e6]
        bvecs = reshape([1.0 0.0 0.0], 1, 3)
        acq = make_acq(bvals, bvecs)

        # In the small-R regime (2π*q*R < first zero of J1 ≈ 3.83),
        # smaller R → S_perp closer to 1
        sig_small = signal(RestrictedCylinder(mu=mu, lambda_par=D, diameter=1e-6), acq)[1]
        sig_large = signal(RestrictedCylinder(mu=mu, lambda_par=D, diameter=4e-6), acq)[1]

        # Smaller radius should have higher signal (less attenuation)
        @test sig_small > sig_large
    end

    # ---- Very small radius: fully restricted, S_perp ≈ 1 ----

    @testset "very small radius gives signal ≈ 1 (fully restricted perp)" begin
        mu = [0.0, 0.0, 1.0]
        D = 1.7e-9
        # Tiny radius: 0.1 μm — fully restricted perpendicular
        cyl = RestrictedCylinder(mu=mu, lambda_par=D, diameter=0.2e-6)

        rng = MersenneTwister(99)
        bvals = fill(1000e6, 10)
        # Mix of directions — some perpendicular to mu
        bvecs = [1.0 0.0 0.0;
                 0.0 1.0 0.0;
                 1/sqrt(2) 1/sqrt(2) 0.0;
                 1/sqrt(3) 1/sqrt(3) 1/sqrt(3);
                 0.0 0.0 1.0;
                 0.5 0.0 sqrt(3)/2;
                 0.0 0.5 sqrt(3)/2;
                 1/sqrt(2) 0.0 1/sqrt(2);
                 0.0 1/sqrt(2) 1/sqrt(2);
                 1.0 0.0 0.0]
        acq = make_acq(bvals, bvecs)

        sig_cyl = signal(cyl, acq)

        # For small R, S_perp ≈ 1, so total signal ≈ S_par = exp(-b * D * cos²θ)
        # which is the Stick signal
        stick = C1Stick(mu=mu, lambda_par=D)
        sig_stick = signal(stick, acq)
        @test sig_cyl ≈ sig_stick atol=1e-3
    end

    # ---- Signal bounded [0, 1] ----

    @testset "signal bounded [0, 1]" begin
        cyl = RestrictedCylinder(mu=[0.3, 0.5, 0.8], lambda_par=1.7e-9, diameter=4e-6)
        bvals = [0.0, 500e6, 1000e6, 2000e6, 3000e6]
        bvecs = [1.0 0.0 0.0;
                 0.0 1.0 0.0;
                 0.0 0.0 1.0;
                 1/sqrt(2) 1/sqrt(2) 0.0;
                 1/sqrt(3) 1/sqrt(3) 1/sqrt(3)]
        acq = make_acq(bvals, bvecs)
        sig = signal(cyl, acq)
        @test all(sig .>= -1e-12)
        @test all(sig .<= 1.0 + 1e-12)
    end

    # ---- Antipodal symmetry ----

    @testset "antipodal symmetry" begin
        D = 1.7e-9
        bvals = [500e6, 1000e6, 2000e6]
        bvecs = [1.0 0.0 0.0; 0.0 1.0 0.0; 1/sqrt(2) 0.0 1/sqrt(2)]
        acq = make_acq(bvals, bvecs)
        sig1 = signal(RestrictedCylinder(mu=[0.0, 0.0, 1.0], lambda_par=D, diameter=4e-6), acq)
        sig2 = signal(RestrictedCylinder(mu=[0.0, 0.0, -1.0], lambda_par=D, diameter=4e-6), acq)
        @test sig1 ≈ sig2 atol=1e-12
    end

    # ---- Perpendicular signal has restriction effect ----

    @testset "perpendicular signal shows restriction (less attenuation than free)" begin
        mu = [0.0, 0.0, 1.0]
        D = 1.7e-9
        R = 2e-6  # 2 μm radius → 4 μm diameter
        cyl = RestrictedCylinder(mu=mu, lambda_par=D, diameter=2*R)

        # Purely perpendicular gradient
        bvals = [1000e6, 2000e6, 3000e6]
        bvecs = repeat([1.0 0.0 0.0], 3, 1)
        acq = make_acq(bvals, bvecs)

        sig = signal(cyl, acq)
        # In perpendicular direction, stick gives 1.0 (no attenuation)
        # Restricted cylinder with finite R gives something between 0 and 1
        # but not exactly 1 (there IS some restriction/attenuation from Soderman)
        @test all(sig .> 0.0)
        @test all(sig .<= 1.0 + 1e-12)
        # At least one measurement should show some attenuation (not all exactly 1)
        # The parallel component is 0 for perpendicular gradient → S_par = 1
        # So the signal is purely S_perp from Soderman
        @test any(sig .< 1.0)
    end

    # ---- Numerical consistency: known Soderman value ----

    @testset "numerical spot-check Soderman formula" begin
        # Hand-compute a known value
        # mu along z, gradient along x (purely perpendicular)
        # b = 1000e6 s/m², delta = 10e-3 s, Delta = 20e-3 s, R = 3e-6 m
        # D_par = 1.7e-9 m²/s
        #
        # cos_angle = 0 (perpendicular), so b_par = 0, S_par = 1
        # q = sqrt(b / (Delta - delta/3)) / (2π) = sqrt(1000e6 / (20e-3 - 10e-3/3)) / (2π)
        #   = sqrt(1000e6 / 0.01667) / (2π) = sqrt(5.999e10) / (2π)
        #   = 244929.1... / 6.28318... = 38979.5... (1/m)
        # q_perp = q * sqrt(1 - 0²) = q = 38979.5 (1/m)
        # x = 2π * q_perp * R = 2π * 38979.5 * 3e-6 = 0.73515...
        # S_perp = |2*J1(x)/x|²

        using SpecialFunctions: besselj
        delta = 10e-3
        Delta = 20e-3
        b = 1000e6
        R = 3e-6
        q = sqrt(b / (Delta - delta/3)) / (2π)
        x = 2π * q * R
        s_perp_expected = (2 * besselj(1, x) / x)^2

        cyl = RestrictedCylinder(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9, diameter=2*R)
        acq = make_acq([b], reshape([1.0 0.0 0.0], 1, 3); delta=delta, Delta=Delta)
        sig = signal(cyl, acq)
        @test sig[1] ≈ s_perp_expected atol=1e-10
    end

end
