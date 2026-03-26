using Test, LinearAlgebra, Random

@testset "SphereGPD Compartment" begin

    # ---- Construction and Traits ----

    @testset "construction and traits" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        @test sphere isa AbstractCompartment
        @test parameter_names(sphere) == (:diameter, :D_intra)
        @test parameter_cardinality(sphere) == Dict(:diameter => 1, :D_intra => 1)
        @test nparams(sphere) == 2
        ranges = parameter_ranges(sphere)
        @test haskey(ranges, :diameter)
        @test haskey(ranges, :D_intra)
        # Diameter range: soma-like, say 2-30 um
        @test ranges[:diameter][1] >= 0.0
        @test ranges[:diameter][2] > 0.0
        # Diffusivity range
        @test ranges[:D_intra][1] >= 0.0
        @test ranges[:D_intra][2] > 0.0
    end

    @testset "_reconstruct" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        sphere2 = DMI._reconstruct(sphere, [12.0e-6, 1.5e-9])
        @test sphere2 isa SphereGPD
        @test sphere2.diameter == 12.0e-6
        @test sphere2.D_intra == 1.5e-9
    end

    # ---- Signal requires delta and Delta ----

    @testset "requires delta and Delta in Acquisition" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        bvals = [0.0, 1000e6, 2000e6]
        bvecs = repeat([1.0 0.0 0.0], 3, 1)
        # Acquisition without delta/Delta should error
        acq_no_timing = Acquisition(bvals, bvecs)
        @test acq_no_timing.delta === nothing
        @test_throws Exception signal(sphere, acq_no_timing)
    end

    # Helper: create acquisition with timing info
    function make_acq(bvals; delta=12.9e-3, Delta=21.8e-3)
        n = length(bvals)
        bvecs = repeat([1.0 0.0 0.0], n, 1)
        return Acquisition(bvals, bvecs, delta, Delta)
    end

    function make_acq_multishell(; delta=12.9e-3, Delta=21.8e-3)
        bvals = vcat(zeros(3), fill(1000e6, 10), fill(2000e6, 10), fill(3000e6, 10))
        n = length(bvals)
        rng = MersenneTwister(42)
        bvecs = randn(rng, n, 3)
        bvecs ./= sqrt.(sum(bvecs.^2, dims=2))
        return Acquisition(bvals, bvecs, delta, Delta)
    end

    # ---- b=0 gives signal=1 ----

    @testset "b=0 gives signal=1" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        acq = make_acq([0.0])
        sig = signal(sphere, acq)
        @test sig[1] ≈ 1.0 atol=1e-12
    end

    # ---- Signal is direction-independent (isotropic) ----

    @testset "signal is direction-independent (isotropic)" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        b = 2000e6
        bvals = [b, b, b]
        bvecs = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        acq = Acquisition(bvals, bvecs, 12.9e-3, 21.8e-3)
        sig = signal(sphere, acq)
        @test sig[1] ≈ sig[2] atol=1e-12
        @test sig[2] ≈ sig[3] atol=1e-12
    end

    # ---- Signal bounded [0, 1] ----

    @testset "signal bounded [0, 1]" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        acq = make_acq_multishell()
        sig = signal(sphere, acq)
        @test all(sig .>= -1e-12)
        @test all(sig .<= 1.0 + 1e-12)
    end

    # ---- R -> 0 limit: fully restricted, signal -> 1 ----

    @testset "small radius: signal approaches 1 (fully restricted)" begin
        # Very tiny sphere -> nearly no diffusion possible
        sphere = SphereGPD(diameter=0.1e-6, D_intra=2.0e-9)
        acq = make_acq([1000e6, 2000e6, 3000e6])
        sig = signal(sphere, acq)
        @test all(sig .> 0.95)
    end

    # ---- R -> inf limit: approaches Ball (free diffusion) ----

    @testset "large radius: signal approaches Ball (free diffusion)" begin
        # Very large sphere -> should approach exp(-bD)
        D = 2.0e-9
        sphere = SphereGPD(diameter=1000.0e-6, D_intra=D)
        ball = G1Ball(lambda_iso=D)
        bvals = [0.0, 500e6, 1000e6, 2000e6]
        acq = make_acq(bvals)
        acq_ball = Acquisition(bvals, repeat([1.0 0.0 0.0], length(bvals), 1))
        sig_sphere = signal(sphere, acq)
        sig_ball = signal(ball, acq_ball)
        @test sig_sphere ≈ sig_ball atol=0.01
    end

    # ---- Signal monotonically decreases with b-value ----

    @testset "signal decreases with increasing b-value" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        bvals = [0.0, 500e6, 1000e6, 2000e6, 3000e6, 5000e6]
        acq = make_acq(bvals)
        sig = signal(sphere, acq)
        for i in 2:length(sig)
            @test sig[i] <= sig[i-1] + 1e-12
        end
    end

    # ---- More restricted than free diffusion ----

    @testset "more restricted than Ball (less attenuation)" begin
        D = 2.0e-9
        sphere = SphereGPD(diameter=10.0e-6, D_intra=D)
        ball = G1Ball(lambda_iso=D)
        bvals = [500e6, 1000e6, 2000e6, 3000e6]
        acq = make_acq(bvals)
        acq_ball = Acquisition(bvals, repeat([1.0 0.0 0.0], length(bvals), 1))
        sig_sphere = signal(sphere, acq)
        sig_ball = signal(ball, acq_ball)
        # Restricted sphere should have higher signal (less attenuation)
        @test all(sig_sphere .>= sig_ball .- 1e-10)
    end

    # ---- Larger radius gives more attenuation ----

    @testset "larger radius gives more attenuation" begin
        acq = make_acq([2000e6])
        sig_small = signal(SphereGPD(diameter=5.0e-6, D_intra=2.0e-9), acq)
        sig_large = signal(SphereGPD(diameter=20.0e-6, D_intra=2.0e-9), acq)
        @test sig_large[1] < sig_small[1]
    end

    # ---- Output length matches acquisition ----

    @testset "output length matches acquisition" begin
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        acq = make_acq_multishell()
        sig = signal(sphere, acq)
        @test length(sig) == length(acq.bvalues)
    end

    # ---- Typical SANDI-like parameters give reasonable signal ----

    @testset "SANDI-like soma parameters" begin
        # Typical soma: ~10 um diameter, D_intra ~2 um^2/ms
        sphere = SphereGPD(diameter=10.0e-6, D_intra=2.0e-9)
        acq = make_acq([1000e6]; delta=12.9e-3, Delta=21.8e-3)
        sig = signal(sphere, acq)
        # Should be reasonably attenuated but not as much as free diffusion
        @test 0.3 < sig[1] < 1.0
    end

end
