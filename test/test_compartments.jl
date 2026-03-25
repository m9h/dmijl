using Test, LinearAlgebra, Random

@testset "Compartment Framework" begin

    # ---- Phase 1: Core Abstractions ----

    @testset "AbstractCompartment" begin
        @test isdefined(DMI, :AbstractCompartment)
        @test isabstracttype(AbstractCompartment)
    end

    @testset "Trait functions defined" begin
        @test isdefined(DMI, :parameter_names)
        @test isdefined(DMI, :parameter_ranges)
        @test isdefined(DMI, :parameter_cardinality)
        @test isdefined(DMI, :nparams)
    end

    @testset "signal() generic function" begin
        @test isdefined(DMI, :signal)
    end

    # ---- Phase 2: Signal Models ----

    @testset "G1Ball" begin
        @testset "construction and traits" begin
            ball = G1Ball(lambda_iso=2.0e-9)
            @test ball isa AbstractCompartment
            @test parameter_names(ball) == (:lambda_iso,)
            @test parameter_cardinality(ball) == Dict(:lambda_iso => 1)
            @test parameter_ranges(ball)[:lambda_iso] == (0.0, 3.0e-9)
            @test nparams(ball) == 1
        end

        @testset "signal matches analytical exp(-b*D)" begin
            D = 2.0e-9
            ball = G1Ball(lambda_iso=D)
            bvals = [0.0, 500e6, 1000e6, 2000e6, 3000e6]
            bvecs = repeat([1.0 0.0 0.0], 5, 1)
            acq = Acquisition(bvals, bvecs)
            sig = signal(ball, acq)
            expected = [exp(-b * D) for b in bvals]
            @test sig ≈ expected atol=1e-12
        end

        @testset "signal is direction-independent" begin
            ball = G1Ball(lambda_iso=1.5e-9)
            bvals = [1000e6, 1000e6, 1000e6]
            bvecs = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
            acq = Acquisition(bvals, bvecs)
            sig = signal(ball, acq)
            @test sig[1] ≈ sig[2] atol=1e-12
            @test sig[2] ≈ sig[3] atol=1e-12
        end

        @testset "b=0 gives signal=1" begin
            ball = G1Ball(lambda_iso=2.5e-9)
            acq = Acquisition([0.0], reshape([1.0 0.0 0.0], 1, 3))
            @test signal(ball, acq)[1] ≈ 1.0 atol=1e-15
        end
    end

    @testset "C1Stick" begin
        @testset "construction and traits" begin
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            @test stick isa AbstractCompartment
            @test parameter_names(stick) == (:mu, :lambda_par)
            @test parameter_cardinality(stick) == Dict(:mu => 3, :lambda_par => 1)
            @test nparams(stick) == 4
        end

        @testset "parallel gradient: full attenuation" begin
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            acq = Acquisition([1000e6], reshape([0.0 0.0 1.0], 1, 3))
            sig = signal(stick, acq)
            @test sig[1] ≈ exp(-1000e6 * 1.7e-9) atol=1e-10
        end

        @testset "perpendicular gradient: no attenuation" begin
            stick = C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9)
            acq = Acquisition([1000e6], reshape([1.0 0.0 0.0], 1, 3))
            sig = signal(stick, acq)
            @test sig[1] ≈ 1.0 atol=1e-10
        end

        @testset "antipodal symmetry" begin
            acq = hcp_like_acquisition()
            s1 = signal(C1Stick(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9), acq)
            s2 = signal(C1Stick(mu=[0.0, 0.0, -1.0], lambda_par=1.7e-9), acq)
            @test s1 ≈ s2 atol=1e-12
        end

        @testset "b=0 gives signal=1" begin
            stick = C1Stick(mu=[1.0, 0.0, 0.0], lambda_par=2.0e-9)
            acq = Acquisition([0.0], reshape([0.0 0.0 1.0], 1, 3))
            @test signal(stick, acq)[1] ≈ 1.0 atol=1e-15
        end

        @testset "signal bounded [0, 1]" begin
            acq = hcp_like_acquisition()
            sig = signal(C1Stick(mu=[0.3, 0.5, 0.8], lambda_par=1.7e-9), acq)
            @test all(sig .>= -1e-12)
            @test all(sig .<= 1.0 + 1e-12)
        end
    end

    @testset "G2Zeppelin" begin
        @testset "construction and traits" begin
            zep = G2Zeppelin(mu=[0.0, 0.0, 1.0], lambda_par=1.7e-9, lambda_perp=0.5e-9)
            @test zep isa AbstractCompartment
            @test parameter_names(zep) == (:mu, :lambda_par, :lambda_perp)
            @test nparams(zep) == 5
        end

        @testset "isotropic case degenerates to Ball" begin
            D = 1.5e-9
            zep = G2Zeppelin(mu=[0.0, 0.0, 1.0], lambda_par=D, lambda_perp=D)
            ball = G1Ball(lambda_iso=D)
            acq = hcp_like_acquisition()
            @test signal(zep, acq) ≈ signal(ball, acq) atol=1e-12
        end

        @testset "lambda_perp=0 degenerates to Stick" begin
            mu = [0.0, 0.0, 1.0]
            d_par = 1.7e-9
            zep = G2Zeppelin(mu=mu, lambda_par=d_par, lambda_perp=0.0)
            stick = C1Stick(mu=mu, lambda_par=d_par)
            acq = hcp_like_acquisition()
            @test signal(zep, acq) ≈ signal(stick, acq) atol=1e-12
        end

        @testset "signal bounded [0, 1]" begin
            zep = G2Zeppelin(mu=[0.3, 0.5, 0.8], lambda_par=1.7e-9, lambda_perp=0.5e-9)
            acq = hcp_like_acquisition()
            sig = signal(zep, acq)
            @test all(sig .>= -1e-12)
            @test all(sig .<= 1.0 + 1e-12)
        end

        @testset "rotation equivariance" begin
            rng = MersenneTwister(42)
            axis = randn(rng, 3); axis ./= norm(axis)
            angle = rand(rng) * 2pi
            K = [0 -axis[3] axis[2]; axis[3] 0 -axis[1]; -axis[2] axis[1] 0]
            R = I + sin(angle) * K + (1 - cos(angle)) * K^2

            acq = hcp_like_acquisition()
            mu = [0.0, 0.0, 1.0]
            mu_rot = R * mu
            g_rot = acq.gradient_directions * R'

            sig1 = signal(G2Zeppelin(mu=mu, lambda_par=1.7e-9, lambda_perp=0.5e-9), acq)
            acq_rot = Acquisition(acq.bvalues, g_rot)
            sig2 = signal(G2Zeppelin(mu=mu_rot, lambda_par=1.7e-9, lambda_perp=0.5e-9), acq_rot)
            @test sig1 ≈ sig2 atol=1e-8
        end
    end

    @testset "S1Dot" begin
        @testset "signal is always 1" begin
            dot = S1Dot()
            acq = hcp_like_acquisition()
            sig = signal(dot, acq)
            @test all(sig .== 1.0)
            @test length(sig) == length(acq.bvalues)
        end

        @testset "no parameters" begin
            dot = S1Dot()
            @test parameter_names(dot) == ()
            @test nparams(dot) == 0
        end
    end

    @testset "Cross-validate G1Ball vs BallStickModel" begin
        D = 2.0e-9
        acq = hcp_like_acquisition()
        # Old monolithic model: ball-only means f1=0, f2=0, d_stick irrelevant
        old_model = BallStickModel(acq.bvalues, acq.gradient_directions)
        # params: [d_ball, d_stick, f1, f2, mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]
        old_sig = simulate(old_model, [D, D, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        # New compartment
        new_sig = signal(G1Ball(lambda_iso=D), acq)
        @test old_sig ≈ new_sig atol=1e-12
    end
end
