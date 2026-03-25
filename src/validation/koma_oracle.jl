"""
KomaMRI validation oracle.

Validates our forward models against KomaMRI Bloch simulation
as an independent ground truth.

KomaMRI is an optional dependency -- loaded at runtime via `@eval`.
MRIBuilder (a declared dependency) is used for PGSE gradient design.
"""

using Statistics, LinearAlgebra, Random, Printf

# ------------------------------------------------------------------ #
# PGSE gradient design helpers (pure Julia, no external deps)
# ------------------------------------------------------------------ #

"""Proton gyromagnetic ratio in rad/(s*T)."""
const GAMMA_PROTON = 2.6752218744e8

"""
    pgse_gradient_strength(b, delta, Delta)

Compute the gradient strength G (T/m) required to achieve a target
b-value (s/m^2) for a pulsed-gradient spin-echo (PGSE) sequence with
gradient pulse duration `delta` (s) and diffusion time `Delta` (s).

Uses the Stejskal-Tanner equation:
    b = gamma^2 * G^2 * delta^2 * (Delta - delta/3)
"""
function pgse_gradient_strength(b::Float64, delta::Float64, Delta::Float64)
    if b <= 0
        return 0.0
    end
    denom = GAMMA_PROTON^2 * delta^2 * (Delta - delta / 3)
    @assert denom > 0 "Invalid PGSE timing: Delta must be > delta/3"
    return sqrt(b / denom)
end

"""
    pgse_bvalue(G, delta, Delta)

Compute the b-value (s/m^2) for a PGSE sequence with gradient strength
G (T/m), pulse duration `delta` (s), and diffusion time `Delta` (s).
"""
function pgse_bvalue(G::Float64, delta::Float64, Delta::Float64)
    return GAMMA_PROTON^2 * G^2 * delta^2 * (Delta - delta / 3)
end

# ------------------------------------------------------------------ #
# Validation result struct
# ------------------------------------------------------------------ #

"""
    KomaValidationResult

Structured result from KomaMRI cross-validation.

# Fields
- `passed::Bool`: true if all relative errors are below tolerance
- `D_values::Vector{Float64}`: diffusion coefficients tested (m^2/s)
- `b_values::Vector{Float64}`: b-values tested (s/m^2)
- `analytical_signals::Dict`: analytical S(b) per D
- `koma_signals::Dict`: KomaMRI simulated S(b) per D
- `relative_errors::Dict`: |S_koma - S_ana| / S_ana per D
- `max_error::Float64`: worst-case relative error across all conditions
- `mean_error::Float64`: mean relative error across all conditions
"""
struct KomaValidationResult
    passed::Bool
    D_values::Vector{Float64}
    b_values::Vector{Float64}
    analytical_signals::Dict{Float64, Vector{Float64}}
    koma_signals::Dict{Float64, Vector{Float64}}
    relative_errors::Dict{Float64, Vector{Float64}}
    max_error::Float64
    mean_error::Float64
end

# ------------------------------------------------------------------ #
# KomaMRI runtime helpers (defined via @eval after loading KomaMRI)
# ------------------------------------------------------------------ #

"""
    _ensure_koma_helpers!()

Load KomaMRI and define helper functions that depend on KomaMRI types.
Returns `true` if KomaMRI was loaded successfully, `false` otherwise.
"""
function _ensure_koma_helpers!()
    try
        @eval using KomaMRI
    catch e
        @warn "KomaMRI not available, skipping validation" exception = e
        return false
    end

    # Define the PGSE sequence builder using KomaMRI types.
    # This must be @eval'd because KomaMRI types are not known at compile time.
    # All KomaMRI types are fully qualified to avoid dispatch conflicts.
    @eval function _koma_build_pgse(
        G::Float64, delta::Float64, Delta::Float64,
        direction::Vector{Float64};
        TR::Float64 = 3.0,
    )
        dir = direction ./ max(norm(direction), 1e-12)
        Gx, Gy, Gz = G .* dir

        rf_dur = 1e-4          # 0.1 ms hard pulse
        wait = max(Delta - delta, 1e-6)

        sys = KomaMRI.Scanner()

        # -- Block 1: 90-degree excitation --
        b1 = KomaMRI.PulseDesigner.RF_hard(pi / 2, rf_dur, sys = sys)

        # -- Block 2: first diffusion gradient lobe (duration delta) --
        gx1 = KomaMRI.Grad(Gx, delta)
        gy1 = KomaMRI.Grad(Gy, delta)
        gz1 = KomaMRI.Grad(Gz, delta)
        b2 = KomaMRI.Sequence([gx1;; gy1;; gz1])

        # -- Block 3: dead time between gradient lobes --
        b3 = KomaMRI.Sequence([KomaMRI.Grad(0.0, wait);; KomaMRI.Grad(0.0, wait);; KomaMRI.Grad(0.0, wait)])

        # -- Block 4: 180-degree refocusing pulse --
        b4 = KomaMRI.PulseDesigner.RF_hard(pi, rf_dur, sys = sys)

        # -- Block 5: second diffusion gradient lobe --
        gx2 = KomaMRI.Grad(Gx, delta)
        gy2 = KomaMRI.Grad(Gy, delta)
        gz2 = KomaMRI.Grad(Gz, delta)
        b5 = KomaMRI.Sequence([gx2;; gy2;; gz2])

        # -- Block 6: ADC readout (single sample at echo) --
        adc_dur = 1e-4
        adc = KomaMRI.ADC(1, adc_dur)
        # Sequence block with ADC: zero gradients as carrier
        b6 = KomaMRI.Sequence(
            [KomaMRI.Grad(0.0, adc_dur);; KomaMRI.Grad(0.0, adc_dur);; KomaMRI.Grad(0.0, adc_dur)],
            reshape(KomaMRI.RF[], 0, 1),
            [adc],
        )

        # -- Block 7: dead time to fill TR --
        elapsed = 2 * rf_dur + 2 * delta + wait + adc_dur
        dead = max(TR - elapsed, 1e-4)
        b7 = KomaMRI.Sequence([KomaMRI.Grad(0.0, dead);; KomaMRI.Grad(0.0, dead);; KomaMRI.Grad(0.0, dead)])

        return b1 + b2 + b3 + b4 + b5 + b6 + b7
    end

    # Define the phantom builder.
    @eval function _koma_build_phantom(D::Float64)
        return KomaMRI.Phantom(
            x = [0.0],
            y = [0.0],
            z = [0.0],
            ρ  = [1.0],
            T1 = [10.0],    # very long T1 to avoid T1 weighting
            T2 = [1.0],     # long T2 to minimize T2 decay
            Dλ1 = [D],
            Dλ2 = [D],
            Dθ  = [0.0],
        )
    end

    # Define the simulation runner.
    # Use fully-qualified KomaMRI.simulate to avoid dispatch ambiguity
    # with DMI.simulate (which operates on forward models).
    @eval function _koma_simulate(obj, seq)
        sys = KomaMRI.Scanner()
        sim_params = Dict{String, Any}(
            "return_type" => "raw",
            "Nblocks"     => 1,
        )
        raw = KomaMRI.simulate(obj, seq, sys; sim_params = sim_params)
        # raw is typically a matrix of complex values (n_adc_samples x n_coils).
        # For a single-spin phantom with one ADC sample, extract the magnitude.
        return abs(raw[1])
    end

    return true
end

# ------------------------------------------------------------------ #
# Main validation: free isotropic diffusion
# ------------------------------------------------------------------ #

"""
    validate_free_diffusion_koma(; D_values, b_values, delta, Delta, tol, ...)

Cross-validate our analytical free-diffusion model `S = S0 * exp(-b*D)` against
KomaMRI Bloch simulation using PGSE sequences.

For each (D, b) pair:
1. Computes the required gradient strength G from the Stejskal-Tanner equation
2. Builds a KomaMRI PGSE sequence (90-G1-180-G2-ADC)
3. Creates a KomaMRI `Phantom` with isotropic diffusion coefficient D
4. Runs the Bloch simulation via `KomaMRI.simulate()`
5. Normalises by the b=0 signal and compares against `exp(-b*D)`

# Keyword arguments
- `D_values`: diffusion coefficients to test (m^2/s). Default: [1e-9, 2e-9, 3e-9].
- `b_values`: target b-values (s/m^2). Default: multi-shell 0--3000 s/mm^2.
- `delta`: gradient pulse duration (s). Default: 10 ms.
- `Delta`: diffusion time (s). Default: 40 ms.
- `tol`: relative error tolerance for pass/fail. Default: 0.05 (5%).
- `direction`: gradient direction (unit vector). Default: [1,0,0].
- `verbose`: print detailed results. Default: true.

# Returns
`KomaValidationResult` with signals, errors, and pass/fail; or `nothing` if
KomaMRI is not available.
"""
function validate_free_diffusion_koma(;
    D_values::Vector{Float64} = [1.0e-9, 2.0e-9, 3.0e-9],
    b_values::Vector{Float64} = [0.0, 500e6, 1000e6, 2000e6, 3000e6],
    delta::Float64 = 10e-3,   # 10 ms gradient pulse
    Delta::Float64 = 40e-3,   # 40 ms diffusion time
    tol::Float64 = 0.05,
    direction::Vector{Float64} = [1.0, 0.0, 0.0],
    verbose::Bool = true,
)
    # --- Conditionally load KomaMRI and define helpers ---
    _ensure_koma_helpers!() || return nothing

    if verbose
        println("=" ^ 60)
        println("KomaMRI Validation: Free Isotropic Diffusion")
        @printf("  PGSE timing: delta = %.1f ms, Delta = %.1f ms\n", delta * 1e3, Delta * 1e3)
        println("  Gradient direction: $direction")
        println("=" ^ 60)
    end

    # Storage
    analytical_signals = Dict{Float64, Vector{Float64}}()
    koma_signals       = Dict{Float64, Vector{Float64}}()
    relative_errors    = Dict{Float64, Vector{Float64}}()
    all_errors         = Float64[]

    for D in D_values
        analytical_signals[D] = Float64[]
        koma_signals[D]       = Float64[]
        relative_errors[D]    = Float64[]

        # Build KomaMRI phantom with isotropic diffusion D
        obj = @eval _koma_build_phantom($D)

        if verbose
            @printf("  D = %.2e m^2/s:\n", D)
        end

        for b in b_values
            # Analytical prediction: S/S0 = exp(-bD)
            S_analytical = exp(-b * D)

            # Gradient strength from Stejskal-Tanner
            G = pgse_gradient_strength(b, delta, Delta)

            # Sanity-check round-trip
            b_check = pgse_bvalue(G, delta, Delta)
            @assert abs(b_check - b) / max(b, 1.0) < 1e-6 "b-value round-trip failed: got $b_check, expected $b"

            if G > 0.3 && verbose
                @printf("    WARNING: G = %.1f mT/m exceeds typical scanner limits\n", G * 1e3)
            end

            # Build and run PGSE sequence in KomaMRI
            seq = @eval _koma_build_pgse($G, $delta, $Delta, $direction)
            S_koma = @eval _koma_simulate($obj, $seq)

            push!(analytical_signals[D], S_analytical)
            push!(koma_signals[D], Float64(S_koma))
        end

        # Normalise KomaMRI signals by the b=0 measurement
        S0_koma = koma_signals[D][1]
        if S0_koma > 1e-12
            koma_signals[D] ./= S0_koma
        else
            if verbose
                @warn "b=0 signal near zero for D=$(D); normalisation skipped"
            end
        end

        # Compute relative errors
        for (i, b) in enumerate(b_values)
            S_ana = analytical_signals[D][i]
            S_sim = koma_signals[D][i]
            rel_err = if S_ana > 1e-10
                abs(S_sim - S_ana) / S_ana
            else
                abs(S_sim - S_ana)  # both near zero
            end
            push!(relative_errors[D], rel_err)
            push!(all_errors, rel_err)

            if verbose
                b_smm2 = b / 1e6
                status = rel_err > tol ? "FAIL" : "ok"
                @printf("    b = %6.0f s/mm^2: analytical = %.6f  koma = %.6f  rel_err = %.4f%%  %s\n",
                        b_smm2, S_ana, S_sim, rel_err * 100, status)
            end
        end
    end

    max_err  = isempty(all_errors) ? 0.0 : maximum(all_errors)
    mean_err = isempty(all_errors) ? 0.0 : mean(all_errors)
    passed   = max_err <= tol

    if verbose
        println("-" ^ 60)
        @printf("  Max  relative error: %.4f%%  (tolerance: %.1f%%)\n", max_err * 100, tol * 100)
        @printf("  Mean relative error: %.4f%%\n", mean_err * 100)
        println("  Result: ", passed ? "PASSED" : "FAILED")
        println("=" ^ 60)
    end

    return KomaValidationResult(
        passed, D_values, b_values,
        analytical_signals, koma_signals, relative_errors,
        max_err, mean_err,
    )
end

# ------------------------------------------------------------------ #
# Signal property validation using module forward models
# ------------------------------------------------------------------ #

"""
    validate_signal_properties_koma(forward_model; n_test, seed)

Validate that a forward model from this module (e.g., `BallStickModel`,
`NODDIModel`) produces physically reasonable signals across random
microstructure parameter configurations.

Checks for every random draw:
1. All signal values >= 0 (physical: no negative magnetisation magnitude)
2. All signal values <= 1 (normalised signal cannot exceed S0)
3. Signal at b=0 equals 1.0 (normalisation identity)
4. Mean signal per shell decreases with b-value (diffusion attenuation)

Uses the module's own `simulate()` dispatch -- no KomaMRI dependency.

# Arguments
- `forward_model`: a model struct with a `simulate(model, params)` method
  (e.g., `BallStickModel`, `NODDIModel`)
- `n_test`: number of random configurations (default: 100)
- `seed`: RNG seed for reproducibility (default: 42)

# Returns
`true` if all tests pass, `false` otherwise.
"""
function validate_signal_properties_koma(forward_model; n_test::Int = 100, seed::Int = 42)
    rng = MersenneTwister(seed)

    println("=" ^ 60)
    println("Signal Property Validation ($n_test random configurations)")
    println("  Model type: $(typeof(forward_model))")
    println("=" ^ 60)

    n_pass = 0
    n_fail = 0
    failure_details = String[]

    for i in 1:n_test
        # Generate random parameters via multiple dispatch
        params = _random_params(rng, forward_model)

        # Use the module's forward model
        signal = simulate(forward_model, params)

        # --- Physics checks ---
        all_positive = all(signal .>= -1e-10)
        all_bounded  = all(signal .<= 1.0 + 1e-10)

        # b=0 signal should be 1.0
        b0_mask = forward_model.bvalues .< 100e6
        b0_correct = if any(b0_mask)
            all(isapprox.(signal[b0_mask], 1.0, atol = 1e-8))
        else
            true
        end

        # Mean signal should decrease (or stay flat) with increasing b-shell
        monotonic = true
        if length(unique(forward_model.bvalues)) > 1
            shells = sort(unique(forward_model.bvalues))
            prev_mean = Inf
            for s in shells
                mask = abs.(forward_model.bvalues .- s) .< 1e6
                if any(mask)
                    shell_mean = mean(signal[mask])
                    if shell_mean > prev_mean + 0.05
                        monotonic = false
                    end
                    prev_mean = shell_mean
                end
            end
        end

        if all_positive && all_bounded && b0_correct && monotonic
            n_pass += 1
        else
            n_fail += 1
            if length(failure_details) < 5
                detail = @sprintf(
                    "  FAIL #%d: positive=%s bounded=%s b0=%s monotonic=%s",
                    i, all_positive, all_bounded, b0_correct, monotonic,
                )
                push!(failure_details, detail)
            end
        end
    end

    pass_rate = 100.0 * n_pass / n_test
    @printf("  Passed: %d/%d (%.1f%%)\n", n_pass, n_test, pass_rate)
    for detail in failure_details
        println(detail)
    end

    return n_pass == n_test
end

# ------------------------------------------------------------------ #
# Random parameter generators (multiple dispatch by model type)
# ------------------------------------------------------------------ #

"""
    _random_params(rng, model::BallStickModel)

Random physically-plausible parameters for Ball+2Stick model.
Returns `[d_ball, d_stick, f1, f2, mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]`.
"""
function _random_params(rng, model::BallStickModel)
    d_ball  = 1.0e-9 + rand(rng) * 2.5e-9
    d_stick = 0.5e-9 + rand(rng) * 2.0e-9
    f1 = 0.1 + rand(rng) * 0.6
    f2 = 0.05 + rand(rng) * min(0.4, 0.9 - f1)
    mu1 = randn(rng, 3); mu1 ./= norm(mu1)
    mu2 = randn(rng, 3); mu2 ./= norm(mu2)
    return [d_ball, d_stick, f1, f2, mu1..., mu2...]
end

"""
    _random_params(rng, model::NODDIModel)

Random physically-plausible parameters for NODDI model.
Returns `[f_intra, f_iso, kappa, d_par, mu_x, mu_y, mu_z]`.
"""
function _random_params(rng, model::NODDIModel)
    f_intra = 0.1 + rand(rng) * 0.7
    f_iso   = rand(rng) * min(0.3, 0.9 - f_intra)
    kappa   = 0.5 + rand(rng) * 30.0
    d_par   = 1.0e-9 + rand(rng) * 2.0e-9
    mu = randn(rng, 3); mu ./= norm(mu)
    return [f_intra, f_iso, kappa, d_par, mu...]
end

"""
    _random_params(rng, model)

Fallback: generates Ball+2Stick-like parameters for unrecognised model types.
"""
function _random_params(rng, model)
    d_ball  = 1.0e-9 + rand(rng) * 2.5e-9
    d_stick = 0.5e-9 + rand(rng) * 2.0e-9
    f1 = 0.1 + rand(rng) * 0.6
    f2 = 0.05 + rand(rng) * min(0.4, 0.9 - f1)
    mu1 = randn(rng, 3); mu1 ./= norm(mu1)
    mu2 = randn(rng, 3); mu2 ./= norm(mu2)
    return [d_ball, d_stick, f1, f2, mu1..., mu2...]
end
