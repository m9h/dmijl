"""
    KomaMRI MR-ARFI Bloch simulation.

Builds KomaMRI Phantom objects with Motion-encoded tissue displacement
and MR-ARFI pulse sequences (bipolar MSG + GRE readout), then runs
Bloch simulation to produce ground-truth phase maps.

KomaMRI is an optional dependency loaded at runtime via `@eval`,
following the same pattern as `validation/koma_oracle.jl`.

References:
    - Kaye, Chen, Pauly (2011). MRM 65(3):738-743.
    - Castets et al. (2024). KomaMRI motion simulation. MRM.
"""

# ------------------------------------------------------------------ #
# Runtime KomaMRI loading
# ------------------------------------------------------------------ #

const _KOMA_ARFI_LOADED = Ref(false)

"""
    _ensure_koma_arfi!() -> Bool

Load KomaMRI and define ARFI-specific helpers.
Returns true if KomaMRI was loaded successfully.
"""
function _ensure_koma_arfi!()
    _KOMA_ARFI_LOADED[] && return true

    try
        @eval using KomaMRI
    catch e
        @warn "KomaMRI not available; Bloch ARFI simulation disabled" exception = e
        return false
    end

    # --------------------------------------------------------------- #
    # Phantom builder with displacement-encoded Motion
    # --------------------------------------------------------------- #

    @eval function _koma_build_arfi_phantom(
        positions::Matrix{Float64},       # (N, 3) spin positions in metres
        T1::Vector{Float64},              # seconds
        T2::Vector{Float64},              # seconds
        PD::Vector{Float64},              # proton density (0-1)
        displacement_z::Vector{Float64},  # per-spin displacement (metres)
        fus_onset::Float64,               # when displacement starts (seconds)
        fus_duration::Float64,            # displacement duration (seconds)
    )
        N = size(positions, 1)
        x = positions[:, 1]
        y = positions[:, 2]
        z = positions[:, 3]

        # Build per-spin Motion using KomaMRI's Translate action.
        #
        # Motion model: spins are stationary, then translate by
        # displacement_z during the FUS window, then return to rest.
        #
        # TimeCurve: piecewise-linear temporal profile
        #   t=0 -> rest
        #   t=fus_onset -> begin displacement
        #   t=fus_onset+fus_duration -> end displacement
        #   t=end -> rest
        #
        # We create a MotionList with one Motion per unique displacement
        # value, or use a single Translate if all are equal.

        # Determine total sequence time for TimeCurve normalization
        # Use a reasonable upper bound
        t_total = max(fus_onset + fus_duration + 0.1, 1.0)

        # Temporal profile nodes (normalized to [0, 1])
        t_nodes = Float64[
            0.0,
            max(fus_onset - 1e-4, 0.0) / t_total,
            fus_onset / t_total,
            (fus_onset + fus_duration) / t_total,
            min((fus_onset + fus_duration + 1e-4), t_total) / t_total,
            1.0,
        ]
        # Ensure strictly increasing
        for i in 2:length(t_nodes)
            if t_nodes[i] <= t_nodes[i-1]
                t_nodes[i] = t_nodes[i-1] + 1e-6
            end
        end

        # Build motions for each spin
        # Group spins by displacement to reduce Motion objects
        motions = []

        # Try to use Translate + SpinRange for efficiency
        # If all displacements are the same, one Motion suffices
        unique_disps = unique(displacement_z)

        if length(unique_disps) == 1 && unique_disps[1] != 0.0
            # All spins have same displacement
            dz = unique_disps[1]
            tc = KomaMRI.TimeCurve(t_nodes, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
            motion = KomaMRI.Motion(
                KomaMRI.Translate(0.0, 0.0, dz),
                tc,
                KomaMRI.AllSpins(),
            )
            push!(motions, motion)
        elseif all(displacement_z .== 0.0)
            # No motion
        else
            # Per-spin or grouped motions
            # Group by displacement value (binned to reduce count)
            for dz_val in unique_disps
                dz_val == 0.0 && continue
                mask = displacement_z .== dz_val
                spin_indices = findall(mask)

                tc = KomaMRI.TimeCurve(t_nodes, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

                for idx in spin_indices
                    m = KomaMRI.Motion(
                        KomaMRI.Translate(0.0, 0.0, dz_val),
                        tc,
                        KomaMRI.SpinRange(idx:idx),
                    )
                    push!(motions, m)
                end
            end
        end

        motion_field = if isempty(motions)
            KomaMRI.NoMotion()
        elseif length(motions) == 1
            motions[1]
        else
            KomaMRI.MotionList(motions...)
        end

        return KomaMRI.Phantom(;
            x = x, y = y, z = z,
            ρ = PD,
            T1 = T1,
            T2 = T2,
            motion = motion_field,
        )
    end

    # --------------------------------------------------------------- #
    # MR-ARFI sequence builder (bipolar MSG + GRE readout)
    # --------------------------------------------------------------- #

    @eval function _koma_build_arfi_sequence(params::ARFISequenceParams)
        sys = KomaMRI.Scanner()
        dir = params.encoding_direction
        dir = dir ./ max(norm(dir), 1e-12)

        G = params.msg_amplitude
        rf_dur = 1e-4  # 0.1 ms hard pulse (fast excitation)

        # Block 1: RF excitation
        b1 = KomaMRI.PulseDesigner.RF_hard(params.flip_angle, rf_dur, sys = sys)

        # Block 2: MSG lobe 1 (positive polarity)
        gx1 = KomaMRI.Grad(G * dir[1], params.msg_duration)
        gy1 = KomaMRI.Grad(G * dir[2], params.msg_duration)
        gz1 = KomaMRI.Grad(G * dir[3], params.msg_duration)
        b2 = KomaMRI.Sequence([gx1;; gy1;; gz1])

        # Block 3: FUS window (dead time — ultrasound fires here)
        b3 = KomaMRI.Sequence([
            KomaMRI.Grad(0.0, params.fus_duration);;
            KomaMRI.Grad(0.0, params.fus_duration);;
            KomaMRI.Grad(0.0, params.fus_duration)
        ])

        # Block 4: MSG lobe 2 (negative polarity — bipolar pair)
        gx2 = KomaMRI.Grad(-G * dir[1], params.msg_duration)
        gy2 = KomaMRI.Grad(-G * dir[2], params.msg_duration)
        gz2 = KomaMRI.Grad(-G * dir[3], params.msg_duration)
        b4 = KomaMRI.Sequence([gx2;; gy2;; gz2])

        # Block 5: GRE readout (ADC)
        adc_dur = 1e-4  # short ADC for single sample
        adc = KomaMRI.ADC(1, adc_dur)
        b5 = KomaMRI.Sequence(
            [KomaMRI.Grad(0.0, adc_dur);; KomaMRI.Grad(0.0, adc_dur);; KomaMRI.Grad(0.0, adc_dur)],
            reshape(KomaMRI.RF[], 0, 1),
            [adc],
        )

        # Block 6: TR dead time
        elapsed = rf_dur + 2 * params.msg_duration + params.fus_duration + adc_dur
        dead = max(params.tr - elapsed, 1e-4)
        b6 = KomaMRI.Sequence([
            KomaMRI.Grad(0.0, dead);;
            KomaMRI.Grad(0.0, dead);;
            KomaMRI.Grad(0.0, dead)
        ])

        return b1 + b2 + b3 + b4 + b5 + b6
    end

    # --------------------------------------------------------------- #
    # Simulation runner
    # --------------------------------------------------------------- #

    @eval function _koma_simulate_arfi(phantom, sequence)
        sys = KomaMRI.Scanner()
        sim_params = Dict{String, Any}(
            "return_type" => "raw",
            "Nblocks"     => 1,
        )
        raw = KomaMRI.simulate(phantom, sequence, sys; sim_params = sim_params)
        return raw
    end

    _KOMA_ARFI_LOADED[] = true
    return true
end

# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

"""
    build_arfi_phantom(labels, displacement, grid_spacing; fus_onset, fus_duration)

Build a KomaMRI Phantom from tissue labels and displacement field.

Maps labels to MR properties (T1, T2, PD), creates spin positions
on a grid, assigns per-spin displacement as KomaMRI Motion objects,
and filters background voxels (PD=0).

# Arguments
- `labels`: integer tissue label array (2D or 3D)
- `displacement`: displacement field (metres), same shape as labels
- `grid_spacing`: voxel size (metres)
- `fus_onset`: FUS pulse onset time (seconds), default 10 ms
- `fus_duration`: FUS pulse duration (seconds), default 10 ms

# Returns
KomaMRI Phantom with Motion-encoded displacement, or `nothing` if
KomaMRI is not available.
"""
function build_arfi_phantom(
    labels::AbstractArray{<:Integer},
    displacement::AbstractArray{<:Real},
    grid_spacing::Float64;
    fus_onset::Float64 = 10e-3,
    fus_duration::Float64 = 10e-3,
)
    _ensure_koma_arfi!() || return nothing

    # Map labels to MR properties
    T1, T2, PD = map_labels_to_mr(labels)

    # Generate spin positions on grid
    flat_labels = vec(labels)
    flat_disp = vec(Float64.(displacement))

    ndim = ndims(labels)
    if ndim == 2
        ny, nx = size(labels)
        positions = zeros(Float64, length(flat_labels), 3)
        idx = 1
        for iy in 1:ny, ix in 1:nx
            positions[idx, 1] = (ix - 1) * grid_spacing
            positions[idx, 2] = (iy - 1) * grid_spacing
            positions[idx, 3] = 0.0
            idx += 1
        end
    elseif ndim == 3
        nz, ny, nx = size(labels)
        positions = zeros(Float64, length(flat_labels), 3)
        idx = 1
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            positions[idx, 1] = (ix - 1) * grid_spacing
            positions[idx, 2] = (iy - 1) * grid_spacing
            positions[idx, 3] = (iz - 1) * grid_spacing
            idx += 1
        end
    else
        error("labels must be 2D or 3D, got $(ndim)D")
    end

    # Filter background spins (PD = 0)
    mask = PD .> 0
    positions_filt = positions[mask, :]
    T1_filt = T1[mask]
    T2_filt = T2[mask]
    PD_filt = PD[mask]
    disp_filt = flat_disp[mask]

    return @eval _koma_build_arfi_phantom(
        $positions_filt, $T1_filt, $T2_filt, $PD_filt,
        $disp_filt, $fus_onset, $fus_duration,
    )
end

"""
    build_arfi_sequence(seq_params::ARFISequenceParams)

Build a KomaMRI MR-ARFI pulse sequence.

Structure: RF exc → MSG1(+) → [FUS window] → MSG2(-) → GRE ADC

Returns KomaMRI Sequence, or `nothing` if KomaMRI is not available.
"""
function build_arfi_sequence(seq_params::ARFISequenceParams)
    _ensure_koma_arfi!() || return nothing
    return @eval _koma_build_arfi_sequence($seq_params)
end

"""
    simulate_arfi_koma(intensity, labels, seq_params, grid_spacing;
                       fus_onset=10e-3, fus_duration=10e-3) -> ARFIResult

Full MR-ARFI simulation with KomaMRI Bloch equations.

1. Compute displacement from intensity (radiation force + spectral solve)
2. Build KomaMRI phantom with displacement-encoded motion
3. Build MR-ARFI sequence (bipolar MSG)
4. Run Bloch simulation
5. Extract phase and recover displacement

Also runs the analytical pipeline for comparison.

Returns `nothing` if KomaMRI is not available.
"""
function simulate_arfi_koma(
    intensity::AbstractArray{<:Real},
    labels::AbstractArray{<:Integer},
    seq_params::ARFISequenceParams,
    grid_spacing::Float64;
    fus_onset::Float64 = 10e-3,
    fus_duration::Float64 = 10e-3,
)
    _ensure_koma_arfi!() || return nothing

    # Step 1: Analytical chain (radiation force -> displacement -> phase)
    analytical = simulate_arfi_analytical(intensity, labels, seq_params, grid_spacing)

    # Step 2: Build KomaMRI phantom with displacement motion
    phantom = build_arfi_phantom(
        labels, analytical.displacement, grid_spacing;
        fus_onset = fus_onset, fus_duration = fus_duration,
    )
    phantom === nothing && return nothing

    # Step 3: Build MR-ARFI sequence
    sequence = build_arfi_sequence(seq_params)
    sequence === nothing && return nothing

    # Step 4: Run Bloch simulation
    raw_signal = @eval _koma_simulate_arfi($phantom, $sequence)

    # Step 5: Extract phase from complex signal
    koma_phase = if raw_signal !== nothing
        angle.(raw_signal)
    else
        nothing
    end

    # Step 6: Build result
    return ARFIResult(
        analytical.displacement,
        analytical.phase_map,
        analytical.radiation_force,
        analytical.shear_modulus,
        seq_params,
        raw_signal,
        koma_phase isa AbstractArray ? Float64.(vec(koma_phase)) : nothing,
        nothing,  # recovered displacement (would need spatial mapping)
    )
end

"""
    validate_arfi_single_spin(; displacement=5e-6, msg_amplitude=40e-3,
                                msg_duration=5e-3, tol=0.1) -> Bool

Validate MR-ARFI phase encoding with a single spin.

Creates one spin with known displacement, runs Bloch simulation,
and compares the resulting phase to the analytical prediction
dphi = gamma * G * delta * u.

Returns true if the phase error is within tolerance, or `nothing`
if KomaMRI is not available.
"""
function validate_arfi_single_spin(;
    displacement::Float64 = 5e-6,
    msg_amplitude::Float64 = 40e-3,
    msg_duration::Float64 = 5e-3,
    fus_duration::Float64 = 10e-3,
    tol::Float64 = 0.1,
    verbose::Bool = true,
)
    _ensure_koma_arfi!() || return nothing

    seq_params = ARFISequenceParams(
        msg_amplitude = msg_amplitude,
        msg_duration = msg_duration,
        fus_duration = fus_duration,
    )

    # Analytical prediction
    phase_analytical = GAMMA_PROTON * msg_amplitude * msg_duration * displacement

    # Build single-spin phantom
    positions = [0.0 0.0 0.0]
    T1 = [10.0]   # very long to avoid T1 weighting
    T2 = [1.0]    # long T2
    PD = [1.0]
    disp_z = [displacement]

    phantom = @eval _koma_build_arfi_phantom(
        $positions, $T1, $T2, $PD, $disp_z, 10e-3, $fus_duration,
    )

    # Build sequence and simulate
    sequence = @eval _koma_build_arfi_sequence($seq_params)
    raw = @eval _koma_simulate_arfi($phantom, $sequence)

    # Also simulate without displacement for reference
    disp_zero = [0.0]
    phantom_ref = @eval _koma_build_arfi_phantom(
        $positions, $T1, $T2, $PD, $disp_zero, 10e-3, $fus_duration,
    )
    raw_ref = @eval _koma_simulate_arfi($phantom_ref, $sequence)

    # Phase difference
    phase_koma = angle(raw[1]) - angle(raw_ref[1])

    rel_error = abs(phase_koma - phase_analytical) / max(abs(phase_analytical), 1e-12)

    if verbose
        println("=" ^ 50)
        println("MR-ARFI Single-Spin Validation")
        println("  Displacement: $(displacement * 1e6) µm")
        println("  MSG: $(msg_amplitude * 1e3) mT/m × $(msg_duration * 1e3) ms")
        println("  Analytical phase: $(phase_analytical) rad")
        println("  KomaMRI phase:    $(phase_koma) rad")
        println("  Relative error:   $(rel_error * 100)%")
        println("  Result: ", rel_error <= tol ? "PASSED" : "FAILED")
        println("=" ^ 50)
    end

    return rel_error <= tol
end
