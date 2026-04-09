"""
    Analytical MR-ARFI forward model.

End-to-end chain: intensity -> radiation force -> displacement -> phase.
All functions are Zygote-compatible for differentiable optimization.

References:
    - Kaye, Chen, Pauly (2011). MRM 65(3):738-743.
    - Kaye, Pauly (2013). MRM 69(3):724-733.
"""

# Reuse GAMMA_PROTON from koma_oracle.jl (already defined in the module)

"""
    predict_arfi_phase(displacement, seq_params) -> Array{Float64}

Predict MR-ARFI phase shift from tissue displacement (analytical).

    dphi = gamma * G_MSG * delta * u

# Arguments
- `displacement`: tissue displacement (metres), any shape
- `seq_params::ARFISequenceParams`: MR-ARFI sequence parameters

# Returns
Phase shift (radians), same shape as displacement.
"""
function predict_arfi_phase(displacement::AbstractArray{<:Real}, seq_params::ARFISequenceParams)
    enc = arfi_encoding_coefficient(seq_params)
    return enc .* displacement
end

"""
    predict_arfi_phase(displacement, msg_amplitude, msg_duration)

Convenience overload with explicit MSG parameters.
"""
function predict_arfi_phase(displacement::AbstractArray{<:Real},
                            msg_amplitude::Real, msg_duration::Real)
    enc = GAMMA_PROTON * msg_amplitude * msg_duration
    return enc .* displacement
end

"""
    recover_displacement_from_phase(phase_on, phase_off, seq_params) -> Array{Float64}

Recover tissue displacement from MR-ARFI phase difference.

    u = (phi_on - phi_off) / (gamma * G_MSG * delta)

# Arguments
- `phase_on`: phase image with FUS active (radians)
- `phase_off`: phase image without FUS (radians)
- `seq_params::ARFISequenceParams`: MR-ARFI sequence parameters

# Returns
Estimated displacement (metres).
"""
function recover_displacement_from_phase(
    phase_on::AbstractArray{<:Real},
    phase_off::AbstractArray{<:Real},
    seq_params::ARFISequenceParams,
)
    enc = arfi_encoding_coefficient(seq_params)
    return (phase_on .- phase_off) ./ enc
end

function recover_displacement_from_phase(
    phase_on::AbstractArray{<:Real},
    phase_off::AbstractArray{<:Real},
    msg_amplitude::Real,
    msg_duration::Real,
)
    enc = GAMMA_PROTON * msg_amplitude * msg_duration
    return (phase_on .- phase_off) ./ enc
end

"""
    arfi_encoding_coefficient(seq_params) -> Float64

Displacement encoding coefficient: gamma * G_MSG * delta (rad/m).

Typical value: 40 mT/m, 5 ms -> 53,506 rad/m.
"""
function arfi_encoding_coefficient(seq_params::ARFISequenceParams)
    return GAMMA_PROTON * seq_params.msg_amplitude * seq_params.msg_duration
end

"""
    arfi_encoding_sensitivity(seq_params) -> (encoding_coeff, min_displacement)

Compute displacement encoding coefficient and minimum detectable
displacement assuming a phase noise floor of 0.05 rad.

# Returns
- `encoding_coeff`: gamma * G * delta (rad/m)
- `min_displacement`: minimum detectable displacement (metres)
"""
function arfi_encoding_sensitivity(seq_params::ARFISequenceParams)
    enc = arfi_encoding_coefficient(seq_params)
    phase_noise = 0.05  # rad (single-shot GRE noise floor)
    min_disp = phase_noise / enc
    return (enc, min_disp)
end

# ------------------------------------------------------------------ #
# End-to-end analytical pipeline
# ------------------------------------------------------------------ #

"""
    simulate_arfi_analytical(intensity, labels, seq_params, grid_spacing) -> ARFIResult

End-to-end analytical MR-ARFI prediction (no Bloch simulation).

Chain: intensity -> radiation force -> spectral displacement -> phase.

# Arguments
- `intensity`: acoustic intensity field (W/m^2)
- `labels`: integer tissue labels (same spatial shape as intensity)
- `seq_params::ARFISequenceParams`: MR-ARFI sequence parameters
- `grid_spacing::Float64`: uniform grid spacing (metres)

# Returns
`ARFIResult` with displacement, phase, radiation force, and shear modulus fields.
"""
function simulate_arfi_analytical(
    intensity::AbstractArray{<:Real},
    labels::AbstractArray{<:Integer},
    seq_params::ARFISequenceParams,
    grid_spacing::Float64,
)
    # Step 1: Map labels to acoustic and mechanical properties
    sound_speed, _, attenuation_db = map_labels_to_acoustic(labels)
    attenuation_db = reshape(attenuation_db, size(labels))
    sound_speed = reshape(sound_speed, size(labels))

    # Step 2: Radiation force F = 2*alpha*I/c
    force = compute_radiation_force_from_db(
        Float64.(intensity), sound_speed, attenuation_db,
    )

    # Step 3: Shear modulus
    mu = map_labels_to_shear_modulus(labels)
    mu = reshape(mu, size(labels))

    # Step 4: Spectral displacement solve
    displacement = solve_displacement_spectral(force, mu, grid_spacing)

    # Step 5: MR-ARFI phase prediction
    phase_map = predict_arfi_phase(displacement, seq_params)

    return ARFIResult(
        displacement, phase_map, force, Float64.(mu),
        seq_params,
        nothing, nothing, nothing,
    )
end
