"""
    Acoustic radiation force computation.

Computes the volumetric radiation force density F = 2αI/c from
acoustic intensity and tissue absorption. All operations are
element-wise and Zygote-compatible.

References:
    - Nyborg (1965). Acoustic streaming. Physical Acoustics 2B.
    - Sarvazyan et al. (1998). Shear wave elasticity imaging. UMB 24(9):1419-1435.
"""

"""
    compute_radiation_force(intensity, sound_speed, attenuation_np_m)

Compute acoustic radiation force density from intensity field.

    F(r) = 2 * alpha(r) * I(r) / c(r)

# Arguments
- `intensity`: acoustic intensity (W/m^2)
- `sound_speed`: speed of sound (m/s)
- `attenuation_np_m`: absorption coefficient (Nepers/m)

# Returns
Radiation force density (N/m^3), same shape as inputs.
"""
function compute_radiation_force(
    intensity::AbstractArray{<:Real},
    sound_speed::AbstractArray{<:Real},
    attenuation_np_m::AbstractArray{<:Real},
)
    return @. 2.0 * attenuation_np_m * intensity / sound_speed
end

# Scalar overload for convenience
function compute_radiation_force(intensity::Real, sound_speed::Real, attenuation_np_m::Real)
    return 2.0 * attenuation_np_m * intensity / sound_speed
end

"""
    compute_radiation_force_from_db(intensity, sound_speed, attenuation_db_cm)

Convenience: computes radiation force with attenuation in dB/cm/MHz (openlifu convention).
Converts to Np/m internally, then applies F = 2*alpha*I/c.
"""
function compute_radiation_force_from_db(
    intensity::AbstractArray{<:Real},
    sound_speed::AbstractArray{<:Real},
    attenuation_db_cm::AbstractArray{<:Real},
)
    alpha_np_m = @. db_cm_to_neper_m(attenuation_db_cm)
    return compute_radiation_force(intensity, sound_speed, alpha_np_m)
end
