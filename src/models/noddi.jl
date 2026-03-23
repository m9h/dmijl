"""
NODDI (Neurite Orientation Dispersion and Density Imaging) forward model.

Parameters: [f_intra, f_iso, kappa, d_par, mu_x, mu_y, mu_z]
- f_intra: intra-cellular volume fraction (NDI)
- f_iso: isotropic (CSF) volume fraction
- kappa: Watson concentration parameter (→ ODI)
- d_par: parallel diffusivity
- mu: fiber orientation (unit vector)
"""

using SpecialFunctions: besseli

struct NODDIModel
    bvalues::Vector{Float64}
    gradient_directions::Matrix{Float64}  # (n_meas, 3)
    d_iso::Float64  # isotropic diffusivity (CSF), typically 3.0e-9
    d_perp_tortuosity::Bool  # if true, d_perp = d_par * (1 - f_intra)
end

NODDIModel(bvals, bvecs) = NODDIModel(bvals, bvecs, 3.0e-9, true)

"""Watson distribution normalization: c(κ) = ₁F₁(1/2; 3/2; κ)"""
function watson_norm(kappa::Real)
    # Dawson function approximation for numerical stability
    if abs(kappa) < 1e-6
        return 1.0
    elseif kappa > 0
        return sqrt(π) * erfi(sqrt(kappa)) / (2 * sqrt(kappa))
    else
        return sqrt(π) * erf(sqrt(-kappa)) / (2 * sqrt(-kappa))
    end
end

"""ODI from Watson kappa: ODI = 2/π * arctan(1/κ)"""
kappa_to_odi(kappa) = 2 / π * atan(1 / max(kappa, 1e-8))

function simulate(model::NODDIModel, params::AbstractVector)
    f_intra = params[1]
    f_iso   = params[2]
    kappa   = params[3]
    d_par   = params[4]
    mu      = params[5:7]
    mu      = mu ./ max(norm(mu), 1e-8)

    f_extra = clamp(1.0 - f_intra - f_iso, 0.0, 1.0)

    # Tortuosity constraint
    d_perp = model.d_perp_tortuosity ? d_par * (1 - f_intra) : d_par * 0.3

    b = model.bvalues
    g = model.gradient_directions

    # Isotropic compartment
    s_iso = @. exp(-b * model.d_iso)

    # Watson-distributed sticks (intra-cellular)
    # Approximate: integrate Watson ODF × stick kernel
    cos_theta = g * mu
    # For Watson with high κ: signal ≈ exp(-b * d_par * cos²θ)
    # Full integral requires numerical quadrature; use SH approximation
    s_intra = @. exp(-b * d_par * cos_theta^2)

    # Extra-cellular (Zeppelin): axially symmetric tensor
    s_extra = @. exp(-b * (d_perp + (d_par - d_perp) * cos_theta^2))

    return @. f_intra * s_intra + f_extra * s_extra + f_iso * s_iso
end
