"""
Diffusion Tensor Imaging (DTI) forward model.

Parameters: [λ₁, λ₂, λ₃, θ, φ, ψ] — eigenvalues + Euler angles
or equivalently [D11, D12, D13, D22, D23, D33] — unique tensor elements.

Derived metrics: FA, MD, AD, RD.
"""

struct DTIModel
    bvalues::Vector{Float64}
    gradient_directions::Matrix{Float64}  # (n_meas, 3)
end

"""Simulate signal from eigenvalue parameterisation."""
function simulate(model::DTIModel, params::AbstractVector)
    λ1, λ2, λ3 = params[1], params[2], params[3]

    # Euler angles → rotation matrix
    θ, φ, ψ = params[4], params[5], params[6]

    cθ, sθ = cos(θ), sin(θ)
    cφ, sφ = cos(φ), sin(φ)
    cψ, sψ = cos(ψ), sin(ψ)

    # ZYZ convention rotation matrix
    R = [cφ*cθ*cψ - sφ*sψ  -cφ*cθ*sψ - sφ*cψ  cφ*sθ;
         sφ*cθ*cψ + cφ*sψ  -sφ*cθ*sψ + cφ*cψ  sφ*sθ;
         -sθ*cψ             sθ*sψ               cθ]

    # Diffusion tensor D = R * diag(λ) * R'
    D = R * diagm([λ1, λ2, λ3]) * R'

    b = model.bvalues
    g = model.gradient_directions

    signal = similar(b)
    for i in eachindex(b)
        gi = @view g[i, :]
        signal[i] = exp(-b[i] * dot(gi, D * gi))
    end
    return signal
end

# ---- Derived metrics ----

function compute_fa(λ1, λ2, λ3)
    md = (λ1 + λ2 + λ3) / 3
    num = sqrt((λ1 - md)^2 + (λ2 - md)^2 + (λ3 - md)^2)
    den = sqrt(λ1^2 + λ2^2 + λ3^2)
    return den > 0 ? sqrt(3 / 2) * num / den : 0.0
end

compute_md(λ1, λ2, λ3) = (λ1 + λ2 + λ3) / 3
compute_ad(λ1, λ2, λ3) = λ1
compute_rd(λ1, λ2, λ3) = (λ2 + λ3) / 2
