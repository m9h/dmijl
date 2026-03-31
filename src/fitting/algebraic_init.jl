"""
    Algebraic initializers for multi-compartment model fitting.

Provides DTI-based starting points for NLLS fitting, reducing sensitivity
to local minima and improving convergence speed.

Strategy:
1. Fit DTI (diffusion tensor) to the data
2. Extract eigenvalues and eigenvectors
3. Map DTI metrics to multi-compartment parameter estimates
"""

"""
    dti_init(acq::Acquisition, signal_vec::AbstractVector) -> Dict{Symbol, Any}

Fit a diffusion tensor to the data and return DTI-derived metrics.

Uses weighted linear least squares on log(S/S₀) = -b * g' D g.

Returns Dict with:
- `:eigenvalues` — (λ₁, λ₂, λ₃) sorted descending
- `:eigenvectors` — (3×3) matrix, columns are eigenvectors
- `:FA` — fractional anisotropy
- `:MD` — mean diffusivity
- `:AD` — axial diffusivity (λ₁)
- `:RD` — radial diffusivity (mean of λ₂, λ₃)
"""
function dti_init(acq::Acquisition, signal_vec::AbstractVector)
    b = acq.bvalues
    g = acq.gradient_directions
    n_meas = length(b)

    # Identify b=0 and DW measurements
    b0_mask = b .< 100e6
    dw_mask = .!b0_mask

    S0 = mean(signal_vec[b0_mask])
    S0 = max(S0, 1e-10)

    # Log-linear model: log(S/S₀) = -B * d
    # where B is the (n_dw × 6) design matrix and d = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    b_dw = b[dw_mask]
    g_dw = g[dw_mask, :]
    y = log.(max.(signal_vec[dw_mask] ./ S0, 1e-10))

    # Design matrix: each row is -b * [gx², gy², gz², 2gxgy, 2gxgz, 2gygz]
    n_dw = sum(dw_mask)
    B = zeros(n_dw, 6)
    for i in 1:n_dw
        gx, gy, gz = g_dw[i, :]
        B[i, :] = -b_dw[i] .* [gx^2, gy^2, gz^2, 2gx*gy, 2gx*gz, 2gy*gz]
    end

    # Weighted least squares (weights = S²)
    w = max.(signal_vec[dw_mask], 1e-10).^2
    W = Diagonal(w)
    d = (B' * W * B) \ (B' * W * y)

    # Reconstruct symmetric tensor
    D_tensor = [d[1] d[4] d[5];
                d[4] d[2] d[6];
                d[5] d[6] d[3]]

    # Force positive semi-definite
    D_tensor = Symmetric(D_tensor)
    eig = eigen(D_tensor)
    λ = max.(real.(eig.values), 0.0)
    V = real.(eig.vectors)

    # Sort descending
    idx = sortperm(λ, rev=true)
    λ = λ[idx]
    V = V[:, idx]

    MD = mean(λ)
    AD = λ[1]
    RD = (λ[2] + λ[3]) / 2
    FA_num = sqrt(0.5 * ((λ[1]-λ[2])^2 + (λ[2]-λ[3])^2 + (λ[3]-λ[1])^2))
    FA_den = sqrt(sum(λ.^2))
    FA = FA_den > 0 ? FA_num / FA_den : 0.0

    return Dict(
        :eigenvalues => λ,
        :eigenvectors => V,
        :FA => FA,
        :MD => MD,
        :AD => AD,
        :RD => RD,
        :S0 => S0,
    )
end

"""
    ball_stick_init(acq, signal_vec) -> Vector{Float64}

Initialize Ball+Stick model parameters from DTI:
- f_stick ≈ FA (rough proxy)
- D_ball ≈ MD
- D_stick ≈ AD
- mu ≈ primary eigenvector

Returns parameter vector compatible with MultiCompartmentModel(G1Ball, C1Stick).
"""
function ball_stick_init(acq::Acquisition, signal_vec::AbstractVector)
    dti = dti_init(acq, signal_vec)

    f_stick = clamp(dti[:FA], 0.1, 0.9)
    f_ball = 1.0 - f_stick
    D_ball = clamp(dti[:MD], 0.5e-9, 3.0e-9)
    D_stick = clamp(dti[:AD], 0.5e-9, 3.0e-9)
    mu = dti[:eigenvectors][:, 1]  # primary eigenvector
    mu = mu ./ max(norm(mu), 1e-12)

    # Parameter order for MultiCompartmentModel(G1Ball, C1Stick):
    # [lambda_iso, mu_x, mu_y, mu_z, lambda_par, f_ball, f_stick]
    return [D_ball, mu[1], mu[2], mu[3], D_stick, f_ball, f_stick]
end

"""
    noddi_init(acq, signal_vec) -> Dict{Symbol, Float64}

Initialize NODDI-like parameters from DTI:
- f_intra ≈ FA
- D_parallel ≈ AD (clamped to 1.7e-9 as in NODDI)
- kappa ≈ function of FA (higher FA → higher concentration)
- mu ≈ primary eigenvector
"""
function noddi_init(acq::Acquisition, signal_vec::AbstractVector)
    dti = dti_init(acq, signal_vec)

    f_intra = clamp(dti[:FA], 0.05, 0.95)
    # Watson concentration from FA (empirical mapping)
    # FA ≈ 0 → kappa ≈ 0 (isotropic), FA ≈ 1 → kappa → ∞
    kappa = max(0.0, 10.0 * dti[:FA]^2 / (1.0 - dti[:FA]^2 + 1e-10))
    kappa = min(kappa, 100.0)

    return Dict(
        :f_intra => f_intra,
        :f_iso => clamp(1.0 - f_intra - 0.1, 0.0, 0.5),
        :D_parallel => 1.7e-9,
        :kappa => kappa,
        :mu => dti[:eigenvectors][:, 1],
    )
end
