"""
    Differentiable FEM signal computation.

Enables gradient-based fitting of tissue parameters (R, D, f) by computing
∂signal/∂p via finite differences through the full FEM pipeline.

For ∂signal/∂D: the stiffness matrix S depends linearly on D, so rebuilding
matrices is fast. The eigendecomposition is the bottleneck but only takes
~100ms for typical meshes.

For ∂signal/∂R: requires rebuilding the mesh (geometry changes), which is
slower (~1-5s). Cached geometries at discrete R values can accelerate this.
"""

"""
    FEMSignalCache

Caches FEM geometries at discrete parameter values for fast interpolation.
"""
struct FEMSignalCache
    geometries::Dict{Float64, FEMGeometry}  # R → prebuilt geometry
    neig::Int
end

"""
    build_fem_cache(R_values, D; neig=100) -> FEMSignalCache

Precompute FEM geometries for a grid of cylinder radii.
"""
function build_fem_cache(R_values::Vector{Float64}, D::Float64; neig::Int=100)
    geometries = Dict{Float64, FEMGeometry}()
    for R in R_values
        geometries[R] = build_fem_cylinder(R, D; neig=neig)
    end
    return FEMSignalCache(geometries, neig)
end

"""
    fem_signal_gradient(R, D, delta, Delta, bvalues, directions;
                        neig=100, eps_R=1e-8, eps_D=1e-11) -> (S, dS_dR, dS_dD)

Compute FEM signal and its gradients w.r.t. radius R and diffusivity D
via central finite differences.

Returns:
- `S`: signal vector at (R, D)
- `dS_dR`: ∂S/∂R (signal sensitivity to radius)
- `dS_dD`: ∂S/∂D (signal sensitivity to diffusivity)
"""
function fem_signal_gradient(R::Float64, D::Float64,
                             delta::Float64, Delta::Float64,
                             bvalues::Vector{Float64},
                             directions::Matrix{Float64};
                             neig::Int=50,
                             eps_R::Float64=1e-8,
                             eps_D::Float64=1e-11)
    S = fem_cylinder_signal(R, D, delta, Delta, bvalues, directions; neig=neig)

    # ∂S/∂D via central finite differences (cheap — same mesh, rebuild matrices)
    S_Dp = fem_cylinder_signal(R, D + eps_D, delta, Delta, bvalues, directions; neig=neig)
    S_Dm = fem_cylinder_signal(R, D - eps_D, delta, Delta, bvalues, directions; neig=neig)
    dS_dD = (S_Dp .- S_Dm) ./ (2 * eps_D)

    # ∂S/∂R via central finite differences (expensive — new mesh per R)
    S_Rp = fem_cylinder_signal(R + eps_R, D, delta, Delta, bvalues, directions; neig=neig)
    S_Rm = fem_cylinder_signal(R - eps_R, D, delta, Delta, bvalues, directions; neig=neig)
    dS_dR = (S_Rp .- S_Rm) ./ (2 * eps_R)

    return S, dS_dR, dS_dD
end

"""
    fem_axcaliber_signal(R, D_intra, D_extra, f_intra, mu,
                         acq::Acquisition; neig=50) -> Vector{Float64}

Multi-compartment AxCaliber signal using FEM for the intra-cellular
cylinder and analytical free diffusion for the extra-cellular space.

S = f * S_fem_cylinder(R, D_intra) + (1-f) * exp(-b * D_extra)

This is the direct FEM replacement for the Van Gelderen PINN signal.
"""
function fem_axcaliber_signal(R::Float64, D_intra::Float64, D_extra::Float64,
                              f_intra::Float64, mu::Vector{Float64},
                              acq::Acquisition; neig::Int=50)
    delta = acq.delta === nothing ? 10e-3 : acq.delta
    Delta = acq.Delta === nothing ? 30e-3 : acq.Delta

    n_hat = mu ./ max(norm(mu), 1e-12)
    n_meas = length(acq.bvalues)

    # Build FEM geometry for the intra-cellular cylinder
    geom = build_fem_cylinder(R, D_intra; neig=neig)

    signals = zeros(Float64, n_meas)
    for m in 1:n_meas
        b = acq.bvalues[m]
        g_dir = acq.gradient_directions[m, :]

        # Extra-cellular: free Gaussian diffusion
        S_extra = exp(-b * D_extra)

        if b < 1e3
            signals[m] = 1.0
        else
            # Intra-cellular: FEM solves the Bloch-Torrey PDE in the cylinder
            # The gradient direction relative to the cylinder axis matters
            S_intra = fem_signal(geom, delta, Delta, b, g_dir)
            signals[m] = f_intra * S_intra + (1 - f_intra) * S_extra
        end
    end

    return signals
end
