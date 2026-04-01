"""
    FEM Bloch-Torrey forward model using SpinDoctor.jl.

Computes dMRI signals for restricted diffusion in cylinders (and spheres)
using the finite element method — exact solutions without the GPD
approximation used by Van Gelderen / Soderman / Murday-Cotts.

Key advantage over analytical models: handles finite pulse duration δ
exactly, which matters for WAND AxCaliber (δ = 11 ms).

Unit conventions:
- DMI.jl: metres, seconds, m²/s, s/m² (SI)
- SpinDoctor: µm, µs, µm²/µs (internal)
- Conversion at the boundary of this module.
"""

using SpinDoctor
using LinearAlgebra

# Unit conversion factors: SI → SpinDoctor
const _M_TO_UM = 1e6       # metres → µm
const _S_TO_US = 1e6       # seconds → µs
const _D_SI_TO_SD = 1e6    # m²/s → µm²/µs  (1e-12/1e-6 = 1e-6... wait)
# Actually: D [m²/s] = D [µm²/µs] * (1e-6)²/(1e-6) = D [µm²/µs] * 1e-6
# So D_sd = D_si / 1e-6 = D_si * 1e6
# SpinDoctor γ = 2.67513e-4 µm⁻¹ µs⁻¹ T⁻¹
const _GAMMA_SD = 2.67513e-4

"""
    FEMGeometry

Cached SpinDoctor geometry: mesh + FEM matrices + eigendecomposition.
Reusable across different gradient conditions for the same geometry.
"""
struct FEMGeometry
    mesh::Any           # SpinDoctor FEMesh
    model::Any          # SpinDoctor Model
    matrices::Any       # SpinDoctor assembled matrices (M, S, R, Mx, Q)
    lap_eig::Any        # Laplace eigendecomposition (values, funcs, moments, massrelax)
    R_um::Float64       # cylinder radius in µm
    D_um::Float64       # diffusivity in µm²/µs
end

"""
    build_fem_cylinder(R, D; neig=100, height_factor=5) -> FEMGeometry

Build a SpinDoctor cylinder geometry and precompute FEM matrices
and Laplace eigendecomposition.

# Arguments
- `R::Float64`: cylinder radius in metres
- `D::Float64`: intrinsic diffusivity in m²/s
- `neig::Int`: number of Laplace eigenmodes (default 100)
- `height_factor::Float64`: cylinder height as multiple of radius
"""
function build_fem_cylinder(R::Float64, D::Float64;
                            neig::Int=100, height_factor::Float64=5.0)
    R_um = R * _M_TO_UM
    D_um = D * _M_TO_UM^2 / _S_TO_US  # m²/s → µm²/µs

    setup = SpinDoctor.CylinderSetup(;
        ncell=1,
        rmin=R_um, rmax=R_um,
        dmin=R_um * 0.1, dmax=R_um * 0.1,
        height=R_um * height_factor,
    )

    coeffs = SpinDoctor.coefficients(setup;
        D = (; cell = [D_um * I(3)]),
        T₂ = (; cell = [Inf]),
        ρ = (; cell = [1.0]),
        κ = (; cell_interfaces = zeros(0), cell_boundaries = [0.0]),
        γ = _GAMMA_SD,
    )

    mesh, = SpinDoctor.create_geometry(setup)
    model = SpinDoctor.Model(; mesh, coeffs...)
    matrices = SpinDoctor.assemble_matrices(model)

    lap = SpinDoctor.Laplace(; model, matrices, neig_max=neig)
    lap_eig = SpinDoctor.solve(lap)

    return FEMGeometry(mesh, model, matrices, lap_eig, R_um, D_um)
end

"""
    fem_signal(geom::FEMGeometry, delta, Delta, b, direction; ninterval=500) -> Float64

Compute the dMRI signal attenuation S/S₀ for a single (b, direction)
measurement using the matrix formalism.

# Arguments (SI units)
- `delta::Float64`: gradient pulse duration in seconds
- `Delta::Float64`: diffusion time in seconds
- `b::Float64`: b-value in s/m²
- `direction::Vector{Float64}`: unit gradient direction (3D)
"""
function fem_signal(geom::FEMGeometry, delta::Float64, Delta::Float64,
                    b::Float64, direction::AbstractVector;
                    ninterval::Int=500)
    b < 1e3 && return 1.0  # b ≈ 0

    # Convert timing to µs
    delta_us = delta * _S_TO_US
    Delta_us = Delta * _S_TO_US

    # Build PGSE profile
    profile = SpinDoctor.PGSE(delta_us, Delta_us)

    # Compute gradient amplitude from b-value
    # b = γ² g² ∫F² in SpinDoctor units
    g = sqrt(b / SpinDoctor.int_F²(profile)) / _GAMMA_SD

    dir = Float64.(direction)
    dir = dir ./ max(norm(dir), 1e-12)
    gradient = SpinDoctor.ScalarGradient(dir, profile, g)

    # Solve using matrix formalism (eigenspace projection)
    mf = SpinDoctor.MatrixFormalism(; geom.model, geom.matrices, geom.lap_eig)
    ξ = SpinDoctor.solve(mf, gradient; ninterval=ninterval)

    # Signal = integral of magnetization
    S = abs(SpinDoctor.compute_signal(geom.matrices.M, ξ))

    # Normalize by b=0 signal
    ρ = SpinDoctor.initial_conditions(geom.model)
    S0 = abs(sum(geom.matrices.M * ρ))

    return S / max(S0, 1e-30)
end

"""
    fem_cylinder_signal(R, D, delta, Delta, bvalues, directions; neig=100) -> Vector{Float64}

End-to-end: build cylinder geometry, compute signals for all measurements.

# Arguments (SI units)
- `R::Float64`: cylinder radius in metres
- `D::Float64`: diffusivity in m²/s
- `delta::Float64`: gradient pulse duration in seconds
- `Delta::Float64`: diffusion time in seconds
- `bvalues::Vector{Float64}`: b-values in s/m²
- `directions::Matrix{Float64}`: gradient directions (n_meas × 3)
"""
function fem_cylinder_signal(R::Float64, D::Float64,
                             delta::Float64, Delta::Float64,
                             bvalues::Vector{Float64},
                             directions::Matrix{Float64};
                             neig::Int=100)
    geom = build_fem_cylinder(R, D; neig=neig)
    n_meas = length(bvalues)
    signals = ones(Float64, n_meas)

    for m in 1:n_meas
        signals[m] = fem_signal(geom, delta, Delta, bvalues[m], directions[m, :])
    end

    return signals
end
