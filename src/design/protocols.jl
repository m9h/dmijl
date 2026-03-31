"""
    Standard protocol recipes for common dMRI acquisition schemes.
"""

"""
    hcp_protocol(; n_dirs=90, seed=42) -> Acquisition

HCP-like multi-shell protocol: b = [0, 1000, 2000, 3000] s/mm^2.
6 b=0 + 30 directions per shell.
"""
function hcp_protocol(; n_dirs::Int=90, seed::Int=42)
    n_per_shell = n_dirs ÷ 3
    n_b0 = 6
    dirs = electrostatic_directions(n_dirs; seed=seed)

    bvals = vcat(
        zeros(n_b0),
        fill(1e9, n_per_shell),     # 1000 s/mm² = 1e9 s/m²
        fill(2e9, n_per_shell),
        fill(3e9, n_dirs - 2 * n_per_shell)
    )
    all_dirs = vcat(repeat([1.0 0.0 0.0], n_b0), dirs)
    return Acquisition(bvals, all_dirs, 10.6e-3, 43.1e-3)
end

"""
    noddi_protocol(; n_dirs=60, seed=42) -> Acquisition

NODDI-optimized 2-shell: b = [0, 700, 2000] s/mm^2.
Ref: Zhang et al., NeuroImage 2012.
"""
function noddi_protocol(; n_dirs::Int=60, seed::Int=42)
    n_per_shell = n_dirs ÷ 2
    n_b0 = 6
    dirs = electrostatic_directions(n_dirs; seed=seed)

    bvals = vcat(
        zeros(n_b0),
        fill(0.7e9, n_per_shell),
        fill(2e9, n_dirs - n_per_shell)
    )
    all_dirs = vcat(repeat([1.0 0.0 0.0], n_b0), dirs)
    return Acquisition(bvals, all_dirs, 11e-3, 30e-3)
end

"""
    axon_diameter_protocol(; G_max=0.3, n_dirs=30, seed=42) -> Acquisition

Protocol optimized for axon diameter: high b-values, short diffusion times.
Designed for Connectom-class scanners (G_max >= 300 mT/m).
Ref: Alexander 2008.
"""
function axon_diameter_protocol(; G_max::Float64=0.3, n_dirs::Int=30, seed::Int=42)
    n_b0 = 6
    delta = 8e-3
    Delta = 20e-3
    b_max = max_bvalue(G_max, delta, Delta)
    dirs = electrostatic_directions(n_dirs; seed=seed)

    # 4 shells with logarithmically spaced b-values
    n_per_shell = n_dirs ÷ 4
    remainder = n_dirs - 3 * n_per_shell
    b_shells = [b_max * f for f in [0.1, 0.3, 0.6, 1.0]]

    bvals = vcat(
        zeros(n_b0),
        fill(b_shells[1], n_per_shell),
        fill(b_shells[2], n_per_shell),
        fill(b_shells[3], n_per_shell),
        fill(b_shells[4], remainder)
    )
    all_dirs = vcat(repeat([1.0 0.0 0.0], n_b0), dirs)
    return Acquisition(bvals, all_dirs, delta, Delta)
end
