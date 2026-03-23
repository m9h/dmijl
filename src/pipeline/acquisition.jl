"""
Acquisition specification for dMRI experiments.
"""

using LinearAlgebra

struct Acquisition
    bvalues::Vector{Float64}
    gradient_directions::Matrix{Float64}  # (n_meas, 3)
    delta::Union{Nothing, Float64}        # gradient pulse duration
    Delta::Union{Nothing, Float64}        # diffusion time
end

Acquisition(bvals, bvecs) = Acquisition(bvals, bvecs, nothing, nothing)

n_measurements(a::Acquisition) = length(a.bvalues)
b0_mask(a::Acquisition) = a.bvalues .< 100e6

"""Create an HCP-like 90-direction multi-shell acquisition."""
function hcp_like_acquisition(; seed::Int=0)
    rng = MersenneTwister(seed)

    function rand_vecs(rng, n)
        z = randn(rng, n, 3)
        return z ./ sqrt.(sum(z.^2, dims=2))
    end

    v0 = repeat([1.0 0.0 0.0], 6, 1)
    v1 = rand_vecs(rng, 30)
    v2 = rand_vecs(rng, 30)
    v3 = rand_vecs(rng, 24)

    bvals = vcat(zeros(6), fill(1e9, 30), fill(2e9, 30), fill(3e9, 24))
    bvecs = vcat(v0, v1, v2, v3)

    return Acquisition(bvals, bvecs)
end

"""Load acquisition from a bval/bvec file pair (FSL format)."""
function load_acquisition(bval_path::String, bvec_path::String)
    bvals = parse.(Float64, split(strip(read(bval_path, String))))
    bvec_lines = readlines(bvec_path)
    bvecs = hcat([parse.(Float64, split(strip(l))) for l in bvec_lines]...)'
    # bvecs should be (n_meas, 3)
    if size(bvecs, 2) != 3
        bvecs = bvecs'
    end
    return Acquisition(bvals, Matrix(bvecs))
end
