"""
    Hardware constraints and gradient direction generation for dMRI protocol design.
"""

const GYROMAGNETIC_RATIO = 2.6752218744e8  # rad/s/T (proton)

"""
    max_bvalue(G_max, delta, Delta) -> Float64

Maximum achievable b-value (s/m²) for PGSE given hardware gradient limit.
"""
function max_bvalue(G_max::Float64, delta::Float64, Delta::Float64)
    GYROMAGNETIC_RATIO^2 * G_max^2 * delta^2 * (Delta - delta / 3.0)
end

"""
    required_gradient(b, delta, Delta) -> Float64

Gradient strength (T/m) required to achieve target b-value.
"""
function required_gradient(b::Float64, delta::Float64, Delta::Float64)
    sqrt(b / (GYROMAGNETIC_RATIO^2 * delta^2 * (Delta - delta / 3.0)))
end

"""
    is_feasible(acq::Acquisition, G_max::Float64) -> Bool

Check if all measurements are achievable with the given gradient hardware.
"""
function is_feasible(acq::Acquisition, G_max::Float64)
    (acq.delta === nothing || acq.Delta === nothing) && return true
    b_max = max_bvalue(G_max, acq.delta, acq.Delta)
    return all(b .<= b_max * 1.01 for b in acq.bvalues)
end

"""
    electrostatic_directions(n; max_iter=5000, seed=42) -> Matrix{Float64}

Generate n approximately uniformly distributed directions on the unit hemisphere
using electrostatic repulsion. Returns (n × 3) matrix of unit vectors.
"""
function electrostatic_directions(n::Int; max_iter::Int=5000, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    points = randn(rng, n, 3)
    points ./= sqrt.(sum(points.^2, dims=2))
    # Force to upper hemisphere (antipodal symmetry in dMRI)
    for i in 1:n
        points[i, 3] < 0 && (points[i, :] .*= -1)
    end

    lr = 0.01
    for iter in 1:max_iter
        forces = zeros(n, 3)
        for i in 1:n, j in (i+1):n
            diff = points[i, :] - points[j, :]
            dist = max(norm(diff), 1e-10)
            f = diff ./ dist^3
            forces[i, :] .+= f
            forces[j, :] .-= f
            # Also repel from antipodal points (dMRI symmetry)
            diff_anti = points[i, :] + points[j, :]
            dist_anti = max(norm(diff_anti), 1e-10)
            f_anti = diff_anti ./ dist_anti^3
            forces[i, :] .+= f_anti
            forces[j, :] .+= f_anti
        end
        points .+= lr .* forces
        points ./= sqrt.(sum(points.^2, dims=2))
        for i in 1:n
            points[i, 3] < 0 && (points[i, :] .*= -1)
        end
        lr *= 0.999
    end

    return points
end

"""
    compare_protocols(model, acquisitions, params; sigma=0.02, labels=nothing)

Compare multiple acquisition protocols on FIM-based criteria.
Returns a vector of named tuples with metrics for each protocol.
"""
function compare_protocols(model, acquisitions::Vector{<:Acquisition},
                           params::AbstractVector;
                           sigma::Float64=0.02,
                           labels::Union{Nothing, Vector{String}}=nothing)
    results = map(enumerate(acquisitions)) do (i, acq)
        F = fisher_information(model, acq, params; sigma=sigma)
        label = labels !== nothing ? labels[i] : "Protocol $i"
        cr = crlb(model, acq, params; sigma=sigma)
        (label=label,
         n_measurements=length(acq.bvalues),
         D_opt=d_optimality(F),
         A_opt=a_optimality(F),
         E_opt=e_optimality(F),
         crlb=cr,
         condition_number=cond(F + 1e-12 * I))
    end
    return results
end
