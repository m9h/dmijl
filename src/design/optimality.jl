"""
    Optimality criteria for experimental design.
"""

"""D-optimality: log det(F). Minimizes confidence ellipsoid volume."""
d_optimality(F::AbstractMatrix) = logdet(Symmetric(F + 1e-12 * I))

"""A-optimality: -trace(F⁻¹). Minimizes mean parameter variance."""
a_optimality(F::AbstractMatrix) = -tr(inv(Symmetric(F + 1e-12 * I)))

"""E-optimality: λ_min(F). Maximizes worst-case parameter precision."""
e_optimality(F::AbstractMatrix) = minimum(eigvals(Symmetric(F + 1e-12 * I)))

"""Weighted A-optimality: -w'diag(F⁻¹). Emphasizes specific parameters."""
function weighted_a_optimality(F::AbstractMatrix, weights::AbstractVector)
    return -dot(weights, diag(inv(Symmetric(F + 1e-12 * I))))
end

"""
    optimality_criterion(F; criterion=:D, weights=nothing) -> Float64

Unified interface to optimality criteria.
"""
function optimality_criterion(F::AbstractMatrix; criterion::Symbol=:D,
                               weights::Union{Nothing, AbstractVector}=nothing)
    criterion == :D && return d_optimality(F)
    criterion == :A && return a_optimality(F)
    criterion == :E && return e_optimality(F)
    criterion == :weighted_A && return weighted_a_optimality(F, weights)
    error("Unknown criterion: $criterion")
end
