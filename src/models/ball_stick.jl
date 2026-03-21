"""
Ball + 2-Stick forward model for dMRI.

Parameters: [d_ball, d_stick, f1, f2, mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]
"""

struct BallStickModel
    bvalues::Vector{Float64}
    gradient_directions::Matrix{Float64}  # (n_meas, 3)
end

function simulate(model::BallStickModel, params::AbstractVector)
    d_ball  = params[1]
    d_stick = params[2]
    f1      = params[3]
    f2      = params[4]
    f_ball  = clamp(1.0 - f1 - f2, 0.0, 1.0)

    mu1 = params[5:7]
    mu1 = mu1 ./ max(norm(mu1), 1e-8)
    mu2 = params[8:10]
    mu2 = mu2 ./ max(norm(mu2), 1e-8)

    b = model.bvalues
    g = model.gradient_directions

    cos1 = g * mu1
    cos2 = g * mu2

    s1 = @. exp(-b * d_stick * cos1^2)
    s2 = @. exp(-b * d_stick * cos2^2)
    s_ball = @. exp(-b * d_ball)

    return @. f1 * s1 + f2 * s2 + f_ball * s_ball
end

function simulate_batch(model::BallStickModel, theta::AbstractMatrix)
    # theta: (n_batch, 10)
    n = size(theta, 1)
    signals = similar(theta, n, length(model.bvalues))
    for i in 1:n
        signals[i, :] = simulate(model, @view theta[i, :])
    end
    return signals
end
