"""VP-SDE noise schedule."""

struct VPSchedule
    beta_min::Float64
    beta_max::Float64
end

VPSchedule() = VPSchedule(0.01, 5.0)  # gentle default (best from autoresearch)

function alpha_bar(s::VPSchedule, t::Real)
    log_ab = -0.5 * (s.beta_min * t + 0.5 * (s.beta_max - s.beta_min) * t^2)
    return exp(log_ab)
end

function noise_and_signal(s::VPSchedule, t::Real)
    ab = alpha_bar(s, t)
    return sqrt(ab), sqrt(1.0 - ab)
end

beta(s::VPSchedule, t::Real) = s.beta_min + t * (s.beta_max - s.beta_min)
