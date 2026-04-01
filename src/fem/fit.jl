"""
    Gradient-based fitting of AxCaliber parameters using the FEM forward model.
"""

using Optim

"""
    fit_fem_axcaliber(signal_obs, acq; R0=3e-6, D_intra0=1.7e-9,
                      D_extra0=1.0e-9, f0=0.5, mu0=[0,0,1], neig=50) -> NamedTuple

Fit AxCaliber parameters to observed signal using the FEM Bloch-Torrey
forward model and Optim.jl L-BFGS-B.

# Arguments
- `signal_obs`: observed signal vector (normalized by S₀)
- `acq`: Acquisition with bvalues, gradient_directions, delta, Delta

# Returns
Named tuple with fitted parameters and residuals.
"""
function fit_fem_axcaliber(signal_obs::Vector{Float64}, acq::Acquisition;
                           R0::Float64=3e-6,
                           D_intra0::Float64=1.7e-9,
                           D_extra0::Float64=1.0e-9,
                           f0::Float64=0.5,
                           mu0::Vector{Float64}=[0.0, 0.0, 1.0],
                           neig::Int=50,
                           max_iter::Int=50,
                           verbose::Bool=true)

    # Pack parameters: [R, D_intra, D_extra, f_intra]
    # (mu fixed for now — orientation from DTI init)
    x0 = [R0, D_intra0, D_extra0, f0]
    lower = [0.5e-6, 0.1e-9, 0.1e-9, 0.05]
    upper = [10e-6,  3.0e-9, 3.0e-9, 0.95]

    mu = mu0 ./ max(norm(mu0), 1e-12)

    function objective(x)
        R, D_intra, D_extra, f_intra = x
        S_pred = fem_axcaliber_signal(R, D_intra, D_extra, f_intra, mu, acq; neig=neig)
        residual = sum((S_pred .- signal_obs).^2) / length(signal_obs)
        verbose && println("  R=$(round(R*1e6, digits=2))µm, D_i=$(round(D_intra*1e9, digits=2)), " *
                          "D_e=$(round(D_extra*1e9, digits=2)), f=$(round(f_intra, digits=3)), " *
                          "MSE=$(round(residual, sigdigits=4))")
        return residual
    end

    result = Optim.optimize(objective, lower, upper, x0,
                            Optim.Fminbox(Optim.NelderMead()),
                            Optim.Options(iterations=max_iter))

    x_opt = Optim.minimizer(result)
    R_fit, D_intra_fit, D_extra_fit, f_fit = x_opt

    S_fit = fem_axcaliber_signal(R_fit, D_intra_fit, D_extra_fit, f_fit, mu, acq; neig=neig)
    residual = sum((S_fit .- signal_obs).^2) / length(signal_obs)

    return (
        R = R_fit,
        D_intra = D_intra_fit,
        D_extra = D_extra_fit,
        f_intra = f_fit,
        mu = mu,
        signal_fit = S_fit,
        residual = residual,
        converged = Optim.converged(result),
    )
end
