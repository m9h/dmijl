"""
    EPG — Extended Phase Graph for multi-echo spin-echo sequences.

Models T2-relaxation and stimulated echo pathways in CPMG-like sequences.
Handles imperfect refocusing pulses (B1 inhomogeneity), critical for
quantitative T2 mapping and myelin water imaging.

Implementation follows Weigel (2015) JMRI extended phase graph review.
"""

using BlochSimulators: EPGSimulator, simulate_magnetization,
    ConfigurationStates, T₁T₂B₁

"""
    EPGCompartment(; T1, T2, B1=1.0, TE, ETL, refocus_angle=π)

Extended Phase Graph compartment for multi-echo spin-echo signal.
"""
Base.@kwdef struct EPGCompartment
    T1::Float64
    T2::Float64
    B1::Float64 = 1.0
    TE::Float64
    ETL::Int
    refocus_angle::Float64 = π
end

"""
    epg_signal(T1, T2, B1, TE, ETL; refocus_angle=π) -> Vector{Float64}

Compute CPMG multi-echo signal using the Extended Phase Graph formalism.

Sequence: 90°_x — [TE/2 — α_y — TE/2 — echo]_ETL

Returns a vector of length ETL with signal magnitude at each echo.
"""
function epg_signal(T1::Real, T2::Real, B1::Real, TE::Real, ETL::Int;
                    refocus_angle::Real=π)
    nstates = ETL + 1

    # EPG states: index 1 = order 0, index n = order n-1
    Fp = zeros(ComplexF64, nstates)
    Fm = zeros(ComplexF64, nstates)
    Z = zeros(ComplexF64, nstates)
    Z[1] = 1.0 + 0im

    # 90° excitation
    _epg_rf!(Fp, Fm, Z, π / 2)

    E1 = exp(-TE / 2 / T1)
    E2 = exp(-TE / 2 / T2)
    echoes = zeros(Float64, ETL)

    for echo in 1:ETL
        # TE/2 before refocusing: relax then dephase
        _epg_relax!(Fp, Fm, Z, E1, E2)
        _epg_shift!(Fp, Fm)

        # Refocusing pulse
        _epg_rf!(Fp, Fm, Z, B1 * refocus_angle)

        # TE/2 after refocusing: dephase then relax
        _epg_shift!(Fp, Fm)
        _epg_relax!(Fp, Fm, Z, E1, E2)

        # Echo = F+[order 0]
        echoes[echo] = abs(Fp[1])
    end

    return echoes
end

# ---- EPG primitives (Weigel 2015) ----

function _epg_rf!(Fp, Fm, Z, alpha)
    c2 = cos(alpha / 2)^2
    s2 = sin(alpha / 2)^2
    sc = sin(alpha) / 2
    ca = cos(alpha)

    @inbounds for n in eachindex(Fp)
        fp = Fp[n]; fm = Fm[n]; z = Z[n]
        Fp[n] = c2 * fp + s2 * conj(fm) + im * sc * z
        Fm[n] = s2 * conj(fp) + c2 * fm - im * sc * z
        Z[n]  = -im * sc * (fp - conj(fm)) + ca * z
    end
end

function _epg_relax!(Fp, Fm, Z, E1, E2)
    @inbounds for n in eachindex(Fp)
        Fp[n] *= E2
        Fm[n] *= E2
        Z[n] *= E1
    end
    Z[1] += (1 - E1)
end

function _epg_shift!(Fp, Fm)
    # F+ shifts to higher orders
    @inbounds for n in length(Fp):-1:2
        Fp[n] = Fp[n-1]
    end
    Fp[1] = conj(Fm[1])
    # F- shifts to lower orders
    @inbounds for n in 1:(length(Fm)-1)
        Fm[n] = Fm[n+1]
    end
    Fm[end] = 0.0 + 0im
end
