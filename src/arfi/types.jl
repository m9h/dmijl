"""
    MR-ARFI type definitions.

Types for acoustic radiation force impulse (ARFI) imaging simulation,
connecting focused ultrasound treatment planning to MRI signal prediction.

References:
    - Kaye, Chen, Pauly (2011). MRM 65(3):738-743.
    - Linka, St Pierre, Kuhl (2023). Acta Biomaterialia 160:134-151.
"""

# ------------------------------------------------------------------ #
# Tissue property structs
# ------------------------------------------------------------------ #

"""
    AcousticProperties

Acoustic properties for a single tissue type.

# Fields
- `sound_speed::Float64`: m/s
- `density::Float64`: kg/m^3
- `attenuation::Float64`: dB/cm/MHz
- `specific_heat::Float64`: J/kg/K
- `thermal_conductivity::Float64`: W/m/K
"""
struct AcousticProperties
    sound_speed::Float64
    density::Float64
    attenuation::Float64
    specific_heat::Float64
    thermal_conductivity::Float64
end

"""
    TissueMRProperties

MR tissue properties for phantom construction.

# Fields
- `T1::Float64`: spin-lattice relaxation time (seconds)
- `T2::Float64`: spin-spin relaxation time (seconds)
- `PD::Float64`: proton density (0-1, relative)
"""
struct TissueMRProperties
    T1::Float64
    T2::Float64
    PD::Float64
end

# ------------------------------------------------------------------ #
# Sequence and solution structs
# ------------------------------------------------------------------ #

"""
    ARFISequenceParams

Parameters for an MR-ARFI pulse sequence (Butts Pauly 2011).

Sequence structure: RF exc -> MSG1(+) -> [FUS window] -> MSG2(-) -> GRE readout

# Fields
- `msg_amplitude::Float64`: motion-sensitizing gradient amplitude (T/m)
- `msg_duration::Float64`: duration of each MSG lobe (seconds)
- `fus_duration::Float64`: FUS pulse window between MSG lobes (seconds)
- `fov::Float64`: field of view (metres)
- `matrix_size::Int`: spatial resolution (N x N)
- `slice_thickness::Float64`: slice thickness (metres)
- `te::Float64`: echo time (seconds)
- `tr::Float64`: repetition time (seconds)
- `flip_angle::Float64`: flip angle (radians)
- `encoding_direction::Vector{Float64}`: unit vector for MSG encoding axis
"""
struct ARFISequenceParams
    msg_amplitude::Float64
    msg_duration::Float64
    fus_duration::Float64
    fov::Float64
    matrix_size::Int
    slice_thickness::Float64
    te::Float64
    tr::Float64
    flip_angle::Float64
    encoding_direction::Vector{Float64}
end

function ARFISequenceParams(;
    msg_amplitude::Float64 = 40e-3,
    msg_duration::Float64 = 5e-3,
    fus_duration::Float64 = 10e-3,
    fov::Float64 = 0.256,
    matrix_size::Int = 128,
    slice_thickness::Float64 = 5e-3,
    te::Float64 = 25e-3,
    tr::Float64 = 500e-3,
    flip_angle::Float64 = deg2rad(30.0),
    encoding_direction::Vector{Float64} = [0.0, 0.0, 1.0],
)
    return ARFISequenceParams(
        msg_amplitude, msg_duration, fus_duration,
        fov, matrix_size, slice_thickness,
        te, tr, flip_angle, encoding_direction,
    )
end

"""
    TUSSolution

Data from an openlifu treatment planning solution.

Contains the acoustic simulation output (pressure/intensity fields) and
beamforming parameters (delays, apodizations) needed to run the MR-ARFI chain.

All spatial quantities are in SI units (metres, Pa, W/m^2).
"""
struct TUSSolution
    delays::Matrix{Float64}
    apodizations::Matrix{Float64}
    frequency::Float64
    voltage::Float64
    pulse_duration::Float64
    foci::Vector{Vector{Float64}}
    target::Vector{Float64}
    p_max::Array{Float64}
    intensity::Array{Float64}
    grid_spacing::Float64
end

"""
    ARFIResult

Complete result from an MR-ARFI simulation.

# Fields
- `displacement::Array{Float64}`: tissue displacement (metres)
- `phase_map::Array{Float64}`: predicted MR-ARFI phase (radians)
- `radiation_force::Array{Float64}`: radiation force density (N/m^3)
- `shear_modulus::Array{Float64}`: tissue shear modulus (Pa)
- `seq_params::ARFISequenceParams`: sequence parameters used
- `koma_signal::Any`: raw KomaMRI output (nothing if analytical only)
- `koma_phase::Union{Nothing, Array{Float64}}`: phase from Bloch sim
- `recovered_displacement::Union{Nothing, Array{Float64}}`: u from phase
"""
struct ARFIResult
    displacement::Array{Float64}
    phase_map::Array{Float64}
    radiation_force::Array{Float64}
    shear_modulus::Array{Float64}
    seq_params::ARFISequenceParams
    koma_signal::Any
    koma_phase::Union{Nothing, Array{Float64}}
    recovered_displacement::Union{Nothing, Array{Float64}}
end
