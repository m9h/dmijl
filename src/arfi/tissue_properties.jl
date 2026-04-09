"""
    Tissue property lookup tables for MR-ARFI simulation.

Maps integer tissue labels to acoustic, MR, and mechanical properties.

Label convention (matching SCI Institute head model):
    0 = background (water)
    1 = scalp
    2 = skull (cortical bone)
    3 = CSF
    4 = gray matter
    5 = white matter

Acoustic properties: ITRUSST benchmark (Aubry et al. 2022, JASA 152(2):1003-1019).
Shear moduli: Kuhl CANN (Linka, St Pierre, Kuhl 2023, Acta Biomaterialia 160:134-151).
MR properties: approximate values for simulation.
"""

# ------------------------------------------------------------------ #
# Conversion constant
# ------------------------------------------------------------------ #

"""Nepers-to-dB conversion factor: 1 Np = 8.686 dB."""
const NEPER_TO_DB = 8.685889638

"""
    db_cm_to_neper_m(alpha_db_cm)

Convert attenuation from dB/cm/MHz to Nepers/m/MHz.
"""
db_cm_to_neper_m(alpha_db_cm::Real) = alpha_db_cm * 100.0 / NEPER_TO_DB

"""
    neper_m_to_db_cm(alpha_np_m)

Convert attenuation from Nepers/m/MHz to dB/cm/MHz.
"""
neper_m_to_db_cm(alpha_np_m::Real) = alpha_np_m * NEPER_TO_DB / 100.0

# ------------------------------------------------------------------ #
# Acoustic property table
# ------------------------------------------------------------------ #

const ACOUSTIC_TABLE = Dict{Int, AcousticProperties}(
    0 => AcousticProperties(1500.0, 1000.0, 0.0,  4182.0, 0.598),  # water
    1 => AcousticProperties(1610.0, 1090.0, 3.5,  3391.0, 0.37),   # scalp
    2 => AcousticProperties(4080.0, 1900.0, 4.74, 1100.0, 0.30),   # skull
    3 => AcousticProperties(1500.0, 1000.0, 0.0,  4182.0, 0.598),  # CSF
    4 => AcousticProperties(1560.0, 1040.0, 5.3,  3630.0, 0.51),   # gray matter
    5 => AcousticProperties(1560.0, 1040.0, 5.3,  3600.0, 0.50),   # white matter
)

# ------------------------------------------------------------------ #
# MR property table
# ------------------------------------------------------------------ #

const MR_TABLE = Dict{Int, TissueMRProperties}(
    0 => TissueMRProperties(0.0,  0.0,   0.0),   # background
    1 => TissueMRProperties(1.0,  0.05,  0.8),    # scalp
    2 => TissueMRProperties(0.3,  0.02,  0.3),    # skull
    3 => TissueMRProperties(4.0,  2.0,   1.0),    # CSF
    4 => TissueMRProperties(1.6,  0.08,  0.85),   # gray matter
    5 => TissueMRProperties(0.8,  0.07,  0.75),   # white matter
)

# ------------------------------------------------------------------ #
# Kuhl CANN shear modulus table (Pa)
# ------------------------------------------------------------------ #

"""
Kuhl CANN region-specific shear moduli (Pa).

From Linka, St Pierre, Kuhl (2023) Acta Biomaterialia 160:134-151.
Individual regions:
  - Cortex:          1.82 kPa
  - Basal ganglia:   0.88 kPa
  - Corona radiata:  0.94 kPa
  - Corpus callosum: 0.54 kPa

Label-averaged values:
  - GM: average of cortex + basal ganglia = 1.35 kPa
  - WM: average of corona radiata + corpus callosum = 0.74 kPa
"""
const SHEAR_TABLE = Dict{Int, Float64}(
    0 => 0.0,       # background
    1 => 1.1e3,     # scalp (estimated)
    2 => 5.0e3,     # skull (very stiff)
    3 => 0.0,       # CSF (fluid)
    4 => 1.35e3,    # gray matter
    5 => 0.74e3,    # white matter
)

# ------------------------------------------------------------------ #
# Mapping functions
# ------------------------------------------------------------------ #

"""
    map_labels_to_acoustic(labels) -> (sound_speed, density, attenuation)

Map integer tissue labels to acoustic property arrays.
Out-of-range labels default to water (label 0).

Returns arrays of (sound_speed [m/s], density [kg/m^3], attenuation [dB/cm/MHz]).
"""
function map_labels_to_acoustic(labels::AbstractArray{<:Integer})
    default = ACOUSTIC_TABLE[0]
    sound_speed = map(l -> get(ACOUSTIC_TABLE, Int(l), default).sound_speed, labels)
    density     = map(l -> get(ACOUSTIC_TABLE, Int(l), default).density, labels)
    attenuation = map(l -> get(ACOUSTIC_TABLE, Int(l), default).attenuation, labels)
    return (
        collect(Float64, sound_speed),
        collect(Float64, density),
        collect(Float64, attenuation),
    )
end

"""
    map_labels_to_shear_modulus(labels) -> Array{Float64}

Map tissue labels to shear modulus (Pa) using Kuhl CANN values.
"""
function map_labels_to_shear_modulus(labels::AbstractArray{<:Integer})
    return collect(Float64, map(l -> get(SHEAR_TABLE, Int(l), 0.0), labels))
end

"""
    map_labels_to_mr(labels) -> (T1, T2, PD)

Map tissue labels to MR properties (T1 [s], T2 [s], proton density [0-1]).
"""
function map_labels_to_mr(labels::AbstractArray{<:Integer})
    default = MR_TABLE[0]
    T1 = map(l -> get(MR_TABLE, Int(l), default).T1, labels)
    T2 = map(l -> get(MR_TABLE, Int(l), default).T2, labels)
    PD = map(l -> get(MR_TABLE, Int(l), default).PD, labels)
    return (
        collect(Float64, T1),
        collect(Float64, T2),
        collect(Float64, PD),
    )
end
