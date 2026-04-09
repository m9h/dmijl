"""
    OpenLIFU Solution I/O.

Load treatment planning solutions from openlifu-python's JSON + netCDF
serialization format. Converts openlifu conventions (mm, W/cm^2) to
SI units (m, W/m^2) used by the ARFI module.

NCDatasets.jl is an optional dependency loaded at runtime.
"""

using JSON3

# ------------------------------------------------------------------ #
# Runtime loading for NCDatasets
# ------------------------------------------------------------------ #

const _NCDATA_LOADED = Ref(false)

"""
    _ensure_ncdata!() -> Bool

Load NCDatasets.jl at runtime. Returns true if available.
"""
function _ensure_ncdata!()
    _NCDATA_LOADED[] && return true
    try
        @eval using NCDatasets
        _NCDATA_LOADED[] = true
        return true
    catch e
        @warn "NCDatasets not available; netCDF loading disabled" exception = e
        return false
    end
end

# ------------------------------------------------------------------ #
# JSON parsing
# ------------------------------------------------------------------ #

"""
    _parse_solution_json(json_path) -> Dict

Parse an openlifu Solution JSON file. Extracts delays, apodizations,
pulse parameters, focal points, and target position.

Unit conversions:
- Target/foci positions: mm -> m (based on Point.units field)
- Delays: seconds (no conversion needed)
"""
function _parse_solution_json(json_path::AbstractString)
    data = open(json_path) do f
        JSON3.read(f)
    end

    # Extract delays and apodizations
    delays_raw = get(data, :delays, nothing)
    apod_raw = get(data, :apodizations, nothing)

    delays = if delays_raw !== nothing
        _to_matrix(delays_raw)
    else
        zeros(1, 1)
    end

    apodizations = if apod_raw !== nothing
        _to_matrix(apod_raw)
    else
        ones(size(delays))
    end

    # Extract pulse parameters
    pulse = get(data, :pulse, Dict())
    frequency = Float64(get(pulse, :frequency, 400e3))
    voltage = Float64(get(data, :voltage, 0.0))
    pulse_duration = Float64(get(pulse, :duration, 20e-6))

    # Extract target position (convert mm -> m)
    target_data = get(data, :target, Dict())
    target_pos = _extract_position(target_data)

    # Extract foci
    foci_data = get(data, :foci, [])
    foci = [_extract_position(f) for f in foci_data]
    if isempty(foci) && !all(target_pos .== 0)
        foci = [copy(target_pos)]
    end

    return Dict(
        "delays" => delays,
        "apodizations" => apodizations,
        "frequency" => frequency,
        "voltage" => voltage,
        "pulse_duration" => pulse_duration,
        "target" => target_pos,
        "foci" => foci,
    )
end

"""
    _extract_position(point_data) -> Vector{Float64}

Extract [x, y, z] position in metres from an openlifu Point dict.
Handles mm -> m conversion based on units field.
"""
function _extract_position(point_data)
    pos = Float64.(get(point_data, :position, [0.0, 0.0, 0.0]))
    units = string(get(point_data, :units, "mm"))
    if units == "mm"
        pos .*= 1e-3
    end
    return pos
end

"""Convert nested arrays/vectors to a Matrix{Float64}."""
function _to_matrix(raw)
    if raw isa AbstractMatrix
        return Float64.(raw)
    elseif raw isa AbstractVector && !isempty(raw) && raw[1] isa AbstractVector
        n_rows = length(raw)
        n_cols = length(raw[1])
        M = zeros(Float64, n_rows, n_cols)
        for (i, row) in enumerate(raw)
            for (j, v) in enumerate(row)
                M[i, j] = Float64(v)
            end
        end
        return M
    else
        return reshape(Float64.(collect(raw)), 1, :)
    end
end

# ------------------------------------------------------------------ #
# NetCDF loading
# ------------------------------------------------------------------ #

"""
    _load_simulation_netcdf(nc_path) -> (p_max, intensity, grid_spacing, coords)

Load openlifu simulation result from netCDF file.

Unit conversions:
- Coordinates: mm -> m (based on xarray units attribute)
- Intensity: W/cm^2 -> W/m^2 (multiply by 1e4)
- Pressure: Pa (no conversion)

Requires NCDatasets.jl (loaded at runtime).
"""
function _load_simulation_netcdf(nc_path::AbstractString)
    _ensure_ncdata!() || error("NCDatasets.jl required for netCDF loading")

    @eval begin
        ds = NCDatasets.Dataset($nc_path)
        try
            # Read pressure and intensity fields
            p_max = if haskey(ds, "p_max")
                Array{Float64}(ds["p_max"][:])
            else
                error("netCDF missing 'p_max' variable")
            end

            intensity = if haskey(ds, "intensity")
                arr = Array{Float64}(ds["intensity"][:])
                # Convert W/cm^2 -> W/m^2
                arr .* 1e4
            else
                # Estimate from p_max: I = p^2 / (2Z), Z ≈ 1.5e6 for water
                p_max .^ 2 ./ (2.0 * 1.5e6)
            end

            # Extract coordinates and grid spacing
            coords = Dict{String, Vector{Float64}}()
            grid_spacing = 1e-3  # default 1 mm

            for dim_name in ["x", "y", "z"]
                if haskey(ds, dim_name)
                    coord_vals = Float64.(ds[dim_name][:])
                    # Check units attribute
                    units = get(ds[dim_name].attrib, "units", "mm")
                    if string(units) == "mm"
                        coord_vals .*= 1e-3
                    end
                    coords[dim_name] = coord_vals
                    if length(coord_vals) > 1
                        grid_spacing = abs(coord_vals[2] - coord_vals[1])
                    end
                end
            end

            (p_max, intensity, grid_spacing, coords)
        finally
            close(ds)
        end
    end
end

# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

"""
    load_tus_solution(json_path; nc_path=nothing) -> TUSSolution

Load an openlifu treatment planning Solution from JSON + netCDF files.

The JSON contains beamforming parameters (delays, apodizations, pulse,
target). The netCDF contains the acoustic simulation result (p_max,
intensity fields).

If `nc_path` is not provided, looks for a .nc file with the same base
name as the JSON file.

# Example
```julia
sol = load_tus_solution("solution.json", nc_path="simulation_result.nc")
result = simulate_arfi_analytical(sol.intensity, labels, seq_params, sol.grid_spacing)
```
"""
function load_tus_solution(json_path::AbstractString;
                           nc_path::Union{Nothing, AbstractString} = nothing)
    # Parse JSON metadata
    meta = _parse_solution_json(json_path)

    # Find netCDF file
    if nc_path === nothing
        base = splitext(json_path)[1]
        nc_path = base * ".nc"
        if !isfile(nc_path)
            nc_path = base * "_simulation_result.nc"
        end
    end

    # Load simulation result
    p_max, intensity, grid_spacing, coords = if isfile(nc_path)
        _load_simulation_netcdf(nc_path)
    else
        @warn "netCDF file not found at $nc_path; using empty fields"
        (zeros(1), zeros(1), 1e-3, Dict{String, Vector{Float64}}())
    end

    return TUSSolution(
        meta["delays"],
        meta["apodizations"],
        meta["frequency"],
        meta["voltage"],
        meta["pulse_duration"],
        meta["foci"],
        meta["target"],
        p_max,
        intensity,
        grid_spacing,
    )
end

"""
    compute_radiation_force_from_solution(sol::TUSSolution, labels) -> Array{Float64}

Convenience: compute radiation force directly from a TUSSolution and tissue labels.
"""
function compute_radiation_force_from_solution(
    sol::TUSSolution,
    labels::AbstractArray{<:Integer},
)
    sound_speed, _, attenuation_db = map_labels_to_acoustic(labels)
    sound_speed = reshape(sound_speed, size(labels))
    attenuation_db = reshape(attenuation_db, size(labels))
    return compute_radiation_force_from_db(
        Float64.(sol.intensity), sound_speed, attenuation_db,
    )
end
