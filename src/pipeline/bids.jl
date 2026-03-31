"""
    BIDS data loaders for dMRI and qMRI datasets.

Handles BIDS-formatted NIfTI + JSON sidecar loading for:
- Diffusion MRI (DWI): bval/bvec + NIfTI
- Multi-echo GRE (MEGRE): per-echo magnitude + phase
- Quantitative MRI (VFA, QMT): per-acquisition volumes + JSON metadata
"""

using NIfTI
using JSON3

"""
    BIDSSubject(; root, subject, session=nothing)

Reference to a BIDS subject directory.
"""
struct BIDSSubject
    root::String
    subject::String
    session::Union{Nothing, String}
end

function bids_path(subj::BIDSSubject, modality::String)
    if subj.session !== nothing
        return joinpath(subj.root, subj.subject, subj.session, modality)
    else
        return joinpath(subj.root, subj.subject, modality)
    end
end

"""
    load_dwi(subj::BIDSSubject; acq=nothing) -> (data, acquisition, mask_or_nothing)

Load diffusion-weighted imaging data from a BIDS subject.

Returns:
- `data`: 4D Float32 array (x, y, z, volumes)
- `acquisition`: Acquisition struct with bvalues, gradient_directions, delta, Delta
- `header`: NIfTI header for saving outputs
"""
function load_dwi(subj::BIDSSubject; acq::Union{Nothing,String}=nothing)
    dwi_dir = bids_path(subj, "dwi")
    prefix = "$(subj.subject)"
    if subj.session !== nothing
        prefix *= "_$(subj.session)"
    end
    if acq !== nothing
        prefix *= "_acq-$(acq)"
    end

    # Find DWI file
    nii_pattern = joinpath(dwi_dir, "$(prefix)*_dwi.nii.gz")
    nii_files = filter(f -> occursin("_dwi.nii.gz", f), readdir(dwi_dir, join=true))
    if acq !== nothing
        nii_files = filter(f -> occursin("acq-$(acq)", f), nii_files)
    end
    # Prefer magnitude
    mag_files = filter(f -> occursin("part-mag", f), nii_files)
    nii_file = isempty(mag_files) ? first(nii_files) : first(mag_files)

    # Load bval/bvec
    bval_file = replace(nii_file, ".nii.gz" => ".bval")
    bvec_file = replace(nii_file, ".nii.gz" => ".bvec")

    acq_struct = load_acquisition(bval_file, bvec_file)

    # Check JSON for timing parameters
    json_file = replace(nii_file, ".nii.gz" => ".json")
    if isfile(json_file)
        meta = JSON3.read(read(json_file, String))
        if haskey(meta, :t_bdel)  # WAND AxCaliber big delta
            acq_struct = Acquisition(acq_struct.bvalues, acq_struct.gradient_directions,
                                     acq_struct.delta, meta.t_bdel)
        end
    end

    # Load NIfTI
    img = niread(nii_file)
    data = Float32.(img)
    hdr = header(img)

    return data, acq_struct, hdr
end

"""
    load_megre(subj::BIDSSubject; suffix="MEGRE") -> (mag_files, phase_files, TEs, header)

Find multi-echo GRE magnitude and phase files for a BIDS subject.

Returns file paths and echo times extracted from JSON sidecars.
"""
function load_megre(subj::BIDSSubject; suffix::String="MEGRE")
    anat_dir = bids_path(subj, "anat")
    prefix = "$(subj.subject)"
    if subj.session !== nothing
        prefix *= "_$(subj.session)"
    end

    # Find all echo files
    all_files = readdir(anat_dir, join=true)
    mag_files = sort(filter(f -> occursin("part-mag", f) && occursin(suffix, f) && endswith(f, ".nii.gz"), all_files))
    phase_files = sort(filter(f -> occursin("part-phase", f) && occursin(suffix, f) && endswith(f, ".nii.gz"), all_files))

    # Extract echo times from JSON sidecars
    TEs = Float64[]
    for f in mag_files
        json_file = replace(f, ".nii.gz" => ".json")
        if isfile(json_file)
            meta = JSON3.read(read(json_file, String))
            push!(TEs, meta.EchoTime * 1000)  # s → ms
        end
    end

    hdr = isempty(mag_files) ? nothing : header(niread(first(mag_files)))

    return mag_files, phase_files, TEs, hdr
end

"""
    load_qmt(subj::BIDSSubject) -> (volumes, protocol, header)

Load quantitative magnetization transfer data from a BIDS subject.

Returns:
- `volumes`: Dict mapping labels to 3D arrays
- `protocol`: Vector of (label, flip_angle, mt_offset) tuples
- `header`: NIfTI header
"""
function load_qmt(subj::BIDSSubject)
    anat_dir = bids_path(subj, "anat")
    prefix = "$(subj.subject)"
    if subj.session !== nothing
        prefix *= "_$(subj.session)"
    end

    all_files = sort(filter(f -> occursin("QMT", f) && occursin("part-mag", f) && endswith(f, ".nii.gz"),
                            readdir(anat_dir, join=true)))

    volumes = Dict{String, Array{Float64,3}}()
    protocol = Tuple{String, Int, Int}[]

    for f in all_files
        # Extract label from filename
        basename_f = basename(f)
        # Remove prefix and suffix to get the variable part
        label = replace(basename_f, "$(prefix)_" => "", "_part-mag_QMT.nii.gz" => "")
        # Handle run- suffix
        label = replace(label, r"_run-\d+" => "")

        json_file = replace(f, ".nii.gz" => ".json")
        fa = 0
        offset = 0
        if isfile(json_file)
            meta = JSON3.read(read(json_file, String))
            fa_raw = get(meta, :FlipAngle, 0)
            offset_raw = get(meta, :MTOffsetFrequency, 0)
            fa = fa_raw isa Number ? Int(fa_raw) : tryparse(Int, string(fa_raw)) |> x -> x === nothing ? 0 : x
            offset = offset_raw isa Number ? Int(offset_raw) : tryparse(Int, string(offset_raw)) |> x -> x === nothing ? 0 : x
        end

        if haskey(volumes, label)
            # Average repeated measurements
            volumes[label] = (volumes[label] .+ Float64.(niread(f))) ./ 2
        else
            volumes[label] = Float64.(niread(f))
            push!(protocol, (label, fa, offset))
        end
    end

    hdr = isempty(all_files) ? nothing : header(niread(first(all_files)))
    return volumes, protocol, hdr
end

"""
    find_bids_subjects(root; session=nothing) -> Vector{BIDSSubject}

Find all subjects in a BIDS dataset root directory.
"""
function find_bids_subjects(root::String; session::Union{Nothing,String}=nothing)
    subjects = BIDSSubject[]
    for d in readdir(root)
        if startswith(d, "sub-") && isdir(joinpath(root, d))
            push!(subjects, BIDSSubject(root, d, session))
        end
    end
    return sort(subjects, by=s -> s.subject)
end
