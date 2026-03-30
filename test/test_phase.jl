using Test
using DMI
using Statistics: cor

@testset "Phase processing" begin
    RAWDIR = "/data/raw/wand/sub-08033/ses-06/anat"

    # Skip if data not available
    if !isdir(RAWDIR)
        @warn "WAND data not found at $RAWDIR, skipping phase tests"
        return
    end

    TEs = Float64[5, 10, 15, 20, 25, 30, 35]
    n_echoes = 7

    mag_files = [joinpath(RAWDIR, "sub-08033_ses-06_echo-$(lpad(e,2,'0'))_part-mag_MEGRE.nii.gz")
                 for e in 1:n_echoes]
    pha_files = [joinpath(RAWDIR, "sub-08033_ses-06_echo-$(lpad(e,2,'0'))_part-phase_MEGRE.nii.gz")
                 for e in 1:n_echoes]

    @testset "load_multi_echo" begin
        mag4d, pha4d, hdr = load_multi_echo(mag_files, pha_files)
        @test size(mag4d, 4) == n_echoes
        @test size(pha4d, 4) == n_echoes
        @test size(mag4d) == size(pha4d)
        # Phase should be in [-π, π]
        @test minimum(pha4d) >= -Float32(π) - 0.01f0
        @test maximum(pha4d) <= Float32(π) + 0.01f0
        # Magnitude should be non-negative
        @test minimum(mag4d) >= 0
    end

    @testset "robust_brain_mask" begin
        mag4d, _, _ = load_multi_echo(mag_files, pha_files)
        mask = robust_brain_mask(mag4d[:,:,:,1])
        @test eltype(mask) == Bool
        @test ndims(mask) == 3
        frac = sum(mask) / length(mask)
        @test 0.1 < frac < 0.8  # reasonable brain fraction
    end

    @testset "process_phase end-to-end" begin
        result = process_phase(mag_files, pha_files, TEs)

        @test result isa PhaseResult
        @test ndims(result.mask) == 3
        @test ndims(result.unwrapped) == 4
        @test size(result.unwrapped, 4) == n_echoes
        @test ndims(result.b0_hz) == 3
        @test ndims(result.t2star_ms) == 3
        @test ndims(result.r2star_s) == 3

        # T2* should be in physiological range where masked
        brain = result.mask .& (result.t2star_ms .> 0)
        if sum(brain) > 0
            @test minimum(result.t2star_ms[brain]) > 0
            @test maximum(result.t2star_ms[brain]) <= 200.0
        end

        # Validate against existing qMRI derivatives
        ref_path = "/data/raw/wand/derivatives/qmri/sub-08033/ses-06/T2star_map.nii.gz"
        if isfile(ref_path)
            using NIfTI
            ref_t2s = Float64.(niread(ref_path)) .* 1000  # s → ms
            nz_min = min(size(result.t2star_ms, 3), size(ref_t2s, 3))
            valid = result.mask[:,:,1:nz_min] .&
                    (result.t2star_ms[:,:,1:nz_min] .> 0) .&
                    (ref_t2s[:,:,1:nz_min] .> 0) .&
                    (ref_t2s[:,:,1:nz_min] .< 200)
            if sum(valid) > 100
                ours = result.t2star_ms[:,:,1:nz_min][valid]
                theirs = ref_t2s[:,:,1:nz_min][valid]
                r = cor(ours, theirs)
                @test r > 0.7  # reasonable correlation with reference
            end
        end

        # Test save/output
        outdir = mktempdir()
        save_phase_result(result, outdir)
        @test isfile(joinpath(outdir, "brain_mask.nii.gz"))
        @test isfile(joinpath(outdir, "T2star.nii.gz"))
        @test isfile(joinpath(outdir, "B0_map_hz.nii.gz"))
        @test isfile(joinpath(outdir, "R2star.nii.gz"))
        @test isfile(joinpath(outdir, "unwrapped_echo1.nii.gz"))
        @test isfile(joinpath(outdir, "voxel_quality.nii.gz"))
        @test isfile(joinpath(outdir, "sensitivity_map.nii.gz"))
    end
end
