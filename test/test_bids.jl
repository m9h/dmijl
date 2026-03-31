using Test
using DMI

@testset "BIDS Data Loaders" begin
    WAND_ROOT = "/data/raw/wand"

    if !isdir(WAND_ROOT)
        @warn "WAND data not found, skipping BIDS tests"
        return
    end

    @testset "find_bids_subjects" begin
        subjects = find_bids_subjects(WAND_ROOT; session="ses-02")
        @test length(subjects) > 0
        @test all(s -> startswith(s.subject, "sub-"), subjects)
        @test subjects[1].session == "ses-02"
    end

    subj = BIDSSubject(WAND_ROOT, "sub-08033", "ses-02")

    @testset "load_dwi — CHARMED" begin
        data, acq, hdr = load_dwi(subj; acq="CHARMED_dir-AP")
        @test ndims(data) == 4
        @test size(data, 4) == length(acq.bvalues)
        @test length(acq.bvalues) > 10
        @test any(acq.bvalues .== 0)  # has b=0 volumes
        @test any(acq.bvalues .> 100)  # has DW volumes (FSL: s/mm²)
        @test size(acq.gradient_directions, 1) == length(acq.bvalues)
    end

    @testset "load_megre" begin
        subj_7t = BIDSSubject(WAND_ROOT, "sub-08033", "ses-06")
        mag_files, phase_files, TEs, hdr = load_megre(subj_7t)
        @test length(mag_files) >= 7
        @test length(phase_files) >= 7
        @test length(TEs) == length(mag_files)
        @test TEs[1] ≈ 5.0  # first echo at 5 ms
        @test issorted(TEs)
    end

    @testset "load_qmt" begin
        volumes, protocol, hdr = load_qmt(subj)
        @test length(volumes) >= 10
        @test length(protocol) >= 10
        # Check mt-off reference exists
        @test any(p -> p[1] == "mt-off", protocol)
        # Check flip angles are present
        @test any(p -> p[2] > 300, protocol)  # cumulative MT FA > 300°
        # Check MT offsets
        @test any(p -> p[3] > 10000, protocol)  # far off-resonance
        @test any(p -> p[3] < 2000 && p[3] > 0, protocol)  # near resonance
    end
end
