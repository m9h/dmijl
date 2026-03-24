#!/usr/bin/env julia
"""
Merge MCMRSimulator data chunks into a single training dataset.

Run after all generate_mcmr_data.sh array tasks complete:
    julia --project=. slurm/merge_mcmr_chunks.jl [input_dir] [output_file]

Defaults:
    input_dir  = data/mcmr_chunks
    output_file = data/mcmr_training_data.jld2
"""

using JLD2, Printf

const PROJECT_ROOT = joinpath(@__DIR__, "..")

input_dir  = length(ARGS) >= 1 ? ARGS[1] : joinpath(PROJECT_ROOT, "data/mcmr_chunks")
output_file = length(ARGS) >= 2 ? ARGS[2] : joinpath(PROJECT_ROOT, "data/mcmr_training_data.jld2")

# Find all chunk files
chunk_files = sort(filter(f -> startswith(f, "chunk_") && endswith(f, ".jld2"),
                          readdir(input_dir)))

if isempty(chunk_files)
    error("No chunk files found in $input_dir")
end

println("=" ^ 60)
println("Merging MCMRSimulator training data chunks")
println("=" ^ 60)
println("  Input directory: $input_dir")
println("  Found $(length(chunk_files)) chunk files")

# Load first chunk to get dimensions
first_data = load(joinpath(input_dir, chunk_files[1]))
param_dim  = size(first_data["params"], 1)
signal_dim = size(first_data["signals"], 1)

println("  Parameter dimension: $param_dim")
println("  Signal dimension: $signal_dim")

# Count total samples
total_samples = 0
for f in chunk_files
    data = load(joinpath(input_dir, f))
    total_samples += size(data["params"], 2)
end
println("  Total samples: $total_samples")

# Allocate merged arrays
all_params  = zeros(Float32, param_dim, total_samples)
all_signals = zeros(Float32, signal_dim, total_samples)

# Merge chunks
offset = 0
for (i, f) in enumerate(chunk_files)
    data = load(joinpath(input_dir, f))
    n = size(data["params"], 2)
    all_params[:, offset+1:offset+n]  = data["params"]
    all_signals[:, offset+1:offset+n] = data["signals"]
    offset += n

    if i % 10 == 0 || i == length(chunk_files)
        @printf("  Merged %d/%d chunks (%d samples)\n", i, length(chunk_files), offset)
    end
end

@assert offset == total_samples

# Preserve metadata from first chunk
bval_list = first_data["bval_list"]
n_spins   = first_data["N_SPINS"]
geometry  = first_data["GEOMETRY"]

# Save merged dataset
mkpath(dirname(output_file))
@save output_file all_params all_signals bval_list n_spins geometry total_samples

println()
println("Saved merged dataset: $output_file")
@printf("  params:  (%d, %d)  [%.1f MB]\n",
        size(all_params)..., sizeof(all_params) / 1e6)
@printf("  signals: (%d, %d)  [%.1f MB]\n",
        size(all_signals)..., sizeof(all_signals) / 1e6)
println("  n_spins: $n_spins")
println("  geometry: $geometry")
