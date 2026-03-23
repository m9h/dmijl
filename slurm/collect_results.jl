#!/usr/bin/env julia
"""Collect all surrogate sweep results into a leaderboard."""

using JSON, Printf

results_dir = joinpath(@__DIR__, "..", "results")
files = filter(f -> startswith(f, "surrogate_") && endswith(f, ".json"), readdir(results_dir))

results = []
for f in files
    push!(results, JSON.parsefile(joinpath(results_dir, f)))
end

sort!(results, by=r -> r["median_rel_error"])

println("=" ^ 70)
println("SURROGATE SWEEP LEADERBOARD")
println("=" ^ 70)
@printf("  %-25s  %8s  %6s  %6s  %6s\n", "Name", "Med.Err%", "<1%", "<5%", "Time")
println("-" ^ 70)
for r in results
    passed = r["spec1_passed"] ? " ✓" : ""
    @printf("  %-25s  %7.2f%%  %5.1f%%  %5.1f%%  %5.0fs%s\n",
        r["name"],
        r["median_rel_error"] * 100,
        r["pct_under_1pct"],
        r["pct_under_5pct"],
        r["train_time_s"],
        passed,
    )
end
