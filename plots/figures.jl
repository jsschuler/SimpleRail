using CSV
using DataFrames
using Statistics

# This script reads results CSVs and prints summary tables.
# Add a plotting library (e.g., Plots.jl) to generate figures.

function print_coupled_summary()
    !isfile("results/coupled_summary.csv") && return
    df = CSV.read("results/coupled_summary.csv", DataFrame)
    println("=== Coupled problem summary ===")
    for col in names(df)
        println("  $col: $(df[1, col])")
    end
end

function print_decoupled_summary()
    !isfile("results/decoupled.csv") && return
    df = CSV.read("results/decoupled.csv", DataFrame)
    println("=== Decoupled per-corridor welfare ===")
    println(df)
end

function print_sensitivity_summary()
    !isfile("results/sensitivity.csv") && return
    df = CSV.read("results/sensitivity.csv", DataFrame)
    println("=== Sensitivity results ===")
    for grp_key in ["fleet_capacity", "fixed_cost", "d_param"]
        sub = filter(r -> r.varied_param == grp_key, eachrow(df)) |> collect
        println("\nVaried: $grp_key")
        for r in sub
            println("  $(grp_key)=$(getproperty(r, Symbol(grp_key))): " *
                    "coupling_value=$(round(r.coupling_value,digits=3)), " *
                    "sa_sqa_gap=$(round(r.sa_sqa_gap,digits=3))")
        end
    end
end

print_coupled_summary()
print_decoupled_summary()
print_sensitivity_summary()
