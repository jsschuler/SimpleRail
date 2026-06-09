module RailroadPD

using Random
using Statistics
using StatsBase

include("market_structure.jl")
include("demand.jl")
include("qubo.jl")
include("exact_solver.jl")
include("classical_sa.jl")
include("sqa.jl")
include("metrics.jl")

export FIRMS, CORRIDORS, PARTICIPATION
export DEMAND_PARAMS, COST, FIXED_COST
export FLEET_CAPACITY, FLEET_PENALTY, ONEHOT_PENALTY
export build_var_index, random_feasible_config
export build_qubo, compute_welfare_direct, welfare_from_config
export exact_solve_corridor
export run_sa, run_sqa
export run_decoupled_baseline
export strategy_differentiation, fleet_utilization, corridor_welfare
export coupling_value, logical_entropy_profile

end
