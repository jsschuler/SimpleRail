const FIRMS = [
    :pennsylvania,
    :ny_central,
    :erie,
    :bando,
    :new_haven,
    :lackawanna,
]

const CORRIDORS = [:ny_phl, :phl_pit, :pit_chi]

const PARTICIPATION = Dict(
    (:pennsylvania, :ny_phl)  => true,
    (:pennsylvania, :phl_pit) => true,
    (:pennsylvania, :pit_chi) => true,
    (:ny_central,  :ny_phl)  => true,
    (:ny_central,  :phl_pit) => true,
    (:ny_central,  :pit_chi) => true,
    (:erie,        :ny_phl)  => true,
    (:erie,        :phl_pit) => true,
    (:erie,        :pit_chi) => true,
    (:bando,       :ny_phl)  => true,
    (:bando,       :phl_pit) => true,
    (:bando,       :pit_chi) => true,
    (:new_haven,   :ny_phl)  => true,
    (:lackawanna,  :ny_phl)  => true,
    (:lackawanna,  :phl_pit) => true,
)

"""
    active_pairs(participation)

Return all (firm, corridor) pairs where the firm participates.
"""
function active_pairs(participation=PARTICIPATION)
    [(f, k) for (f, k) in keys(participation) if get(participation, (f, k), false)]
end

"""
    active_corridors_for_firm(firm, participation)

Return the list of corridors in which `firm` participates.
"""
function active_corridors_for_firm(firm, participation=PARTICIPATION)
    [k for k in CORRIDORS if get(participation, (firm, k), false)]
end

"""
    active_firms_for_corridor(corridor, participation)

Return the list of firms participating in `corridor`.
"""
function active_firms_for_corridor(corridor, participation=PARTICIPATION)
    [f for f in FIRMS if get(participation, (f, corridor), false)]
end

"""
    build_var_index(participation)

Build the canonical variable index mapping
  (firm, corridor, var_type) -> Int
where var_type ∈ {:b_low, :b_mid, :b_hi, :freq}.

Variables are ordered: iterate over all active (firm, corridor) pairs in a
fixed order, and for each pair emit b_low, b_mid, b_hi, freq in that order.
"""
function build_var_index(participation=PARTICIPATION)
    pairs = sort(active_pairs(participation), by=p -> (string(p[1]), string(p[2])))
    idx = Dict{Tuple{Symbol,Symbol,Symbol}, Int}()
    i = 1
    for (f, k) in pairs
        idx[(f, k, :b_low)] = i;  i += 1
        idx[(f, k, :b_mid)] = i;  i += 1
        idx[(f, k, :b_hi)]  = i;  i += 1
        idx[(f, k, :freq)]  = i;  i += 1
    end
    idx
end

const RATE_LOW  = 1.0
const RATE_MID  = 2.0
const RATE_HIGH = 3.0

const RATE_LEVELS = [:b_low, :b_mid, :b_hi]
const RATE_VALUES = Dict(:b_low => RATE_LOW, :b_mid => RATE_MID, :b_hi => RATE_HIGH)
