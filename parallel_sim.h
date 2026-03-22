/**
 * parallel_sim.h — NSX-style parallel simulator with congestion modeling
 *
 * Same structure as v2 parallel simulator, with link_next_free arrays added.
 *
 * No locking needed on link_next_free arrays: each link index is processed
 * by exactly one thread at a time (enforced by omp for schedule(static)),
 * so there is no concurrent access to the same array element.
 */
#pragma once
#include "types.h"
#include <vector>

class ParallelSim {
public:
    explicit ParallelSim(TopoConfig cfg);
    SimResult run(int num_threads);

private:
    TopoConfig cfg_;
    int N_, NL_, NS_, NPL_;

    std::vector<int>         routing_;
    std::vector<SwitchState> leaf_state_;
    std::vector<SwitchState> spine_state_;

    // Module-local event queues
    std::vector<std::vector<Event>> nic_to_leaf_;
    std::vector<std::vector<Event>> link_to_leaf_up_;
    std::vector<std::vector<Event>> leaf_to_spine_;
    std::vector<std::vector<Event>> link_to_spine_;
    std::vector<std::vector<Event>> spine_to_leaf_;
    std::vector<std::vector<Event>> link_to_leaf_down_;
    std::vector<std::vector<Event>> leaf_to_nic_;
    std::vector<std::vector<Event>> link_to_nic_;

    // Per-link "next free" timestamps — NEW in v3
    std::vector<int64_t> nic_leaf_link_free_;
    std::vector<int64_t> leaf_spine_link_free_;
    std::vector<int64_t> spine_leaf_link_free_;
    std::vector<int64_t> leaf_nic_link_free_;

    std::vector<int>   pkt_seq_;
    std::vector<Stats> nic_stats_;
};
