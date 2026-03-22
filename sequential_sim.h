/**
 * sequential_sim.h — Baseline sequential simulator with congestion modeling
 */
#pragma once
#include "types.h"
#include <vector>

class SequentialSim {
public:
    explicit SequentialSim(TopoConfig cfg);
    SimResult run();

private:
    TopoConfig cfg_;
    int N_, NL_, NS_, NPL_;

    std::vector<int>         routing_;
    std::vector<SwitchState> leaf_state_;
    std::vector<SwitchState> spine_state_;
    Stats stats_;

    // Per-link "next free" timestamps — NEW in v3
    // Tracks when each link finishes transmitting its current packet.
    // Packet must wait if it arrives before the link is free.
    std::vector<int64_t> nic_leaf_link_free_;    // size N  (one per NIC→Leaf link)
    std::vector<int64_t> leaf_spine_link_free_;  // size NL×NS
    std::vector<int64_t> spine_leaf_link_free_;  // size NS×NL
    std::vector<int64_t> leaf_nic_link_free_;    // size N  (one per Leaf→NIC link)
};
