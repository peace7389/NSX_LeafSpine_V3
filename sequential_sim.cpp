/**
 * sequential_sim.cpp — Sequential simulator with congestion modeling
 *
 * Change from v2:
 *   Link phases now model bandwidth limits. Each link tracks when it is
 *   free to send the next packet. If a packet arrives before the link
 *   is free, it waits — exactly like a real network under load.
 *
 * The congestion logic in each link phase:
 *
 *   int64_t ser    = serialization_ps(pkt_size, link_gbps);
 *   int64_t depart = max(arrival_time, link_next_free) + ser;
 *   link_next_free = depart;
 *   e.timestamp    = depart + propagation_delay;
 *
 *   If arrival_time >= link_next_free: no waiting, link was idle.
 *   If arrival_time <  link_next_free: packet waits, link was busy.
 */
#include "sequential_sim.h"
#include <algorithm>
#include <chrono>

SequentialSim::SequentialSim(TopoConfig cfg)
    : cfg_(cfg),
      N_(cfg.total_nics()),
      NL_(cfg.num_leaves),
      NS_(cfg.num_spines),
      NPL_(cfg.nics_per_leaf),
      routing_(cfg.total_nics()),
      leaf_state_(cfg.num_leaves),
      spine_state_(cfg.num_spines),
      stats_{},
      nic_leaf_link_free_(cfg.total_nics(), 0),
      leaf_spine_link_free_(cfg.num_leaves * cfg.num_spines, 0),
      spine_leaf_link_free_(cfg.num_spines * cfg.num_leaves, 0),
      leaf_nic_link_free_(cfg.total_nics(), 0)
{
    for (int i = 0; i < N_; i++)
        routing_[i] = N_ - 1 - i;

    for (int leaf = 0; leaf < NL_; leaf++) {
        auto& ls = leaf_state_[leaf];
        ls.fwd_table.resize(N_);
        ls.port_counters.resize(NPL_ + NS_, 0);
        ls.total_pkts = 0;
        ls.total_bytes = 0;
        ls.max_queue = 0;

        for (int dst = 0; dst < N_; dst++) {
            int dst_leaf = cfg_.nic_to_leaf(dst);
            FwdEntry& fe = ls.fwd_table[dst];
            if (dst_leaf == leaf) {
                fe.out_port = (int16_t)cfg_.nic_local_port(dst);
                fe.spine_id = -1;
            } else {
                fe.out_port = -1;
                fe.spine_id = -1;
            }
            fe.queue_depth = 0;
        }
    }

    for (int spine = 0; spine < NS_; spine++) {
        auto& ss = spine_state_[spine];
        ss.fwd_table.resize(N_);
        ss.port_counters.resize(NL_, 0);
        ss.total_pkts = 0;
        ss.total_bytes = 0;
        ss.max_queue = 0;

        for (int dst = 0; dst < N_; dst++) {
            int dst_leaf = cfg_.nic_to_leaf(dst);
            ss.fwd_table[dst].out_port = (int16_t)dst_leaf;
            ss.fwd_table[dst].spine_id = (int16_t)spine;
            ss.fwd_table[dst].queue_depth = 0;
        }
    }
}

SimResult SequentialSim::run() {
    stats_ = {};

    // Reset switch state
    for (auto& ls : leaf_state_) {
        ls.total_pkts = 0; ls.total_bytes = 0; ls.max_queue = 0;
        std::fill(ls.port_counters.begin(), ls.port_counters.end(), 0);
        for (auto& fe : ls.fwd_table) fe.queue_depth = 0;
    }
    for (auto& ss : spine_state_) {
        ss.total_pkts = 0; ss.total_bytes = 0; ss.max_queue = 0;
        std::fill(ss.port_counters.begin(), ss.port_counters.end(), 0);
        for (auto& fe : ss.fwd_table) fe.queue_depth = 0;
    }

    // Reset link state — links start empty
    std::fill(nic_leaf_link_free_.begin(),   nic_leaf_link_free_.end(),   0);
    std::fill(leaf_spine_link_free_.begin(), leaf_spine_link_free_.end(), 0);
    std::fill(spine_leaf_link_free_.begin(), spine_leaf_link_free_.end(), 0);
    std::fill(leaf_nic_link_free_.begin(),   leaf_nic_link_free_.end(),   0);

    // Module-local queues
    std::vector<std::vector<Event>> nic_to_leaf(N_);
    std::vector<std::vector<Event>> link_to_leaf_up(N_);
    std::vector<std::vector<Event>> leaf_to_spine(NL_ * NS_);
    std::vector<std::vector<Event>> link_to_spine(NL_ * NS_);
    std::vector<std::vector<Event>> spine_to_leaf(NS_ * NL_);
    std::vector<std::vector<Event>> link_to_leaf_down(NS_ * NL_);
    std::vector<std::vector<Event>> leaf_to_nic(N_);
    std::vector<std::vector<Event>> link_to_nic(N_);

    std::vector<int> pkt_seq(N_, 0);
    FastRNG rng(12345);

    auto t0 = std::chrono::high_resolution_clock::now();

    int64_t sim_time = 0;
    int iterations = 0;

    while (sim_time < SIM_DURATION_PS) {

        // ── Phase 1: NIC TX ──
        for (int i = 0; i < N_; i++) {
            nic_to_leaf[i].clear();
            Event e{};
            e.timestamp = sim_time + rng.with_jitter(SERIALIZATION_DELAY_PS);
            e.src_nic   = i;
            e.dst_nic   = routing_[i];
            e.pkt_id    = pkt_seq[i]++;
            e.hop_count = 0;
            e.pkt_size  = 64 + rng.randint(1437);
            nic_to_leaf[i].push_back(e);
            stats_.pkts_generated++;
            stats_.events_processed++;
        }

        // ── Phase 2: NIC→Leaf links — WITH CONGESTION ──
        //
        // Each NIC has its own dedicated link to its leaf switch, so packets
        // from different NICs never compete with each other here.
        // However, within a single NIC, if a packet arrives before the link
        // finishes the previous one, it must wait.
        for (int i = 0; i < N_; i++) {
            link_to_leaf_up[i].clear();
            for (auto& e : nic_to_leaf[i]) {
                int64_t ser    = serialization_ps(e.pkt_size, NIC_LEAF_LINK_GBPS);
                int64_t depart = std::max(e.timestamp, nic_leaf_link_free_[i]) + ser;
                nic_leaf_link_free_[i] = depart;
                e.timestamp = depart + rng.with_jitter(NIC_TO_LEAF_LINK_PS);
                e.hop_count++;
                link_to_leaf_up[i].push_back(e);
                stats_.events_processed++;
            }
        }

        // ── Phase 3: Leaf switches (upward) ──
        for (int idx = 0; idx < NL_ * NS_; idx++)
            leaf_to_spine[idx].clear();

        for (int leaf = 0; leaf < NL_; leaf++) {
            auto& ls = leaf_state_[leaf];
            int base = leaf * NPL_;
            for (int p = 0; p < NPL_; p++) {
                int nic = base + p;
                for (auto& e : link_to_leaf_up[nic]) {
                    FwdEntry& fe = ls.fwd_table[e.dst_nic];
                    int spine = (int)(ecmp_hash(e.src_nic, e.dst_nic, e.pkt_id)
                                      % (uint32_t)NS_);

                    fe.queue_depth += e.pkt_size;
                    if (fe.queue_depth > ls.max_queue)
                        ls.max_queue = fe.queue_depth;

                    ls.port_counters[NPL_ + spine] += e.pkt_size;
                    ls.total_pkts++;
                    ls.total_bytes += e.pkt_size;

                    int64_t extra = (int64_t)fe.queue_depth / 1000;
                    e.timestamp += rng.with_jitter(LEAF_SWITCH_DELAY_PS) + extra;
                    e.hop_count++;

                    fe.queue_depth -= e.pkt_size;
                    if (fe.queue_depth < 0) fe.queue_depth = 0;

                    leaf_to_spine[leaf * NS_ + spine].push_back(e);
                    stats_.events_processed++;
                }
            }
        }

        // ── Phase 4: Leaf→Spine links — WITH CONGESTION ──
        //
        // This is where congestion matters most. Multiple NICs on the same
        // leaf all send upward, but ECMP splits them across spines. If two
        // packets hash to the same spine, they compete for the same link.
        for (int idx = 0; idx < NL_ * NS_; idx++) {
            link_to_spine[idx].clear();
            for (auto& e : leaf_to_spine[idx]) {
                int64_t ser    = serialization_ps(e.pkt_size, LEAF_SPINE_LINK_GBPS);
                int64_t depart = std::max(e.timestamp, leaf_spine_link_free_[idx]) + ser;
                leaf_spine_link_free_[idx] = depart;
                e.timestamp = depart + rng.with_jitter(LEAF_TO_SPINE_LINK_PS);
                e.hop_count++;
                link_to_spine[idx].push_back(e);
                stats_.events_processed++;
            }
        }

        // ── Phase 5: Spine switches ──
        for (int idx = 0; idx < NS_ * NL_; idx++)
            spine_to_leaf[idx].clear();

        for (int spine = 0; spine < NS_; spine++) {
            auto& ss = spine_state_[spine];
            for (int src_leaf = 0; src_leaf < NL_; src_leaf++) {
                int in_idx = src_leaf * NS_ + spine;
                for (auto& e : link_to_spine[in_idx]) {
                    FwdEntry& fe = ss.fwd_table[e.dst_nic];
                    int dst_leaf = fe.out_port;

                    fe.queue_depth += e.pkt_size;
                    if (fe.queue_depth > ss.max_queue)
                        ss.max_queue = fe.queue_depth;

                    ss.port_counters[dst_leaf] += e.pkt_size;
                    ss.total_pkts++;
                    ss.total_bytes += e.pkt_size;

                    int64_t extra = (int64_t)fe.queue_depth / 1000;
                    e.timestamp += rng.with_jitter(SPINE_SWITCH_DELAY_PS) + extra;
                    e.hop_count++;

                    fe.queue_depth -= e.pkt_size;
                    if (fe.queue_depth < 0) fe.queue_depth = 0;

                    spine_to_leaf[spine * NL_ + dst_leaf].push_back(e);
                    stats_.events_processed++;
                }
            }
        }

        // ── Phase 6: Spine→Leaf links — WITH CONGESTION ──
        for (int idx = 0; idx < NS_ * NL_; idx++) {
            link_to_leaf_down[idx].clear();
            for (auto& e : spine_to_leaf[idx]) {
                int64_t ser    = serialization_ps(e.pkt_size, LEAF_SPINE_LINK_GBPS);
                int64_t depart = std::max(e.timestamp, spine_leaf_link_free_[idx]) + ser;
                spine_leaf_link_free_[idx] = depart;
                e.timestamp = depart + rng.with_jitter(LEAF_TO_SPINE_LINK_PS);
                e.hop_count++;
                link_to_leaf_down[idx].push_back(e);
                stats_.events_processed++;
            }
        }

        // ── Phase 7: Leaf switches (downward) ──
        for (int i = 0; i < N_; i++)
            leaf_to_nic[i].clear();

        for (int leaf = 0; leaf < NL_; leaf++) {
            auto& ls = leaf_state_[leaf];
            for (int spine = 0; spine < NS_; spine++) {
                int in_idx = spine * NL_ + leaf;
                for (auto& e : link_to_leaf_down[in_idx]) {
                    int local_port = cfg_.nic_local_port(e.dst_nic);
                    ls.port_counters[local_port] += e.pkt_size;
                    ls.total_pkts++;
                    ls.total_bytes += e.pkt_size;

                    e.timestamp += rng.with_jitter(LEAF_SWITCH_DELAY_PS);
                    e.hop_count++;
                    leaf_to_nic[e.dst_nic].push_back(e);
                    stats_.events_processed++;
                }
            }
        }

        // ── Phase 8: Leaf→NIC links — WITH CONGESTION ──
        for (int i = 0; i < N_; i++) {
            link_to_nic[i].clear();
            for (auto& e : leaf_to_nic[i]) {
                int64_t ser    = serialization_ps(e.pkt_size, NIC_LEAF_LINK_GBPS);
                int64_t depart = std::max(e.timestamp, leaf_nic_link_free_[i]) + ser;
                leaf_nic_link_free_[i] = depart;
                e.timestamp = depart + rng.with_jitter(NIC_TO_LEAF_LINK_PS);
                e.hop_count++;
                link_to_nic[i].push_back(e);
                stats_.events_processed++;
            }
        }

        // ── Phase 9: NIC RX ──
        for (int i = 0; i < N_; i++) {
            for (auto& e : link_to_nic[i]) {
                e.timestamp += rng.with_jitter(NIC_DELAY_PS);
                int64_t latency = e.timestamp - sim_time;
                stats_.pkts_delivered++;
                stats_.total_latency_ps += latency;
                stats_.bytes_delivered  += e.pkt_size;
                stats_.events_processed++;
            }
        }

        sim_time += TICK_PS;
        iterations++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();
    return {stats_, wall, iterations};
}
