/**
 * met_sim.cpp — MET causality simulator
 *
 * The packet path through the network is strictly one-directional:
 *
 *   NIC_TX → Leaf_UP → Spine → Leaf_DOWN → NIC_RX
 *
 * This is a DAG (no cycles). Each stage only depends on the one before it.
 * That is why we split the leaf into two separate modules (UP and DOWN).
 * Without this split, Leaf would wait for Spine and Spine would wait for
 * Leaf — a deadlock.
 *
 * Termination sequence (no barriers needed between these):
 *   1. NIC_TX  generates all packets → closes nic_to_leaf queues
 *   2. Leaf_UP drains nic_to_leaf   → closes leaf_to_spine queues
 *   3. Spine   drains leaf_to_spine → closes spine_to_leaf queues
 *   4. Leaf_DOWN drains spine_to_leaf → closes leaf_to_nic queues
 *   5. NIC_RX  drains leaf_to_nic   → done
 *
 * Each stage runs concurrently with the others using MET promises.
 * Leaf_UP and Spine can pipeline: Spine starts processing as soon as
 * Leaf_UP provides its first MET promise, without waiting for Leaf_UP
 * to finish all packets.
 */
#include "met_sim.h"
#include <algorithm>
#include <chrono>
#include <climits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Minimum delays (lookahead values) — using 0.9× base for jitter lower bound
// Min serialization for a 64-byte packet on 100Gbps = 64×8000/100 = 5120 ps
static constexpr int64_t MIN_SER            = 64LL * 8000 / 100;
static constexpr int64_t LEAF_UP_LOOKAHEAD  = (int64_t)(LEAF_SWITCH_DELAY_PS  * 0.9)
                                             + MIN_SER
                                             + (int64_t)(LEAF_TO_SPINE_LINK_PS * 0.9);
static constexpr int64_t SPINE_LOOKAHEAD    = (int64_t)(SPINE_SWITCH_DELAY_PS * 0.9)
                                             + MIN_SER
                                             + (int64_t)(LEAF_TO_SPINE_LINK_PS * 0.9);
static constexpr int64_t LEAF_DOWN_LOOKAHEAD= (int64_t)(LEAF_SWITCH_DELAY_PS  * 0.9)
                                             + MIN_SER
                                             + (int64_t)(NIC_TO_LEAF_LINK_PS   * 0.9);

// ============================================================
// Constructor
// ============================================================
MetSim::MetSim(TopoConfig cfg)
    : cfg_(cfg),
      N_(cfg.total_nics()),
      NL_(cfg.num_leaves),
      NS_(cfg.num_spines),
      NPL_(cfg.nics_per_leaf),
      routing_(cfg.total_nics()),
      leaf_state_(cfg.num_leaves),
      spine_state_(cfg.num_spines),
      nic_leaf_link_free_(cfg.total_nics(), 0),
      leaf_spine_link_free_(cfg.num_leaves * cfg.num_spines, 0),
      spine_leaf_link_free_(cfg.num_spines * cfg.num_leaves, 0),
      leaf_nic_link_free_(cfg.total_nics(), 0),
      leaf_up_cur_(cfg.num_leaves, 0),
      leaf_dn_cur_(cfg.num_leaves, 0),
      spine_cur_(cfg.num_spines, 0),
      leaf_up_done_(cfg.num_leaves, false),
      leaf_dn_done_(cfg.num_leaves, false),
      spine_done_(cfg.num_spines, false),
      nic_rx_done_(cfg.total_nics(), false),
      nic_to_leaf_q_(cfg.total_nics()),
      leaf_to_spine_q_(cfg.num_leaves * cfg.num_spines),
      spine_to_leaf_q_(cfg.num_spines * cfg.num_leaves),
      leaf_to_nic_q_(cfg.total_nics())
{
    for (int i = 0; i < N_; i++)
        routing_[i] = N_ - 1 - i;

    for (int leaf = 0; leaf < NL_; leaf++) {
        auto& ls = leaf_state_[leaf];
        ls.fwd_table.resize(N_);
        ls.port_counters.resize(NPL_ + NS_, 0);
        ls.total_pkts = ls.total_bytes = ls.max_queue = 0;
        for (int dst = 0; dst < N_; dst++) {
            ls.fwd_table[dst] = {(int16_t)cfg_.nic_local_port(dst), -1, 0};
        }
    }

    for (int spine = 0; spine < NS_; spine++) {
        auto& ss = spine_state_[spine];
        ss.fwd_table.resize(N_);
        ss.port_counters.resize(NL_, 0);
        ss.total_pkts = ss.total_bytes = ss.max_queue = 0;
        for (int dst = 0; dst < N_; dst++) {
            ss.fwd_table[dst] = {(int16_t)cfg_.nic_to_leaf(dst), (int16_t)spine, 0};
        }
    }
}

// ============================================================
// Phase 0: NIC TX (sequential)
// Generates all packets and fills nic_to_leaf_q, then closes them.
// ============================================================
void MetSim::phase_nic_tx(FastRNG& rng, Stats& stats) {
    std::vector<int> pkt_seq(N_, 0);

    for (int64_t sim_time = 0; sim_time < SIM_DURATION_PS; sim_time += TICK_PS) {
        // MET promise: next packet arrives no earlier than next tick + minimum delays.
        // Same for all NICs in this tick so compute once.
        int64_t met = (sim_time + TICK_PS)
                    + (int64_t)(SERIALIZATION_DELAY_PS * 0.9)
                    + MIN_SER
                    + (int64_t)(NIC_TO_LEAF_LINK_PS * 0.9);

        for (int nic = 0; nic < N_; nic++) {
            Event e{};
            e.timestamp = sim_time + rng.with_jitter(SERIALIZATION_DELAY_PS);
            e.gen_time  = sim_time;
            e.src_nic   = nic;
            e.dst_nic   = routing_[nic];
            e.pkt_id    = pkt_seq[nic]++;
            e.hop_count = 0;
            e.pkt_size  = 64 + rng.randint(1437);

            int64_t ser    = serialization_ps(e.pkt_size, NIC_LEAF_LINK_GBPS);
            int64_t depart = std::max(e.timestamp, nic_leaf_link_free_[nic]) + ser;
            nic_leaf_link_free_[nic] = depart;
            e.timestamp = depart + rng.with_jitter(NIC_TO_LEAF_LINK_PS);
            e.hop_count++;

            nic_to_leaf_q_[nic].push(e);
            nic_to_leaf_q_[nic].advance_met(met);
            stats.pkts_generated++;
            stats.events_processed++;
        }
    }

    for (int nic = 0; nic < N_; nic++)
        nic_to_leaf_q_[nic].close();
}

// ============================================================
// Leaf UP module
// Reads from nic_to_leaf_q, writes to leaf_to_spine_q.
// Only handles upward (NIC → Spine) direction.
// ============================================================
bool MetSim::process_leaf_up(int leaf, FastRNG& rng) {
    if (leaf_up_done_[leaf]) return false;

    // Compute safe time from all NIC inputs
    int64_t safe = INT64_MAX;
    int base = leaf * NPL_;
    for (int p = 0; p < NPL_; p++)
        safe = std::min(safe, nic_to_leaf_q_[base + p].get_met());

    if (safe <= leaf_up_cur_[leaf]) return false;

    // Drain events from NIC queues and merge-sort across queues
    std::vector<Event> to_process;
    for (int p = 0; p < NPL_; p++)
        nic_to_leaf_q_[base + p].drain(safe, to_process);
    std::sort(to_process.begin(), to_process.end(),
              [](const Event& a, const Event& b){ return a.timestamp < b.timestamp; });

    auto& ls = leaf_state_[leaf];
    for (auto& e : to_process) {
        FwdEntry& fe = ls.fwd_table[e.dst_nic];
        int spine    = (int)(ecmp_hash(e.src_nic, e.dst_nic, e.pkt_id) % (uint32_t)NS_);

        fe.queue_depth += e.pkt_size;
        if (fe.queue_depth > ls.max_queue) ls.max_queue = fe.queue_depth;
        ls.port_counters[NPL_ + spine] += e.pkt_size;
        ls.total_pkts++;
        ls.total_bytes += e.pkt_size;

        int64_t extra = (int64_t)fe.queue_depth / 1000;
        e.timestamp += rng.with_jitter(LEAF_SWITCH_DELAY_PS) + extra;
        e.hop_count++;

        fe.queue_depth -= e.pkt_size;
        if (fe.queue_depth < 0) fe.queue_depth = 0;

        int link_idx   = leaf * NS_ + spine;
        int64_t ser    = serialization_ps(e.pkt_size, LEAF_SPINE_LINK_GBPS);
        int64_t depart = std::max(e.timestamp, leaf_spine_link_free_[link_idx]) + ser;
        leaf_spine_link_free_[link_idx] = depart;
        e.timestamp = depart + rng.with_jitter(LEAF_TO_SPINE_LINK_PS);
        e.hop_count++;

        leaf_to_spine_q_[link_idx].push(e);
    }

    leaf_up_cur_[leaf] = safe;

    // Advance MET on all output queues to allow downstream to progress
    int64_t out_met = safe + LEAF_UP_LOOKAHEAD;
    for (int s = 0; s < NS_; s++)
        leaf_to_spine_q_[leaf * NS_ + s].advance_met(out_met);

    // Termination: all NIC inputs closed and queues drained
    bool all_closed = true;
    for (int p = 0; p < NPL_; p++)
        if (nic_to_leaf_q_[base + p].get_met() != INT64_MAX) { all_closed = false; break; }

    if (all_closed) {
        bool all_empty = true;
        for (int p = 0; p < NPL_; p++)
            if (!nic_to_leaf_q_[base + p].empty()) { all_empty = false; break; }

        if (all_empty) {
            for (int s = 0; s < NS_; s++)
                leaf_to_spine_q_[leaf * NS_ + s].close();
            leaf_up_done_[leaf] = true;
            active_.fetch_sub(1, std::memory_order_release);
        }
    }

    return !to_process.empty();
}

// ============================================================
// Spine module
// Reads from leaf_to_spine_q, writes to spine_to_leaf_q.
// ============================================================
bool MetSim::process_spine(int spine, FastRNG& rng) {
    if (spine_done_[spine]) return false;

    int64_t safe = INT64_MAX;
    for (int l = 0; l < NL_; l++)
        safe = std::min(safe, leaf_to_spine_q_[l * NS_ + spine].get_met());

    if (safe <= spine_cur_[spine]) return false;

    std::vector<Event> to_process;
    for (int l = 0; l < NL_; l++)
        leaf_to_spine_q_[l * NS_ + spine].drain(safe, to_process);
    std::sort(to_process.begin(), to_process.end(),
              [](const Event& a, const Event& b){ return a.timestamp < b.timestamp; });

    auto& ss = spine_state_[spine];
    for (auto& e : to_process) {
        FwdEntry& fe = ss.fwd_table[e.dst_nic];
        int dst_leaf  = fe.out_port;

        fe.queue_depth += e.pkt_size;
        if (fe.queue_depth > ss.max_queue) ss.max_queue = fe.queue_depth;
        ss.port_counters[dst_leaf] += e.pkt_size;
        ss.total_pkts++;
        ss.total_bytes += e.pkt_size;

        int64_t extra = (int64_t)fe.queue_depth / 1000;
        e.timestamp += rng.with_jitter(SPINE_SWITCH_DELAY_PS) + extra;
        e.hop_count++;

        fe.queue_depth -= e.pkt_size;
        if (fe.queue_depth < 0) fe.queue_depth = 0;

        int link_idx   = spine * NL_ + dst_leaf;
        int64_t ser    = serialization_ps(e.pkt_size, LEAF_SPINE_LINK_GBPS);
        int64_t depart = std::max(e.timestamp, spine_leaf_link_free_[link_idx]) + ser;
        spine_leaf_link_free_[link_idx] = depart;
        e.timestamp = depart + rng.with_jitter(LEAF_TO_SPINE_LINK_PS);
        e.hop_count++;

        spine_to_leaf_q_[link_idx].push(e);
    }

    spine_cur_[spine] = safe;

    int64_t out_met = safe + SPINE_LOOKAHEAD;
    for (int l = 0; l < NL_; l++)
        spine_to_leaf_q_[spine * NL_ + l].advance_met(out_met);

    bool all_closed = true;
    for (int l = 0; l < NL_; l++)
        if (leaf_to_spine_q_[l * NS_ + spine].get_met() != INT64_MAX) { all_closed = false; break; }

    if (all_closed) {
        bool all_empty = true;
        for (int l = 0; l < NL_; l++)
            if (!leaf_to_spine_q_[l * NS_ + spine].empty()) { all_empty = false; break; }

        if (all_empty) {
            for (int l = 0; l < NL_; l++)
                spine_to_leaf_q_[spine * NL_ + l].close();
            spine_done_[spine] = true;
            active_.fetch_sub(1, std::memory_order_release);
        }
    }

    return !to_process.empty();
}

// ============================================================
// Leaf DOWN module
// Reads from spine_to_leaf_q, writes to leaf_to_nic_q.
// Only handles downward (Spine → NIC) direction.
// ============================================================
bool MetSim::process_leaf_dn(int leaf, FastRNG& rng) {
    if (leaf_dn_done_[leaf]) return false;

    int64_t safe = INT64_MAX;
    for (int s = 0; s < NS_; s++)
        safe = std::min(safe, spine_to_leaf_q_[s * NL_ + leaf].get_met());

    if (safe <= leaf_dn_cur_[leaf]) return false;

    std::vector<Event> to_process;
    for (int s = 0; s < NS_; s++)
        spine_to_leaf_q_[s * NL_ + leaf].drain(safe, to_process);
    std::sort(to_process.begin(), to_process.end(),
              [](const Event& a, const Event& b){ return a.timestamp < b.timestamp; });

    auto& ls = leaf_state_[leaf];
    int base = leaf * NPL_;
    for (auto& e : to_process) {
        int local_port = cfg_.nic_local_port(e.dst_nic);
        ls.port_counters[local_port] += e.pkt_size;
        ls.total_pkts++;
        ls.total_bytes += e.pkt_size;

        e.timestamp += rng.with_jitter(LEAF_SWITCH_DELAY_PS);
        e.hop_count++;

        int64_t ser    = serialization_ps(e.pkt_size, NIC_LEAF_LINK_GBPS);
        int64_t depart = std::max(e.timestamp, leaf_nic_link_free_[e.dst_nic]) + ser;
        leaf_nic_link_free_[e.dst_nic] = depart;
        e.timestamp = depart + rng.with_jitter(NIC_TO_LEAF_LINK_PS);
        e.hop_count++;

        leaf_to_nic_q_[e.dst_nic].push(e);
    }

    leaf_dn_cur_[leaf] = safe;

    int64_t out_met = safe + LEAF_DOWN_LOOKAHEAD;
    for (int p = 0; p < NPL_; p++)
        leaf_to_nic_q_[base + p].advance_met(out_met);

    bool all_closed = true;
    for (int s = 0; s < NS_; s++)
        if (spine_to_leaf_q_[s * NL_ + leaf].get_met() != INT64_MAX) { all_closed = false; break; }

    if (all_closed) {
        bool all_empty = true;
        for (int s = 0; s < NS_; s++)
            if (!spine_to_leaf_q_[s * NL_ + leaf].empty()) { all_empty = false; break; }

        if (all_empty) {
            for (int p = 0; p < NPL_; p++)
                leaf_to_nic_q_[base + p].close();
            leaf_dn_done_[leaf] = true;
            active_.fetch_sub(1, std::memory_order_release);
        }
    }

    return !to_process.empty();
}

// ============================================================
// NIC RX module
// ============================================================
bool MetSim::process_nic_rx(int nic, Stats& stats) {
    if (nic_rx_done_[nic]) return false;

    int64_t safe = leaf_to_nic_q_[nic].get_met();
    if (safe == 0) return false;

    std::vector<Event> to_process;
    leaf_to_nic_q_[nic].drain(safe, to_process);

    for (auto& e : to_process) {
        stats.pkts_delivered++;
        stats.total_latency_ps += e.timestamp - e.gen_time;
        stats.bytes_delivered  += e.pkt_size;
        stats.events_processed++;
    }

    if (safe == INT64_MAX && leaf_to_nic_q_[nic].empty()) {
        nic_rx_done_[nic] = true;
        active_.fetch_sub(1, std::memory_order_release);
        return false;
    }

    return !to_process.empty();
}

// ============================================================
// run()
// ============================================================
SimResult MetSim::run(int num_threads) {
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    // Reset all state
    std::fill(nic_leaf_link_free_.begin(),   nic_leaf_link_free_.end(),   0);
    std::fill(leaf_spine_link_free_.begin(), leaf_spine_link_free_.end(), 0);
    std::fill(spine_leaf_link_free_.begin(), spine_leaf_link_free_.end(), 0);
    std::fill(leaf_nic_link_free_.begin(),   leaf_nic_link_free_.end(),   0);
    std::fill(leaf_up_cur_.begin(),          leaf_up_cur_.end(),          0);
    std::fill(leaf_dn_cur_.begin(),          leaf_dn_cur_.end(),          0);
    std::fill(spine_cur_.begin(),            spine_cur_.end(),            0);
    std::fill(leaf_up_done_.begin(),         leaf_up_done_.end(),         false);
    std::fill(leaf_dn_done_.begin(),         leaf_dn_done_.end(),         false);
    std::fill(spine_done_.begin(),           spine_done_.end(),           false);
    std::fill(nic_rx_done_.begin(),          nic_rx_done_.end(),          false);

    for (auto& ls : leaf_state_) {
        ls.total_pkts = ls.total_bytes = ls.max_queue = 0;
        std::fill(ls.port_counters.begin(), ls.port_counters.end(), 0);
        for (auto& fe : ls.fwd_table) fe.queue_depth = 0;
    }
    for (auto& ss : spine_state_) {
        ss.total_pkts = ss.total_bytes = ss.max_queue = 0;
        std::fill(ss.port_counters.begin(), ss.port_counters.end(), 0);
        for (auto& fe : ss.fwd_table) fe.queue_depth = 0;
    }

    // Reset queues with initial MET promises
    for (auto& q : nic_to_leaf_q_)   q.reset();
    for (auto& q : leaf_to_spine_q_) q.reset(LEAF_UP_LOOKAHEAD);
    for (auto& q : spine_to_leaf_q_) q.reset(SPINE_LOOKAHEAD);
    for (auto& q : leaf_to_nic_q_)   q.reset();

    // Active modules: NL leaf_up + NS spine + NL leaf_dn + N nic_rx
    active_.store(NL_ + NS_ + NL_ + N_, std::memory_order_release);

    Stats total_stats{};
    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Phase 0: NIC TX (sequential) ──
    { FastRNG rng(12345); phase_nic_tx(rng, total_stats); }

    // ── Phase 1: Parallel MET processing ──
    std::vector<Stats> thread_stats(num_threads);

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        FastRNG rng(42 + tid * 1000 + 99);

        std::vector<int> my_leaves, my_spines, my_nics;
        for (int l = tid; l < NL_; l += num_threads) my_leaves.push_back(l);
        for (int s = tid; s < NS_; s += num_threads) my_spines.push_back(s);
        for (int n = tid; n < N_;  n += num_threads) my_nics.push_back(n);

        while (active_.load(std::memory_order_acquire) > 0) {
            for (int l : my_leaves) process_leaf_up(l, rng);
            for (int s : my_spines) process_spine  (s, rng);
            for (int l : my_leaves) process_leaf_dn(l, rng);
            for (int n : my_nics)   process_nic_rx (n, thread_stats[tid]);
        }
    }

    for (auto& s : thread_stats) {
        total_stats.pkts_delivered   += s.pkts_delivered;
        total_stats.total_latency_ps += s.total_latency_ps;
        total_stats.bytes_delivered  += s.bytes_delivered;
        total_stats.events_processed += s.events_processed;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    int iters = 0;
    for (int64_t t = 0; t < SIM_DURATION_PS; t += TICK_PS) iters++;

    return {total_stats, wall, iters};
}
