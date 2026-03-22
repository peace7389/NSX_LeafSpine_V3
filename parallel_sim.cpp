/**
 * parallel_sim.cpp — NSX-style parallel simulator with congestion modeling
 *
 * Change from v2:
 *   Link phases (2, 4, 6, 8) now model bandwidth limits using link_next_free.
 *   Everything else is identical to v2.
 *
 * Thread safety of link_next_free:
 *   Each link index maps to exactly one thread via schedule(static).
 *   Thread 0 always owns link indices 0..k, thread 1 owns k+1..2k, etc.
 *   So link_next_free[idx] is only ever written by one thread — no locks needed.
 */
#include "parallel_sim.h"
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

ParallelSim::ParallelSim(TopoConfig cfg)
    : cfg_(cfg),
      N_(cfg.total_nics()),
      NL_(cfg.num_leaves),
      NS_(cfg.num_spines),
      NPL_(cfg.nics_per_leaf),
      routing_(cfg.total_nics()),
      leaf_state_(cfg.num_leaves),
      spine_state_(cfg.num_spines),
      nic_to_leaf_(cfg.total_nics()),
      link_to_leaf_up_(cfg.total_nics()),
      leaf_to_spine_(cfg.num_leaves * cfg.num_spines),
      link_to_spine_(cfg.num_leaves * cfg.num_spines),
      spine_to_leaf_(cfg.num_spines * cfg.num_leaves),
      link_to_leaf_down_(cfg.num_spines * cfg.num_leaves),
      leaf_to_nic_(cfg.total_nics()),
      link_to_nic_(cfg.total_nics()),
      nic_leaf_link_free_(cfg.total_nics(), 0),
      leaf_spine_link_free_(cfg.num_leaves * cfg.num_spines, 0),
      spine_leaf_link_free_(cfg.num_spines * cfg.num_leaves, 0),
      leaf_nic_link_free_(cfg.total_nics(), 0),
      pkt_seq_(cfg.total_nics(), 0),
      nic_stats_(cfg.total_nics())
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

SimResult ParallelSim::run(int num_threads) {
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    // Reset
    for (int i = 0; i < N_; i++) {
        pkt_seq_[i] = 0;
        nic_stats_[i] = {};
    }
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

    // Reset link state
    std::fill(nic_leaf_link_free_.begin(),   nic_leaf_link_free_.end(),   0);
    std::fill(leaf_spine_link_free_.begin(), leaf_spine_link_free_.end(), 0);
    std::fill(spine_leaf_link_free_.begin(), spine_leaf_link_free_.end(), 0);
    std::fill(leaf_nic_link_free_.begin(),   leaf_nic_link_free_.end(),   0);

    int total_iters = 0;
    for (int64_t t = 0; t < SIM_DURATION_PS; t += TICK_PS) total_iters++;

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        FastRNG rng(42 + tid * 1000);

        for (int iter = 0; iter < total_iters; iter++) {
            int64_t sim_time = iter * TICK_PS;
            rng.state = 42 + tid * 1000 + iter * 7;

            // ── Phase 1: NIC TX ──
            #pragma omp for schedule(static)
            for (int i = 0; i < N_; i++) {
                nic_to_leaf_[i].clear();
                Event e{};
                e.timestamp = sim_time + rng.with_jitter(SERIALIZATION_DELAY_PS);
                e.src_nic   = i;
                e.dst_nic   = routing_[i];
                e.pkt_id    = pkt_seq_[i]++;
                e.hop_count = 0;
                e.pkt_size  = 64 + rng.randint(1437);
                nic_to_leaf_[i].push_back(e);
                nic_stats_[i].pkts_generated++;
                nic_stats_[i].events_processed++;
            }

            // ── Phase 2: NIC→Leaf links — WITH CONGESTION ──
            #pragma omp for schedule(static)
            for (int i = 0; i < N_; i++) {
                link_to_leaf_up_[i].clear();
                for (auto& e : nic_to_leaf_[i]) {
                    int64_t ser    = serialization_ps(e.pkt_size, NIC_LEAF_LINK_GBPS);
                    int64_t depart = std::max(e.timestamp, nic_leaf_link_free_[i]) + ser;
                    nic_leaf_link_free_[i] = depart;
                    e.timestamp = depart + rng.with_jitter(NIC_TO_LEAF_LINK_PS);
                    e.hop_count++;
                    link_to_leaf_up_[i].push_back(e);
                    nic_stats_[i].events_processed++;
                }
            }

            // ── Phase 3: Leaf switches (upward) ──
            #pragma omp for schedule(static)
            for (int idx = 0; idx < NL_ * NS_; idx++)
                leaf_to_spine_[idx].clear();

            #pragma omp for schedule(static)
            for (int leaf = 0; leaf < NL_; leaf++) {
                auto& ls = leaf_state_[leaf];
                int base = leaf * NPL_;
                for (int p = 0; p < NPL_; p++) {
                    int nic = base + p;
                    for (auto& e : link_to_leaf_up_[nic]) {
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

                        leaf_to_spine_[leaf * NS_ + spine].push_back(e);
                    }
                }
            }

            // ── Phase 4: Leaf→Spine links — WITH CONGESTION ──
            #pragma omp for schedule(static)
            for (int idx = 0; idx < NL_ * NS_; idx++) {
                link_to_spine_[idx].clear();
                for (auto& e : leaf_to_spine_[idx]) {
                    int64_t ser    = serialization_ps(e.pkt_size, LEAF_SPINE_LINK_GBPS);
                    int64_t depart = std::max(e.timestamp, leaf_spine_link_free_[idx]) + ser;
                    leaf_spine_link_free_[idx] = depart;
                    e.timestamp = depart + rng.with_jitter(LEAF_TO_SPINE_LINK_PS);
                    e.hop_count++;
                    link_to_spine_[idx].push_back(e);
                }
            }

            // ── Phase 5: Spine switches ──
            #pragma omp for schedule(static)
            for (int idx = 0; idx < NS_ * NL_; idx++)
                spine_to_leaf_[idx].clear();

            #pragma omp for schedule(static)
            for (int spine = 0; spine < NS_; spine++) {
                auto& ss = spine_state_[spine];
                for (int src_leaf = 0; src_leaf < NL_; src_leaf++) {
                    int in_idx = src_leaf * NS_ + spine;
                    for (auto& e : link_to_spine_[in_idx]) {
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

                        spine_to_leaf_[spine * NL_ + dst_leaf].push_back(e);
                    }
                }
            }

            // ── Phase 6: Spine→Leaf links — WITH CONGESTION ──
            #pragma omp for schedule(static)
            for (int idx = 0; idx < NS_ * NL_; idx++) {
                link_to_leaf_down_[idx].clear();
                for (auto& e : spine_to_leaf_[idx]) {
                    int64_t ser    = serialization_ps(e.pkt_size, LEAF_SPINE_LINK_GBPS);
                    int64_t depart = std::max(e.timestamp, spine_leaf_link_free_[idx]) + ser;
                    spine_leaf_link_free_[idx] = depart;
                    e.timestamp = depart + rng.with_jitter(LEAF_TO_SPINE_LINK_PS);
                    e.hop_count++;
                    link_to_leaf_down_[idx].push_back(e);
                }
            }

            // ── Phase 7: Leaf switches (downward) ──
            #pragma omp for schedule(static)
            for (int i = 0; i < N_; i++)
                leaf_to_nic_[i].clear();

            #pragma omp for schedule(static)
            for (int leaf = 0; leaf < NL_; leaf++) {
                auto& ls = leaf_state_[leaf];
                for (int spine = 0; spine < NS_; spine++) {
                    int in_idx = spine * NL_ + leaf;
                    for (auto& e : link_to_leaf_down_[in_idx]) {
                        int local_port = cfg_.nic_local_port(e.dst_nic);
                        ls.port_counters[local_port] += e.pkt_size;
                        ls.total_pkts++;
                        ls.total_bytes += e.pkt_size;

                        e.timestamp += rng.with_jitter(LEAF_SWITCH_DELAY_PS);
                        e.hop_count++;
                        leaf_to_nic_[e.dst_nic].push_back(e);
                    }
                }
            }

            // ── Phase 8: Leaf→NIC links — WITH CONGESTION ──
            #pragma omp for schedule(static)
            for (int i = 0; i < N_; i++) {
                link_to_nic_[i].clear();
                for (auto& e : leaf_to_nic_[i]) {
                    int64_t ser    = serialization_ps(e.pkt_size, NIC_LEAF_LINK_GBPS);
                    int64_t depart = std::max(e.timestamp, leaf_nic_link_free_[i]) + ser;
                    leaf_nic_link_free_[i] = depart;
                    e.timestamp = depart + rng.with_jitter(NIC_TO_LEAF_LINK_PS);
                    e.hop_count++;
                    link_to_nic_[i].push_back(e);
                }
            }

            // ── Phase 9: NIC RX ──
            #pragma omp for schedule(static)
            for (int i = 0; i < N_; i++) {
                for (auto& e : link_to_nic_[i]) {
                    e.timestamp += rng.with_jitter(NIC_DELAY_PS);
                    int64_t latency = e.timestamp - sim_time;
                    nic_stats_[i].pkts_delivered++;
                    nic_stats_[i].total_latency_ps += latency;
                    nic_stats_[i].bytes_delivered  += e.pkt_size;
                    nic_stats_[i].events_processed++;
                }
            }

        } // next iteration
    } // end parallel region

    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    Stats total{};
    for (int i = 0; i < N_; i++) {
        total.pkts_generated   += nic_stats_[i].pkts_generated;
        total.pkts_delivered   += nic_stats_[i].pkts_delivered;
        total.total_latency_ps += nic_stats_[i].total_latency_ps;
        total.bytes_delivered  += nic_stats_[i].bytes_delivered;
        total.events_processed += nic_stats_[i].events_processed;
    }
    return {total, wall, total_iters};
}
