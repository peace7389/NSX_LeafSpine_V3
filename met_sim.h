/**
 * met_sim.h — MET (Minimum Enqueue Time) causality simulator
 *
 * Architecture:
 *   Phase 0 (sequential): NIC TX generates all packets into nic_to_leaf queues.
 *   Phase 1 (parallel):   Five module types advance independently using MET.
 *                         No global barriers between them.
 *
 *   Data flow (DAG — no cycles):
 *     NIC_TX → Leaf_UP → Spine → Leaf_DOWN → NIC_RX
 *
 *   The leaf is split into UP and DOWN to avoid deadlock:
 *   if a single Leaf module handled both directions it would wait for Spine
 *   while Spine waited for it — a circular dependency.
 *
 * How MET works:
 *   Each queue carries a "minimum enqueue time" (MET) promise:
 *   "No future event in this queue will have timestamp < MET."
 *
 *   A downstream module reads the MET of all its input queues,
 *   takes the minimum, and processes all events up to that time.
 *   It then promises its own MET to its output queues.
 *
 *   This lets fast modules advance ahead of slow modules
 *   without any global synchronization.
 */
#pragma once
#include "types.h"
#include <atomic>
#include <mutex>
#include <queue>
#include <vector>

// ============================================================
// MetQueue — queue between two modules with MET causality
// ============================================================
struct MetQueue {
    // MET promise written by producer, read by consumer.
    // INT64_MAX = producer is done, no more events ever.
    std::atomic<int64_t> met{0};

    std::mutex mtx;
    // Min-heap ordered by timestamp: smallest timestamp is always at the top.
    // Pushing is O(log n). drain() pops in order — no sort needed at consume time.
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;

    // Producer: push event. Call advance_met() separately to update the promise.
    void push(const Event& e) {
        std::lock_guard<std::mutex> g(mtx);
        events.push(e);
    }

    // Producer: advance MET promise forward. No-op if queue is already closed.
    // Must be called after push() so consumers see the event before acting on MET.
    void advance_met(int64_t new_met) {
        if (met.load(std::memory_order_relaxed) != INT64_MAX)
            met.store(new_met, std::memory_order_release);
    }

    // Producer: no more events will ever arrive.
    void close() { met.store(INT64_MAX, std::memory_order_release); }

    // Consumer: read current MET promise.
    int64_t get_met() const { return met.load(std::memory_order_acquire); }

    // Consumer: pop all events with timestamp <= safe_time, already in order.
    void drain(int64_t safe_time, std::vector<Event>& out) {
        std::lock_guard<std::mutex> g(mtx);
        while (!events.empty() && events.top().timestamp <= safe_time) {
            out.push_back(events.top());
            events.pop();
        }
    }

    bool empty() {
        std::lock_guard<std::mutex> g(mtx);
        return events.empty();
    }

    // Reset queue for a new simulation run.
    void reset(int64_t initial_met = 0) {
        std::lock_guard<std::mutex> g(mtx);
        events = {};
        met.store(initial_met, std::memory_order_relaxed);
    }
};

// ============================================================
// MetSim
// ============================================================
class MetSim {
public:
    explicit MetSim(TopoConfig cfg);
    SimResult run(int num_threads);

private:
    TopoConfig cfg_;
    int N_, NL_, NS_, NPL_;

    std::vector<int>         routing_;
    std::vector<SwitchState> leaf_state_;
    std::vector<SwitchState> spine_state_;

    // Congestion state — owned by the module that produces for each link
    std::vector<int64_t> nic_leaf_link_free_;    // owned by NIC TX
    std::vector<int64_t> leaf_spine_link_free_;  // owned by leaf module
    std::vector<int64_t> spine_leaf_link_free_;  // owned by spine module
    std::vector<int64_t> leaf_nic_link_free_;    // owned by leaf module

    // Per-module current simulation times
    std::vector<int64_t> leaf_up_cur_;   // Leaf UP module
    std::vector<int64_t> leaf_dn_cur_;   // Leaf DOWN module
    std::vector<int64_t> spine_cur_;     // Spine module

    // Module completion tracking
    std::vector<bool>    leaf_up_done_;
    std::vector<bool>    leaf_dn_done_;
    std::vector<bool>    spine_done_;
    std::vector<bool>    nic_rx_done_;
    std::atomic<int>     active_{0};

    // Queues
    std::vector<MetQueue> nic_to_leaf_q_;    // N  — NIC TX → Leaf
    std::vector<MetQueue> leaf_to_spine_q_;  // NL×NS — Leaf → Spine
    std::vector<MetQueue> spine_to_leaf_q_;  // NS×NL — Spine → Leaf
    std::vector<MetQueue> leaf_to_nic_q_;    // N  — Leaf → NIC RX

    // Phase 0: generate all NIC TX packets (single-threaded)
    void phase_nic_tx(FastRNG& rng, Stats& stats);

    // Phase 1 module processors (called in a loop until done)
    bool process_leaf_up(int leaf,  FastRNG& rng);  // NIC → Spine direction
    bool process_leaf_dn(int leaf,  FastRNG& rng);  // Spine → NIC direction
    bool process_spine  (int spine, FastRNG& rng);
    bool process_nic_rx (int nic,   Stats& stats);
};
