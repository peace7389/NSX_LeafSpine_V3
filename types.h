/**
 * types.h — Shared definitions for the Leaf-Spine Network Simulator v3
 *
 * What changed from v2 and WHY:
 *
 *   Congestion modeling added to link phases.
 *
 *   v2 gave every packet the same fixed propagation delay regardless of
 *   how many other packets were using the same link. This is not realistic —
 *   a real link has a bandwidth limit. If two packets want to leave at the
 *   same time, the second one must wait until the first one finishes.
 *
 *   v3 adds:
 *     1. Link bandwidth (Gbps) for each link type
 *     2. Serialization delay: time to put one packet on the wire
 *        = packet_size_bits / link_bandwidth
 *     3. link_next_free: per-link timestamp tracking when the link is free
 *        Packet departs at: max(arrival_time, link_next_free) + serialization
 *
 *   Effect: packets on busy links queue up and wait, creating variable
 *   latency across the network. Some links become bottlenecks while others
 *   stay idle. This is what makes MET causality meaningful — modules
 *   genuinely advance at different speeds.
 *
 * Topology:
 *          ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
 *          │Spine 0│ │Spine 1│ │Spine 2│ │Spine 3│
 *          └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
 *              │╲        │╲        │╲        │╲
 *             ╱│ ╲      ╱│ ╲      ╱│ ╲      ╱│ ╲
 *      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
 *      │Leaf0│ │Leaf1│ │Leaf2│ │Leaf3│
 *      └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
 *       ╱╱╲╲   ╱╱╲╲   ╱╱╲╲   ╱╱╲╲
 *      NICs    NICs    NICs    NICs
 */
#pragma once

#include <cstdint>
#include <vector>

// ============================================================================
// Topology parameters
// ============================================================================
struct TopoConfig {
    int num_spines;
    int num_leaves;
    int nics_per_leaf;

    int total_nics()     const { return num_leaves * nics_per_leaf; }
    int total_switches() const { return num_spines + num_leaves; }
    int nic_to_leaf(int nic_id)     const { return nic_id / nics_per_leaf; }
    int nic_local_port(int nic_id)  const { return nic_id % nics_per_leaf; }
};

// ============================================================================
// Timing constants (picoseconds)
// ============================================================================
static constexpr int64_t PS_PER_NS = 1'000;
static constexpr int64_t PS_PER_US = 1'000'000;

static constexpr int64_t NIC_TO_LEAF_LINK_PS   = 100 * PS_PER_NS;  // propagation
static constexpr int64_t LEAF_TO_SPINE_LINK_PS  = 500 * PS_PER_NS;  // propagation
static constexpr int64_t LEAF_SWITCH_DELAY_PS   = 200 * PS_PER_NS;
static constexpr int64_t SPINE_SWITCH_DELAY_PS  = 200 * PS_PER_NS;
static constexpr int64_t NIC_DELAY_PS           = 100 * PS_PER_NS;
static constexpr int64_t SERIALIZATION_DELAY_PS = 120 * PS_PER_NS;

static constexpr int64_t SIM_DURATION_PS        = 100 * PS_PER_US;  // 0.1 ms

static constexpr int64_t JITTER_PERCENT = 10;

// Base tick (used for iteration counting)
static constexpr int64_t TICK_PS =
    SERIALIZATION_DELAY_PS +
    NIC_TO_LEAF_LINK_PS + LEAF_SWITCH_DELAY_PS +
    LEAF_TO_SPINE_LINK_PS + SPINE_SWITCH_DELAY_PS + LEAF_TO_SPINE_LINK_PS +
    LEAF_SWITCH_DELAY_PS + NIC_TO_LEAF_LINK_PS +
    NIC_DELAY_PS;

// ============================================================================
// Link bandwidth (Gbps) — NEW in v3
//
// NIC-to-Leaf links run at 100 Gbps (standard server NIC speed).
// Leaf-to-Spine links run at 100 Gbps as well.
//
// With 8 NICs per leaf and 4 spines, on average 2 packets compete for each
// leaf-spine link per iteration. This creates real queuing when packets
// arrive at the same time, which is exactly the congestion we want to model.
// ============================================================================
static constexpr int64_t NIC_LEAF_LINK_GBPS   = 100;
static constexpr int64_t LEAF_SPINE_LINK_GBPS = 100;

// Serialization delay: time to transmit one packet over a link.
//   = packet_size_bytes × 8 bits × 1000 ps/ns / link_gbps / 1e9 * 1e9
//   = packet_size_bytes × 8000 / link_gbps  (result in picoseconds)
//
// Example: 1500-byte packet on 100 Gbps link
//   = 1500 × 8000 / 100 = 120,000 ps = 120 ns
inline int64_t serialization_ps(int32_t pkt_bytes, int64_t link_gbps) {
    return (int64_t)pkt_bytes * 8000LL / link_gbps;
}

// ============================================================================
// Forwarding table entry
// ============================================================================
struct FwdEntry {
    int16_t out_port;
    int16_t spine_id;
    int32_t queue_depth;
};

// ============================================================================
// Switch state
// ============================================================================
struct SwitchState {
    std::vector<FwdEntry>  fwd_table;
    std::vector<int64_t>   port_counters;
    int64_t                total_pkts;
    int64_t                total_bytes;
    int32_t                max_queue;
};

// ============================================================================
// Event — a packet at a point in time
// ============================================================================
struct Event {
    int64_t  timestamp;
    int      src_nic;
    int      dst_nic;
    int      pkt_id;
    int      hop_count;
    int32_t  pkt_size;

    bool operator>(const Event& o) const { return timestamp > o.timestamp; }
};

// ============================================================================
// Stats
// ============================================================================
struct Stats {
    int64_t pkts_generated   = 0;
    int64_t pkts_delivered   = 0;
    int64_t total_latency_ps = 0;
    int64_t events_processed = 0;
    int64_t bytes_delivered  = 0;
};

// ============================================================================
// Result
// ============================================================================
struct SimResult {
    Stats  stats;
    double wall_sec;
    int    iterations;
};

// ============================================================================
// ECMP hash — picks which spine to use per packet
// ============================================================================
inline uint32_t ecmp_hash(int src, int dst, int pkt_id) {
    uint32_t h = (uint32_t)src * 2654435761u;
    h ^= (uint32_t)dst * 2246822519u;
    h ^= (uint32_t)pkt_id * 3266489917u;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h;
}

// ============================================================================
// Per-thread RNG (xorshift32)
// ============================================================================
struct FastRNG {
    uint32_t state;

    explicit FastRNG(uint32_t seed) : state(seed ? seed : 1) {}

    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    int64_t with_jitter(int64_t base) {
        int64_t jitter = base * JITTER_PERCENT / 100;
        if (jitter == 0) return base;
        int64_t offset = (int64_t)(next() % (2 * jitter + 1)) - jitter;
        return base + offset;
    }

    int randint(int max) {
        return (int)(next() % (uint32_t)max);
    }
};
