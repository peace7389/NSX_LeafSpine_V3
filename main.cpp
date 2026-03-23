/**
 * main.cpp — Leaf-Spine Network Simulator v3 — Congestion Modeling
 *
 * Build:
 *   g++ -O2 -fopenmp -std=c++17 -o leafspine \
 *       main.cpp sequential_sim.cpp parallel_sim.cpp
 *
 * Run:
 *   ./leafspine                  # default: 4×8×8 = 64 NICs
 *   ./leafspine 4 32 32          # 4096 NICs
 *   ./leafspine 4 64 32          # 8192 NICs
 *
 * What is new in v3:
 *   Links now have bandwidth limits. Packets that arrive at a busy link
 *   must wait — creating real queuing delay. This makes some links slower
 *   than others depending on traffic load.
 *
 *   The benchmark shows average latency alongside throughput so you can
 *   see how congestion increases end-to-end delay compared to v2.
 */
#include "types.h"
#include "sequential_sim.h"
#include "parallel_sim.h"
#include "met_sim.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int TRIALS = 5;

struct Timing {
    double median, min, max, cv_pct;
};

Timing measure(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    double med = times[times.size() / 2];
    double mn  = times.front();
    double mx  = times.back();
    double sum = 0;
    for (double t : times) sum += (t - med) * (t - med);
    double sd  = std::sqrt(sum / times.size());
    return {med, mn, mx, sd / med * 100.0};
}

// Average latency in nanoseconds
double avg_latency_ns(const Stats& s) {
    if (s.pkts_delivered == 0) return 0;
    return (double)s.total_latency_ps / s.pkts_delivered / 1000.0;
}

int main(int argc, char* argv[]) {
    TopoConfig cfg{8, 16, 16};  // 8 spines × 16 leaves × 16 NICs/leaf = 256 NICs
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    if (argc > 3) {
        cfg.num_spines    = std::atoi(argv[1]);
        cfg.num_leaves    = std::atoi(argv[2]);
        cfg.nics_per_leaf = std::atoi(argv[3]);
    }

    int N = cfg.total_nics();

    printf("\n");
    printf("============================================================\n");
    printf("  Leaf-Spine Simulator v3 — With Congestion Modeling\n");
    printf("============================================================\n");
    printf("  Topology  : %d spines × %d leaves × %d NICs/leaf = %d NICs\n",
           cfg.num_spines, cfg.num_leaves, cfg.nics_per_leaf, N);
    printf("  Link speed: %lld Gbps (NIC↔Leaf)  %lld Gbps (Leaf↔Spine)\n",
           (long long)NIC_LEAF_LINK_GBPS, (long long)LEAF_SPINE_LINK_GBPS);
    printf("  Congestion: packets queue when link is busy\n");
    printf("  Jitter    : ±%lld%% on all propagation delays\n",
           (long long)JITTER_PERCENT);
    printf("  Duration  : 0.1 ms simulated  |  %d trials + warmup\n", TRIALS);
    printf("  Cores     : %d available\n", threads);
    printf("============================================================\n\n");

    // ==================================================================
    // [1] Correctness + Congestion Effect
    //     Compare sequential vs parallel packet counts.
    //     Show average latency to demonstrate congestion is active.
    // ==================================================================
    printf("  [1] Correctness + Congestion Effect\n");
    printf("  ----------------------------------------------------------\n");
    {
        SequentialSim seq(cfg);
        ParallelSim   par(cfg);
        auto sr = seq.run();
        auto pr = par.run(1);

        MetSim met(cfg);
        auto mr = met.run(1);

        printf("  Sequential : %lld pkts  avg latency %.1f ns\n",
               (long long)sr.stats.pkts_delivered, avg_latency_ns(sr.stats));
        printf("  Parallel   : %lld pkts  avg latency %.1f ns\n",
               (long long)pr.stats.pkts_delivered, avg_latency_ns(pr.stats));
        printf("  MET        : %lld pkts  avg latency %.1f ns\n",
               (long long)mr.stats.pkts_delivered, avg_latency_ns(mr.stats));
        printf("  Packets    : %s\n",
               (sr.stats.pkts_delivered == pr.stats.pkts_delivered &&
                sr.stats.pkts_delivered == mr.stats.pkts_delivered)
               ? "ALL MATCH" : "DIFFER (check RNG seeding)");

        // Show serialization cost for a typical packet
        int32_t avg_pkt = 780;
        int64_t ser_nic_leaf  = serialization_ps(avg_pkt, NIC_LEAF_LINK_GBPS);
        int64_t ser_leaf_spine = serialization_ps(avg_pkt, LEAF_SPINE_LINK_GBPS);
        printf("\n  Serialization delay (avg 780-byte packet):\n");
        printf("    NIC↔Leaf   link: %lld ps = %.1f ns\n",
               (long long)ser_nic_leaf,  ser_nic_leaf  / 1000.0);
        printf("    Leaf↔Spine link: %lld ps = %.1f ns\n",
               (long long)ser_leaf_spine, ser_leaf_spine / 1000.0);
        printf("    (propagation: NIC↔Leaf = 100 ns, Leaf↔Spine = 500 ns)\n");
    }
    printf("\n");

    // ==================================================================
    // [2] Sequential vs Parallel
    // ==================================================================
    printf("  [2] Sequential vs Parallel (1 thread)\n");
    printf("  ----------------------------------------------------------\n");

    double seq_med = 0;
    {
        { SequentialSim w(cfg); w.run(); }
        std::vector<double> times;
        Stats last{};
        for (int i = 0; i < TRIALS; i++) {
            SequentialSim s(cfg); auto r = s.run();
            times.push_back(r.wall_sec); last = r.stats;
        }
        auto t = measure(times);
        seq_med = t.median;
        printf("  Sequential:  %.6fs (±%.1f%%)  %lld pkts  %.2f Mpkt/s  avg %.1f ns latency\n",
               t.median, t.cv_pct,
               (long long)last.pkts_delivered,
               last.pkts_delivered / t.median / 1e6,
               avg_latency_ns(last));
    }

    double par1_med = 0;
    {
        { ParallelSim w(cfg); w.run(1); }
        std::vector<double> times;
        Stats last{};
        for (int i = 0; i < TRIALS; i++) {
            ParallelSim s(cfg); auto r = s.run(1);
            times.push_back(r.wall_sec); last = r.stats;
        }
        auto t = measure(times);
        par1_med = t.median;
        printf("  Parallel 1T: %.6fs (±%.1f%%)  %lld pkts  %.2f Mpkt/s  avg %.1f ns latency\n",
               t.median, t.cv_pct,
               (long long)last.pkts_delivered,
               last.pkts_delivered / t.median / 1e6,
               avg_latency_ns(last));
    }

    if (threads > 1) {
        { ParallelSim w(cfg); w.run(threads); }
        std::vector<double> times;
        Stats last{};
        for (int i = 0; i < TRIALS; i++) {
            ParallelSim s(cfg); auto r = s.run(threads);
            times.push_back(r.wall_sec); last = r.stats;
        }
        auto t = measure(times);
        printf("  Parallel %dT: %.6fs (±%.1f%%)  %lld pkts  %.2f Mpkt/s  avg %.1f ns latency\n",
               threads, t.median, t.cv_pct,
               (long long)last.pkts_delivered,
               last.pkts_delivered / t.median / 1e6,
               avg_latency_ns(last));
        printf("  Speedup vs sequential:   %.2fx\n", seq_med / t.median);
        printf("  Speedup vs parallel 1T:  %.2fx\n", par1_med / t.median);
    }
    printf("\n");

    // ==================================================================
    // [3] Topology Scaling
    // ==================================================================
    printf("  [3] Topology Scaling (parallel, 1 thread)\n");
    printf("  ----------------------------------------------------------\n");
    printf("  %-8s  %6s  %10s  %10s  %12s\n",
           "Config", "NICs", "Wall(s)", "Mpkt/s", "Avg Lat(ns)");
    printf("  %-8s  %6s  %10s  %10s  %12s\n",
           "--------", "------", "----------", "----------", "------------");

    struct SC { int sp, lf, npl; const char* label; };
    SC configs[] = {
        {8,  8,   8, "8x8x8"},
        {8, 16,  16, "8x16x16"},
        {8, 32,  16, "8x32x16"},
        {8, 32,  32, "8x32x32"},
        {8, 64,  32, "8x64x32"},
        {8, 64,  64, "8x64x64"},
    };

    for (auto& sc : configs) {
        TopoConfig tc{sc.sp, sc.lf, sc.npl};
        { ParallelSim w(tc); w.run(1); }
        std::vector<double> times;
        Stats last{};
        for (int i = 0; i < TRIALS; i++) {
            ParallelSim s(tc); auto r = s.run(1);
            times.push_back(r.wall_sec); last = r.stats;
        }
        auto t = measure(times);
        printf("  %-8s  %6d  %10.6f  %10.2f  %12.1f\n",
               sc.label, tc.total_nics(), t.median,
               last.pkts_delivered / t.median / 1e6,
               avg_latency_ns(last));
    }
    printf("\n");

    // ==================================================================
    // [4] Thread Scaling
    // ==================================================================
    printf("  [4] Thread Scaling\n");
    printf("  ----------------------------------------------------------\n");

    struct TT { TopoConfig tc; const char* label; };
    std::vector<TT> tests;
    tests.push_back({cfg, "User config"});
    if (N < 512)  tests.push_back({{8, 32, 16}, "512 NICs"});
    if (N < 1024) tests.push_back({{8, 32, 32}, "1024 NICs"});
    if (N < 2048) tests.push_back({{8, 64, 32}, "2048 NICs"});

    for (auto& tt : tests) {
        int tn = tt.tc.total_nics();
        printf("\n  %s (%d NICs)\n", tt.label, tn);
        printf("  %-8s  %10s  %8s  %8s  %10s  %12s\n",
               "Threads", "Wall(s)", "Speedup", "Effic%", "Mpkt/s", "Avg Lat(ns)");
        printf("  %-8s  %10s  %8s  %8s  %10s  %12s\n",
               "--------", "----------", "--------", "--------", "----------", "------------");

        std::vector<int> tcounts;
        for (int t = 1; t <= threads; t *= 2) tcounts.push_back(t);
        if (tcounts.back() != threads) tcounts.push_back(threads);

        double base = 0;
        int64_t pkts = 0;

        for (int t : tcounts) {
            { ParallelSim w(tt.tc); w.run(t); }
            std::vector<double> times;
            Stats last{};
            for (int i = 0; i < TRIALS; i++) {
                ParallelSim s(tt.tc); auto r = s.run(t);
                times.push_back(r.wall_sec);
                last = r.stats;
                if (pkts == 0) pkts = r.stats.pkts_delivered;
            }
            auto ts = measure(times);
            if (t == 1) base = ts.median;

            double speedup = base / ts.median;
            printf("  %-8d  %10.6f  %7.2fx  %7.1f%%  %10.2f  %12.1f\n",
                   t, ts.median, speedup, speedup / t * 100,
                   pkts / ts.median / 1e6,
                   avg_latency_ns(last));
        }
    }

    // ==================================================================
    // [5] Scaling to Find the Parallel Crossover Point
    //
    // The bottleneck is not GPU vs CPU — it is topology size.
    // Synchronization overhead is roughly fixed per run (~constant number
    // of ticks and phases). As topology grows, useful work grows linearly.
    // At some crossover point, work dominates overhead and parallel wins.
    //
    // "Ideal-1024" shows what 1024 perfectly parallel workers with zero
    // synchronization overhead would achieve (sequential_time / 1024).
    // This represents the algorithm's theoretical ceiling — the gap
    // between this and real 8T results is the synchronization tax.
    // ==================================================================
    printf("  [5] Scaling to Find the Parallel Crossover Point\n");
    printf("  Sequential vs 8T-Barrier vs 8T-MET vs Ideal-1024T (theoretical)\n");
    printf("  ----------------------------------------------------------\n");

    static constexpr int IDEAL_WORKERS = 1024;

    struct MT { TopoConfig tc; const char* label; };
    std::vector<MT> met_tests = {
        {{8,  16,  16},  "256 NICs"},
        {{8,  32,  32}, "1024 NICs"},
        {{8,  64,  32}, "2048 NICs"},
        {{8,  64,  64}, "4096 NICs"},
        {{8, 128,  64}, "8192 NICs"},
        {{8, 128, 128}, "16384 NICs"},
        {{8, 256, 128}, "32768 NICs"},
    };

    printf("  %-12s  %-12s  %10s  %10s  %8s\n",
           "Topology", "Method", "Wall(s)", "Mpkt/s", "Speedup");
    printf("  %-12s  %-12s  %10s  %10s  %8s\n",
           "------------", "------------", "----------", "----------", "--------");

    for (auto& mt : met_tests) {
        // Sequential baseline
        { SequentialSim w(mt.tc); w.run(); }
        std::vector<double> seq_times;
        Stats seq_last{};
        for (int i = 0; i < TRIALS; i++) {
            SequentialSim s(mt.tc); auto r = s.run();
            seq_times.push_back(r.wall_sec); seq_last = r.stats;
        }
        auto st = measure(seq_times);

        // Barrier-based parallel (all threads)
        { ParallelSim w(mt.tc); w.run(threads); }
        std::vector<double> par_times;
        Stats par_last{};
        for (int i = 0; i < TRIALS; i++) {
            ParallelSim s(mt.tc); auto r = s.run(threads);
            par_times.push_back(r.wall_sec); par_last = r.stats;
        }
        auto pt = measure(par_times);

        // MET-based (all threads)
        { MetSim w(mt.tc); w.run(threads); }
        std::vector<double> met_times;
        Stats met_last{};
        for (int i = 0; i < TRIALS; i++) {
            MetSim s(mt.tc); auto r = s.run(threads);
            met_times.push_back(r.wall_sec); met_last = r.stats;
        }
        auto mt2 = measure(met_times);

        // Theoretical ideal: sequential work divided by IDEAL_WORKERS,
        // zero synchronization overhead assumed.
        double ideal_time = st.median / IDEAL_WORKERS;

        printf("  %-12s  %-12s  %10.6f  %10.2f  %8s\n",
               mt.label, "Sequential",
               st.median,
               seq_last.pkts_delivered / st.median / 1e6,
               "1.00x");
        printf("  %-12s  %-12s  %10.6f  %10.2f  %7.2fx\n",
               "", "Barrier-8T",
               pt.median,
               par_last.pkts_delivered / pt.median / 1e6,
               st.median / pt.median);
        printf("  %-12s  %-12s  %10.6f  %10.2f  %7.2fx\n",
               "", "MET-8T",
               mt2.median,
               met_last.pkts_delivered / mt2.median / 1e6,
               st.median / mt2.median);
        printf("  %-12s  %-12s  %10.6f  %10.2f  %7.0fx  (theoretical)\n",
               "", "Ideal-1024T",
               ideal_time,
               seq_last.pkts_delivered / ideal_time / 1e6,
               (double)IDEAL_WORKERS);
        printf("\n");
    }

    printf("\n");
    return 0;
}
