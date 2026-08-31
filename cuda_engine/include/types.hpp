#pragma once

#include <stdlib.h>

struct ModelWeights {
    float *h_w1, *h_b1, *h_w2, *h_b2, *h_w3, *h_b3, *h_w_fc1, *h_b_fc1, *h_w_fc2, *h_b_fc2;
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_w_fc1, *d_b_fc1, *d_w_fc2, *d_b_fc2;
};

struct CpuBuffers {
    float *buf_c1, *buf_p1, *buf_c2, *buf_p2, *buf_c3, *buf_p3, *buf_fc1;
};

struct GpuBuffers {
    float *d_in, *d_ping, *d_pong, *d_logits;
};

struct BenchmarkResult {
    int batch_size;
    double cpu_mean_ms;
    double cpu_std_ms;
    double gpu_mean_ms;
    double gpu_std_ms;
    double speedup;
    double cpu_throughput;
    double gpu_throughput;
    float max_diff;
};

struct AblationResult {
    int batch_size;
    double v0_naive_mean_ms;
    double v0_naive_std_ms;
    double v1_mem_mean_ms;
    double v1_mem_std_ms;
    double v2_fused_mean_ms;
    double v2_fused_std_ms;
};