#include <stdio.h>
#include <stdlib.h>
#include "include/types.hpp"
#include "include/utils.hpp"
#include "include/cpu_engine.hpp"
#include "include/gpu_engine.cuh"
#include "include/benchmark.cuh"

int main() {
    const int MAX_BATCH     = 256;
    const int TOTAL_SAMPLES = 56580;
    const int L0            = 50;

    ModelWeights weights = load_all_weights();
    CpuBuffers   cpu_buf = allocate_cpu_buffers(MAX_BATCH);
    GpuBuffers   gpu_buf = allocate_gpu_buffers(MAX_BATCH);

    float* h_input   = (float*)malloc(MAX_BATCH * L0 * sizeof(float));
    float* h_labels  = (float*)malloc(MAX_BATCH * sizeof(float));
    int*   h_indices = (int*)malloc(MAX_BATCH * sizeof(int));

    load_random_samples("data/test_inputs.bin", "data/test_labels.bin", h_input, h_labels, h_indices, MAX_BATCH, TOTAL_SAMPLES, L0);

    BenchmarkResult res_nb = evaluate_batch(1,  50, h_input, weights, cpu_buf, gpu_buf);
    BenchmarkResult res_b  = evaluate_batch(64, 50, h_input, weights, cpu_buf, gpu_buf);

    printf("\n==========================================================================================================\n");
    printf("                  STAGE 1: NON-BATCHING (B=1) vs BATCHING (B=64) (Mean +/- Std)                           \n");
    printf("==========================================================================================================\n");
    printf("%-15s | %-6s | %-17s | %-17s | %-8s | %-14s | %-10s\n",
           "Mode", "Batch", "CPU Lat (ms)", "GPU Lat (ms)", "Speedup", "GPU Thput(s/s)", "Max Diff");
    printf("----------------------------------------------------------------------------------------------------------\n");
    printf("%-15s | %-6d | %7.2f +/- %4.2f | %7.2f +/- %4.2f | %7.2fx | %12.1f   | %.2e\n",
           "Non-Batching", res_nb.batch_size, res_nb.cpu_mean_ms, res_nb.cpu_std_ms, res_nb.gpu_mean_ms, res_nb.gpu_std_ms, res_nb.speedup, res_nb.gpu_throughput, res_nb.max_diff);
    printf("%-15s | %-6d | %7.2f +/- %4.2f | %7.2f +/- %4.2f | %7.2fx | %12.1f   | %.2e\n",
           "Batching", res_b.batch_size, res_b.cpu_mean_ms, res_b.cpu_std_ms, res_b.gpu_mean_ms, res_b.gpu_std_ms, res_b.speedup, res_b.gpu_throughput, res_b.max_diff);
    printf("==========================================================================================================\n");

    FILE* fp = fopen("benchmarks/non_batching_vs_batching.csv", "w");
    if (fp) {
        fprintf(fp, "mode,batch_size,cpu_lat_mean_ms,cpu_lat_std_ms,gpu_lat_mean_ms,gpu_lat_std_ms,speedup,cpu_throughput_sps,gpu_throughput_sps,max_diff\n");
        fprintf(fp, "non_batching,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2e\n",
                res_nb.batch_size, res_nb.cpu_mean_ms, res_nb.cpu_std_ms, res_nb.gpu_mean_ms, res_nb.gpu_std_ms, res_nb.speedup, res_nb.cpu_throughput, res_nb.gpu_throughput, res_nb.max_diff);
        fprintf(fp, "batching,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2e\n",
                res_b.batch_size, res_b.cpu_mean_ms, res_b.cpu_std_ms, res_b.gpu_mean_ms, res_b.gpu_std_ms, res_b.speedup, res_b.cpu_throughput, res_b.gpu_throughput, res_b.max_diff);
        fclose(fp);
        printf("Benchmark saved to 'benchmarks/non_batching_vs_batching.csv'\n");
    }

    run_batch_scaling_benchmark(h_input, weights, cpu_buf, gpu_buf);
    run_ablation_study(h_input, weights, gpu_buf);
    run_layer_profiling_benchmark(64, 50, h_input, weights, gpu_buf);
    run_full_dataset_evaluation(weights, cpu_buf, gpu_buf, TOTAL_SAMPLES, 256);

    free(h_input); free(h_labels); free(h_indices);
    free_cpu_buffers(cpu_buf);
    free_gpu_buffers(gpu_buf);
    free_all_weights(weights);

    return 0;
}