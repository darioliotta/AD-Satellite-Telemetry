#pragma once

#include <stdio.h>
#include <chrono>
#include <math.h>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include "types.hpp"
#include "cpu_engine.hpp"
#include "gpu_engine.cuh"

inline float sigmoid_val(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline void compute_mean_std(const std::vector<double>& v, double& mean, double& std) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    mean = sum / v.size();
    double sq_sum = 0.0;
    for (double val : v) {
        sq_sum += (val - mean) * (val - mean);
    }
    std = (v.size() > 1) ? sqrt(sq_sum / (v.size() - 1)) : 0.0;
}

inline BenchmarkResult evaluate_batch(
    int batch_size,
    int num_runs,
    const float* h_input,
    const ModelWeights& weights,
    CpuBuffers& cpu_buf,
    GpuBuffers& gpu_buf
) {
    float* h_cpu_out = (float*)malloc(batch_size * sizeof(float));
    float* h_gpu_out = (float*)malloc(batch_size * sizeof(float));

    // Warmup CPU
    forward_pass_cpu(h_input, h_cpu_out, batch_size, weights, cpu_buf);

    std::vector<double> cpu_times_ms(num_runs);
    for (int r = 0; r < num_runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        forward_pass_cpu(h_input, h_cpu_out, batch_size, weights, cpu_buf);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_times_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Warmup GPU
    cudaMemcpy(gpu_buf.d_in, h_input, batch_size * 50 * sizeof(float), cudaMemcpyHostToDevice);
    forward_pass_gpu_v2_fused(gpu_buf.d_in, gpu_buf.d_logits, batch_size, weights, gpu_buf);
    cudaDeviceSynchronize();

    std::vector<double> gpu_times_ms(num_runs);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int r = 0; r < num_runs; ++r) {
        cudaEventRecord(ev_start);
        forward_pass_gpu_v2_fused(gpu_buf.d_in, gpu_buf.d_logits, batch_size, weights, gpu_buf);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        gpu_times_ms[r] = (double)elapsed_ms;
    }

    cudaMemcpy(h_gpu_out, gpu_buf.d_logits, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        max_diff = fmaxf(max_diff, fabsf(h_cpu_out[i] - h_gpu_out[i]));
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free(h_cpu_out);
    free(h_gpu_out);

    BenchmarkResult res;
    res.batch_size = batch_size;
    compute_mean_std(cpu_times_ms, res.cpu_mean_ms, res.cpu_std_ms);
    compute_mean_std(gpu_times_ms, res.gpu_mean_ms, res.gpu_std_ms);
    res.speedup        = res.cpu_mean_ms / res.gpu_mean_ms;
    res.cpu_throughput = (batch_size / (res.cpu_mean_ms / 1000.0));
    res.gpu_throughput = (batch_size / (res.gpu_mean_ms / 1000.0));
    res.max_diff       = max_diff;

    return res;
}

inline void run_batch_scaling_benchmark(
    const float* h_input,
    const ModelWeights& weights,
    CpuBuffers& cpu_buf,
    GpuBuffers& gpu_buf
) {
    int batches[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int num_batches = 9;
    const int runs = 50;

    FILE* fp = fopen("benchmarks/latency_batch_scaling.csv", "w");
    if (fp) {
        fprintf(fp, "batch_size,cpu_lat_mean_ms,cpu_lat_std_ms,gpu_kernel_lat_mean_ms,gpu_kernel_lat_std_ms,speedup,cpu_throughput_sps,gpu_throughput_sps,max_diff\n");
    }

    FILE* fp_ts = fopen("benchmarks/batch_timestamps.csv", "w");
    if (fp_ts) {
        fprintf(fp_ts, "batch_size,start_epoch_ms,end_epoch_ms\n");
    }

    printf("\n==========================================================================================================\n");
    printf("                  STAGE 2: BATCH SIZE SCALING SWEEP (B = 1 -> 256) (50 Runs Mean +/- Std)                  \n");
    printf("==========================================================================================================\n");
    printf("%-6s | %-17s | %-17s | %-8s | %-14s | %-14s | %-10s\n",
           "Batch", "CPU Lat (ms)", "GPU Lat (ms)", "Speedup", "CPU Thput(s/s)", "GPU Thput(s/s)", "Max Diff");
    printf("----------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_batches; ++i) {
        int B = batches[i];

        printf("  [Scaling %d/%d] Profiling B=%-3d (%d runs)... \r", i + 1, num_batches, B, runs);
        fflush(stdout);

        auto t_start = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        BenchmarkResult res = evaluate_batch(B, runs, h_input, weights, cpu_buf, gpu_buf);

        auto t_end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        if (fp_ts) {
            fprintf(fp_ts, "%d,%ld,%ld\n", B, (long)t_start, (long)t_end);
            fflush(fp_ts);
        }

        printf("%-6d | %7.2f +/- %4.2f | %7.2f +/- %4.2f | %7.2fx | %12.1f   | %12.1f   | %.2e\n",
               res.batch_size, res.cpu_mean_ms, res.cpu_std_ms, res.gpu_mean_ms, res.gpu_std_ms, res.speedup, res.cpu_throughput, res.gpu_throughput, res.max_diff);

        if (fp) {
            fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2e\n",
                    res.batch_size, res.cpu_mean_ms, res.cpu_std_ms, res.gpu_mean_ms, res.gpu_std_ms, res.speedup, res.cpu_throughput, res.gpu_throughput, res.max_diff);
            fflush(fp);
        }
    }
    printf("==========================================================================================================\n");
    if (fp) {
        fclose(fp);
        printf("Scaling results saved to 'benchmarks/latency_batch_scaling.csv'\n");
    }
    if (fp_ts) {
        fclose(fp_ts);
        printf("Batch timestamps saved to 'benchmarks/batch_timestamps.csv'\n");
    }
}

inline void run_ablation_study(
    const float* h_input,
    const ModelWeights& weights,
    GpuBuffers& gpu_buf
) {
    int test_batches[] = {1, 16, 64, 256};
    int num_test_b = 4;
    int runs = 50;

    FILE* fp = fopen("benchmarks/ablation_study.csv", "w");
    if (fp) {
        fprintf(fp, "batch_size,v0_naive_mean_ms,v0_naive_std_ms,v1_mem_mean_ms,v1_mem_std_ms,v2_fused_mean_ms,v2_fused_std_ms\n");
    }

    printf("\n==========================================================================================================\n");
    printf("                  STAGE 2.5: ABLATION STUDY (v0 Naive vs v1 Memory vs v2 Fused)                           \n");
    printf("==========================================================================================================\n");
    printf("%-6s | %-19s | %-19s | %-19s | %-10s\n",
           "Batch", "v0 Naive (ms)", "v1 Mem-Opt (ms)", "v2 Fused (ms)", "Total Gain");
    printf("----------------------------------------------------------------------------------------------------------\n");

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int i = 0; i < num_test_b; ++i) {
        int B = test_batches[i];
        cudaMemcpy(gpu_buf.d_in, h_input, B * 50 * sizeof(float), cudaMemcpyHostToDevice);

        std::vector<double> v0_t(runs), v1_t(runs), v2_t(runs);

        printf("  [Ablation %d/%d] Testing B=%-3d (v0 Naive, %d runs)... \r", i + 1, num_test_b, B, runs);
        fflush(stdout);

        // v0
        for (int r = 0; r < runs; ++r) {
            cudaEventRecord(ev_start);
            forward_pass_gpu_v0_naive(gpu_buf.d_in, gpu_buf.d_logits, B, weights);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float tmp = 0;
            cudaEventElapsedTime(&tmp, ev_start, ev_stop);
            v0_t[r] = (double)tmp;
        }

        printf("  [Ablation %d/%d] Testing B=%-3d (v1 Mem-Opt, %d runs)... \r", i + 1, num_test_b, B, runs);
        fflush(stdout);

        // v1
        for (int r = 0; r < runs; ++r) {
            cudaEventRecord(ev_start);
            forward_pass_gpu_v1_mem(gpu_buf.d_in, gpu_buf.d_logits, B, weights, gpu_buf);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float tmp = 0;
            cudaEventElapsedTime(&tmp, ev_start, ev_stop);
            v1_t[r] = (double)tmp;
        }

        printf("  [Ablation %d/%d] Testing B=%-3d (v2 Fused, %d runs)...   \r", i + 1, num_test_b, B, runs);
        fflush(stdout);

        // v2
        for (int r = 0; r < runs; ++r) {
            cudaEventRecord(ev_start);
            forward_pass_gpu_v2_fused(gpu_buf.d_in, gpu_buf.d_logits, B, weights, gpu_buf);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float tmp = 0;
            cudaEventElapsedTime(&tmp, ev_start, ev_stop);
            v2_t[r] = (double)tmp;
        }

        double m0, s0, m1, s1, m2, s2;
        compute_mean_std(v0_t, m0, s0);
        compute_mean_std(v1_t, m1, s1);
        compute_mean_std(v2_t, m2, s2);

        double total_gain = m0 / m2;

        printf("%-6d | %7.2f +/- %4.2f   | %7.2f +/- %4.2f   | %7.2f +/- %4.2f   | %7.2fx\n",
               B, m0, s0, m1, s1, m2, s2, total_gain);

        if (fp) {
            fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", B, m0, s0, m1, s1, m2, s2);
        }
    }
    printf("==========================================================================================================\n");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    if (fp) {
        fclose(fp);
        printf("Ablation study saved to 'benchmarks/ablation_study.csv'\n");
    }
}

inline void run_layer_profiling_benchmark(
    int batch_size,
    int num_runs,
    const float* h_input,
    const ModelWeights& weights,
    GpuBuffers& gpu_buf
) {
    cudaEvent_t e0, e1, e2, e3, e4, e5, e_h2d_start, e_h2d_stop, e_d2h_start, e_d2h_stop;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);
    cudaEventCreate(&e_h2d_start); cudaEventCreate(&e_h2d_stop);
    cudaEventCreate(&e_d2h_start); cudaEventCreate(&e_d2h_stop);

    dim3 block(64, 1);
    float t_h2d = 0, t_c1 = 0, t_c2 = 0, t_c3 = 0, t_fc1 = 0, t_fc2 = 0, t_d2h = 0;
    float tmp = 0;

    printf("\n  [Profiling Layer-by-Layer] Running %d iterations for B=%d...\n", num_runs, batch_size);

    for (int r = 0; r < num_runs; ++r) {
        if ((r + 1) % 10 == 0 || r == num_runs - 1) {
            printf("    Iter [%d/%d] processed\r", r + 1, num_runs);
            fflush(stdout);
        }

        cudaEventRecord(e_h2d_start);
        cudaMemcpy(gpu_buf.d_in, h_input, batch_size * 50 * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(e_h2d_stop);
        cudaEventSynchronize(e_h2d_stop);
        cudaEventElapsedTime(&tmp, e_h2d_start, e_h2d_stop);
        t_h2d += tmp;

        cudaEventRecord(e0);
        dim3 g_c1((50 + block.x - 1) / block.x, batch_size * 32);
        conv1d_kernel<<<g_c1, block>>>(gpu_buf.d_in, weights.d_w1, weights.d_b1, gpu_buf.d_pong, batch_size, 1, 32, 50, 7, 3);
        dim3 g_p1((25 + block.x - 1) / block.x, batch_size * 32);
        relu_maxpool1d_kernel<<<g_p1, block>>>(gpu_buf.d_pong, gpu_buf.d_ping, batch_size, 32, 50, 25);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&tmp, e0, e1);
        t_c1 += tmp;

        dim3 g_c2((25 + block.x - 1) / block.x, batch_size * 64);
        conv1d_kernel<<<g_c2, block>>>(gpu_buf.d_ping, weights.d_w2, weights.d_b2, gpu_buf.d_pong, batch_size, 32, 64, 25, 5, 2);
        dim3 g_p2((12 + block.x - 1) / block.x, batch_size * 64);
        relu_maxpool1d_kernel<<<g_p2, block>>>(gpu_buf.d_pong, gpu_buf.d_ping, batch_size, 64, 25, 12);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        cudaEventElapsedTime(&tmp, e1, e2);
        t_c2 += tmp;

        dim3 g_c3((12 + block.x - 1) / block.x, batch_size * 128);
        conv1d_kernel<<<g_c3, block>>>(gpu_buf.d_ping, weights.d_w3, weights.d_b3, gpu_buf.d_pong, batch_size, 64, 128, 12, 3, 1);
        dim3 g_p3((6 + block.x - 1) / block.x, batch_size * 128);
        relu_maxpool1d_kernel<<<g_p3, block>>>(gpu_buf.d_pong, gpu_buf.d_ping, batch_size, 128, 12, 6);
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);
        cudaEventElapsedTime(&tmp, e2, e3);
        t_c3 += tmp;

        dim3 g_fc1((64 + block.x - 1) / block.x, batch_size);
        dense_kernel<<<g_fc1, block>>>(gpu_buf.d_ping, weights.d_w_fc1, weights.d_b_fc1, gpu_buf.d_pong, batch_size, 128 * 6, 64, true);
        cudaEventRecord(e4);
        cudaEventSynchronize(e4);
        cudaEventElapsedTime(&tmp, e3, e4);
        t_fc1 += tmp;

        dim3 g_fc2((1 + block.x - 1) / block.x, batch_size);
        dense_kernel<<<g_fc2, block>>>(gpu_buf.d_pong, weights.d_w_fc2, weights.d_b_fc2, gpu_buf.d_logits, batch_size, 64, 1, false);
        cudaEventRecord(e5);
        cudaEventSynchronize(e5);
        cudaEventElapsedTime(&tmp, e4, e5);
        t_fc2 += tmp;

        float* dummy_host = (float*)malloc(batch_size * sizeof(float));
        cudaEventRecord(e_d2h_start);
        cudaMemcpy(dummy_host, gpu_buf.d_logits, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(e_d2h_stop);
        cudaEventSynchronize(e_d2h_stop);
        cudaEventElapsedTime(&tmp, e_d2h_start, e_d2h_stop);
        t_d2h += tmp;
        free(dummy_host);
    }

    t_h2d /= num_runs; t_c1 /= num_runs; t_c2 /= num_runs;
    t_c3  /= num_runs; t_fc1 /= num_runs; t_fc2 /= num_runs; t_d2h /= num_runs;
    float t_kernel_total = t_c1 + t_c2 + t_c3 + t_fc1 + t_fc2;

    printf("\n=========================================================================================\n");
    printf("                  STAGE 3: LAYER-BY-LAYER PROFILING (Batch Size = %d)                     \n", batch_size);
    printf("=========================================================================================\n");
    printf("Host-to-Device (H2D) Transfer   : %8.4f ms\n", t_h2d);
    printf("Layer 1 (Conv1 + ReLU + Pool1)  : %8.4f ms (%5.1f%% of compute)\n", t_c1, (t_c1 / t_kernel_total) * 100.0f);
    printf("Layer 2 (Conv2 + ReLU + Pool2)  : %8.4f ms (%5.1f%% of compute)\n", t_c2, (t_c2 / t_kernel_total) * 100.0f);
    printf("Layer 3 (Conv3 + ReLU + Pool3)  : %8.4f ms (%5.1f%% of compute)\n", t_c3, (t_c3 / t_kernel_total) * 100.0f);
    printf("Dense 1 (FC1 + ReLU)            : %8.4f ms (%5.1f%% of compute)\n", t_fc1, (t_fc1 / t_kernel_total) * 100.0f);
    printf("Dense 2 (FC2 Logits)            : %8.4f ms (%5.1f%% of compute)\n", t_fc2, (t_fc2 / t_kernel_total) * 100.0f);
    printf("Device-to-Host (D2H) Transfer   : %8.4f ms\n", t_d2h);
    printf("-----------------------------------------------------------------------------------------\n");
    printf("Total Pure Kernel Compute       : %8.4f ms\n", t_kernel_total);
    printf("Total End-to-End Latency        : %8.4f ms\n", t_h2d + t_kernel_total + t_d2h);
    printf("=========================================================================================\n");

    FILE* fp = fopen("benchmarks/layer_breakdown.csv", "w");
    if (fp) {
        fprintf(fp, "stage,latency_ms,pct_compute\n");
        fprintf(fp, "h2d,%.4f,0.0\n", t_h2d);
        fprintf(fp, "conv1_pool1,%.4f,%.2f\n", t_c1, (t_c1 / t_kernel_total) * 100.0f);
        fprintf(fp, "conv2_pool2,%.4f,%.2f\n", t_c2, (t_c2 / t_kernel_total) * 100.0f);
        fprintf(fp, "conv3_pool3,%.4f,%.2f\n", t_c3, (t_c3 / t_kernel_total) * 100.0f);
        fprintf(fp, "dense1_fc1,%.4f,%.2f\n", t_fc1, (t_fc1 / t_kernel_total) * 100.0f);
        fprintf(fp, "dense2_fc2,%.4f,%.2f\n", t_fc2, (t_fc2 / t_kernel_total) * 100.0f);
        fprintf(fp, "d2h,%.4f,0.0\n", t_d2h);
        fclose(fp);
        printf("Profiling breakdown saved to 'benchmarks/layer_breakdown.csv'\n");
    }

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);
    cudaEventDestroy(e_h2d_start); cudaEventDestroy(e_h2d_stop);
    cudaEventDestroy(e_d2h_start); cudaEventDestroy(e_d2h_stop);
}

inline void run_full_dataset_evaluation(
    const ModelWeights& weights,
    CpuBuffers& cpu_buf,
    GpuBuffers& gpu_buf,
    int total_samples,
    int batch_size
) {
    printf("\n=========================================================================================\n");
    printf("                  STAGE 4: FULL DATASET EVALUATION (%d Samples)                        \n", total_samples);
    printf("=========================================================================================\n");

    float* all_inputs = load_binary_file("data/test_inputs.bin", (size_t)total_samples * 50);
    float* all_labels = load_binary_file("data/test_labels.bin", (size_t)total_samples);

    FILE* fp = fopen("benchmarks/full_test_predictions.csv", "w");
    if (fp) {
        fprintf(fp, "sample_idx,logit_cpu,logit_gpu,prob_gpu,pred_class,ground_truth,abs_diff\n");
    }

    float* h_cpu_out = (float*)malloc(batch_size * sizeof(float));
    float* h_gpu_out = (float*)malloc(batch_size * sizeof(float));

    int num_batches = (total_samples + batch_size - 1) / batch_size;
    int tp = 0, fp_count = 0, tn = 0, fn = 0;
    float max_diff_dataset = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < num_batches; ++b) {
        if ((b + 1) % 25 == 0 || b == num_batches - 1) {
            printf("  [Dataset Inference] Processed chunk %d/%d (samples %d/%d)\r",
                   b + 1, num_batches, (b + 1 == num_batches) ? total_samples : (b + 1) * batch_size, total_samples);
            fflush(stdout);
        }

        int offset = b * batch_size;
        int cur_b = (offset + batch_size <= total_samples) ? batch_size : (total_samples - offset);

        const float* cur_input_ptr = all_inputs + (size_t)offset * 50;

        forward_pass_cpu(cur_input_ptr, h_cpu_out, cur_b, weights, cpu_buf);

        cudaMemcpy(gpu_buf.d_in, cur_input_ptr, cur_b * 50 * sizeof(float), cudaMemcpyHostToDevice);
        forward_pass_gpu_v2_fused(gpu_buf.d_in, gpu_buf.d_logits, cur_b, weights, gpu_buf);
        cudaMemcpy(h_gpu_out, gpu_buf.d_logits, cur_b * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < cur_b; ++i) {
            int global_idx = offset + i;
            float l_cpu = h_cpu_out[i];
            float l_gpu = h_gpu_out[i];
            float prob  = sigmoid_val(l_gpu);
            int pred    = (prob >= 0.5f) ? 1 : 0;
            int truth   = (int)all_labels[global_idx];
            float diff  = fabsf(l_cpu - l_gpu);

            if (diff > max_diff_dataset) max_diff_dataset = diff;

            if (pred == 1 && truth == 1) tp++;
            else if (pred == 1 && truth == 0) fp_count++;
            else if (pred == 0 && truth == 0) tn++;
            else if (pred == 0 && truth == 1) fn++;

            if (fp) {
                fprintf(fp, "%d,%.5f,%.5f,%.5f,%d,%d,%.2e\n",
                        global_idx, l_cpu, l_gpu, prob, pred, truth, diff);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    if (fp) {
        fclose(fp);
    }

    float precision = (tp + fp_count > 0) ? (float)tp / (tp + fp_count) : 0.0f;
    float recall    = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
    float f1        = (precision + recall > 0) ? 2.0f * (precision * recall) / (precision + recall) : 0.0f;
    float accuracy  = (float)(tp + tn) / total_samples;

    printf("\nTotal Evaluation Time           : %8.2f s (%8.1f samples/sec)\n", total_sec, total_samples / total_sec);
    printf("Max Diff CPU vs GPU on Dataset  : %.2e (Numerical Match Verified)\n", max_diff_dataset);
    printf("Accuracy                        : %8.2f %%\n", accuracy * 100.0f);
    printf("Precision                       : %8.4f\n", precision);
    printf("Recall                          : %8.4f\n", recall);
    printf("F1-Score                        : %8.4f\n", f1);
    printf("Confusion Matrix                : TP=%d | FP=%d | TN=%d | FN=%d\n", tp, fp_count, tn, fn);
    printf("=========================================================================================\n");
    printf("Full dataset predictions saved to 'benchmarks/full_test_predictions.csv'\n");

    free(all_inputs);
    free(all_labels);
    free(h_cpu_out);
    free(h_gpu_out);
}