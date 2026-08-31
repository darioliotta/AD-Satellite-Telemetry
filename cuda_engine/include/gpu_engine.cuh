#pragma once

#include "types.hpp"
#include "utils.hpp"
#include "kernels.cuh"

inline float* upload_to_device(const float* h_buf, size_t count) {
    float* d_buf = nullptr;
    cudaMalloc((void**)&d_buf, count * sizeof(float));
    cudaMemcpy(d_buf, h_buf, count * sizeof(float), cudaMemcpyHostToDevice);
    return d_buf;
}

inline ModelWeights load_all_weights() {
    ModelWeights w;
    size_t w1_sz = 32 * 1 * 7,   b1_sz = 32;
    size_t w2_sz = 64 * 32 * 5,  b2_sz = 64;
    size_t w3_sz = 128 * 64 * 3, b3_sz = 128;
    size_t w_fc1_sz = 64 * (128 * 6), b_fc1_sz = 64;
    size_t w_fc2_sz = 1 * 64,         b_fc2_sz = 1;

    w.h_w1    = load_binary_file("weights/conv1_weight.bin", w1_sz);
    w.h_b1    = load_binary_file("weights/conv1_bias.bin",   b1_sz);
    w.h_w2    = load_binary_file("weights/conv2_weight.bin", w2_sz);
    w.h_b2    = load_binary_file("weights/conv2_bias.bin",   b2_sz);
    w.h_w3    = load_binary_file("weights/conv3_weight.bin", w3_sz);
    w.h_b3    = load_binary_file("weights/conv3_bias.bin",   b3_sz);
    w.h_w_fc1 = load_binary_file("weights/fc1_weight.bin",   w_fc1_sz);
    w.h_b_fc1 = load_binary_file("weights/fc1_bias.bin",     b_fc1_sz);
    w.h_w_fc2 = load_binary_file("weights/fc2_weight.bin",   w_fc2_sz);
    w.h_b_fc2 = load_binary_file("weights/fc2_bias.bin",     b_fc2_sz);

    w.d_w1    = upload_to_device(w.h_w1, w1_sz);
    w.d_b1    = upload_to_device(w.h_b1, b1_sz);
    w.d_w2    = upload_to_device(w.h_w2, w2_sz);
    w.d_b2    = upload_to_device(w.h_b2, b2_sz);
    w.d_w3    = upload_to_device(w.h_w3, w3_sz);
    w.d_b3    = upload_to_device(w.h_b3, b3_sz);
    w.d_w_fc1 = upload_to_device(w.h_w_fc1, w_fc1_sz);
    w.d_b_fc1 = upload_to_device(w.h_b_fc1, b_fc1_sz);
    w.d_w_fc2 = upload_to_device(w.h_w_fc2, w_fc2_sz);
    w.d_b_fc2 = upload_to_device(w.h_b_fc2, b_fc2_sz);

    return w;
}

inline void free_all_weights(ModelWeights& w) {
    free(w.h_w1); free(w.h_b1); free(w.h_w2); free(w.h_b2); free(w.h_w3); free(w.h_b3);
    free(w.h_w_fc1); free(w.h_b_fc1); free(w.h_w_fc2); free(w.h_b_fc2);

    cudaFree(w.d_w1); cudaFree(w.d_b1); cudaFree(w.d_w2); cudaFree(w.d_b2);
    cudaFree(w.d_w3); cudaFree(w.d_b3); cudaFree(w.d_w_fc1); cudaFree(w.d_b_fc1);
    cudaFree(w.d_w_fc2); cudaFree(w.d_b_fc2);
}

inline GpuBuffers allocate_gpu_buffers(int max_batch) {
    GpuBuffers b;
    size_t max_act_size = (size_t)max_batch * 2048;
    cudaMalloc((void**)&b.d_in,     (size_t)max_batch * 50 * sizeof(float));
    cudaMalloc((void**)&b.d_ping,   max_act_size * sizeof(float));
    cudaMalloc((void**)&b.d_pong,   max_act_size * sizeof(float));
    cudaMalloc((void**)&b.d_logits, (size_t)max_batch * sizeof(float));
    return b;
}

inline void free_gpu_buffers(GpuBuffers& b) {
    cudaFree(b.d_in); cudaFree(b.d_ping); cudaFree(b.d_pong); cudaFree(b.d_logits);
}

// ============================================================================
// v0: NAÏVE BASELINE (Dynamic Allocations + Unfused Kernels)
// ============================================================================
inline void forward_pass_gpu_v0_naive(
    const float* d_in,
    float* d_out,
    int batch_size,
    const ModelWeights& w
) {
    dim3 block(64, 1);

    float *d_c1, *d_p1, *d_c2, *d_p2, *d_c3, *d_p3, *d_fc1;
    cudaMalloc((void**)&d_c1,  batch_size * 32 * 50 * sizeof(float));
    cudaMalloc((void**)&d_p1,  batch_size * 32 * 25 * sizeof(float));
    cudaMalloc((void**)&d_c2,  batch_size * 64 * 25 * sizeof(float));
    cudaMalloc((void**)&d_p2,  batch_size * 64 * 12 * sizeof(float));
    cudaMalloc((void**)&d_c3,  batch_size * 128 * 12 * sizeof(float));
    cudaMalloc((void**)&d_p3,  batch_size * 128 * 6 * sizeof(float));
    cudaMalloc((void**)&d_fc1, batch_size * 64 * sizeof(float));

    // Layer 1
    dim3 g_c1((50 + block.x - 1) / block.x, batch_size * 32);
    conv1d_kernel<<<g_c1, block>>>(d_in, w.d_w1, w.d_b1, d_c1, batch_size, 1, 32, 50, 7, 3);
    int c1_elems = batch_size * 32 * 50;
    relu_kernel<<<(c1_elems + 255) / 256, 256>>>(d_c1, c1_elems);
    dim3 g_p1((25 + block.x - 1) / block.x, batch_size * 32);
    maxpool1d_kernel<<<g_p1, block>>>(d_c1, d_p1, batch_size, 32, 50, 25);

    // Layer 2
    dim3 g_c2((25 + block.x - 1) / block.x, batch_size * 64);
    conv1d_kernel<<<g_c2, block>>>(d_p1, w.d_w2, w.d_b2, d_c2, batch_size, 32, 64, 25, 5, 2);
    int c2_elems = batch_size * 64 * 25;
    relu_kernel<<<(c2_elems + 255) / 256, 256>>>(d_c2, c2_elems);
    dim3 g_p2((12 + block.x - 1) / block.x, batch_size * 64);
    maxpool1d_kernel<<<g_p2, block>>>(d_c2, d_p2, batch_size, 64, 25, 12);

    // Layer 3
    dim3 g_c3((12 + block.x - 1) / block.x, batch_size * 128);
    conv1d_kernel<<<g_c3, block>>>(d_p2, w.d_w3, w.d_b3, d_c3, batch_size, 64, 128, 12, 3, 1);
    int c3_elems = batch_size * 128 * 12;
    relu_kernel<<<(c3_elems + 255) / 256, 256>>>(d_c3, c3_elems);
    dim3 g_p3((6 + block.x - 1) / block.x, batch_size * 128);
    maxpool1d_kernel<<<g_p3, block>>>(d_c3, d_p3, batch_size, 128, 12, 6);

    // Dense Layers
    dim3 g_fc1((64 + block.x - 1) / block.x, batch_size);
    dense_kernel<<<g_fc1, block>>>(d_p3, w.d_w_fc1, w.d_b_fc1, d_fc1, batch_size, 128 * 6, 64, true);

    dim3 g_fc2((1 + block.x - 1) / block.x, batch_size);
    dense_kernel<<<g_fc2, block>>>(d_fc1, w.d_w_fc2, w.d_b_fc2, d_out, batch_size, 64, 1, false);

    cudaFree(d_c1); cudaFree(d_p1); cudaFree(d_c2); cudaFree(d_p2);
    cudaFree(d_c3); cudaFree(d_p3); cudaFree(d_fc1);
}

// ============================================================================
// v1: MEMORY & CACHE OPTIMIZED (Static Ping-Pong Buffers + Unfused Kernels)
// ============================================================================
inline void forward_pass_gpu_v1_mem(
    const float* d_in,
    float* d_out,
    int batch_size,
    const ModelWeights& w,
    GpuBuffers& g_buf
) {
    dim3 block(64, 1);

    // Layer 1
    dim3 g_c1((50 + block.x - 1) / block.x, batch_size * 32);
    conv1d_kernel<<<g_c1, block>>>(d_in, w.d_w1, w.d_b1, g_buf.d_pong, batch_size, 1, 32, 50, 7, 3);
    int c1_elems = batch_size * 32 * 50;
    relu_kernel<<<(c1_elems + 255) / 256, 256>>>(g_buf.d_pong, c1_elems);
    dim3 g_p1((25 + block.x - 1) / block.x, batch_size * 32);
    maxpool1d_kernel<<<g_p1, block>>>(g_buf.d_pong, g_buf.d_ping, batch_size, 32, 50, 25);

    // Layer 2
    dim3 g_c2((25 + block.x - 1) / block.x, batch_size * 64);
    conv1d_kernel<<<g_c2, block>>>(g_buf.d_ping, w.d_w2, w.d_b2, g_buf.d_pong, batch_size, 32, 64, 25, 5, 2);
    int c2_elems = batch_size * 64 * 25;
    relu_kernel<<<(c2_elems + 255) / 256, 256>>>(g_buf.d_pong, c2_elems);
    dim3 g_p2((12 + block.x - 1) / block.x, batch_size * 64);
    maxpool1d_kernel<<<g_p2, block>>>(g_buf.d_pong, g_buf.d_ping, batch_size, 64, 25, 12);

    // Layer 3
    dim3 g_c3((12 + block.x - 1) / block.x, batch_size * 128);
    conv1d_kernel<<<g_c3, block>>>(g_buf.d_ping, w.d_w3, w.d_b3, g_buf.d_pong, batch_size, 64, 128, 12, 3, 1);
    int c3_elems = batch_size * 128 * 12;
    relu_kernel<<<(c3_elems + 255) / 256, 256>>>(g_buf.d_pong, c3_elems);
    dim3 g_p3((6 + block.x - 1) / block.x, batch_size * 128);
    maxpool1d_kernel<<<g_p3, block>>>(g_buf.d_pong, g_buf.d_ping, batch_size, 128, 12, 6);

    // Dense Layers
    dim3 g_fc1((64 + block.x - 1) / block.x, batch_size);
    dense_kernel<<<g_fc1, block>>>(g_buf.d_ping, w.d_w_fc1, w.d_b_fc1, g_buf.d_pong, batch_size, 128 * 6, 64, true);

    dim3 g_fc2((1 + block.x - 1) / block.x, batch_size);
    dense_kernel<<<g_fc2, block>>>(g_buf.d_pong, w.d_w_fc2, w.d_b_fc2, d_out, batch_size, 64, 1, false);
}

// ============================================================================
// v2: FULLY OPTIMIZED (Static Buffers + Restrict + Operator Fused Kernel)
// ============================================================================
inline void forward_pass_gpu_v2_fused(
    const float* d_in,
    float* d_out,
    int batch_size,
    const ModelWeights& w,
    GpuBuffers& g_buf
) {
    dim3 block(64, 1);

    dim3 g_c1((50 + block.x - 1) / block.x, batch_size * 32);
    conv1d_kernel<<<g_c1, block>>>(d_in, w.d_w1, w.d_b1, g_buf.d_pong, batch_size, 1, 32, 50, 7, 3);

    dim3 g_p1((25 + block.x - 1) / block.x, batch_size * 32);
    relu_maxpool1d_kernel<<<g_p1, block>>>(g_buf.d_pong, g_buf.d_ping, batch_size, 32, 50, 25);

    dim3 g_c2((25 + block.x - 1) / block.x, batch_size * 64);
    conv1d_kernel<<<g_c2, block>>>(g_buf.d_ping, w.d_w2, w.d_b2, g_buf.d_pong, batch_size, 32, 64, 25, 5, 2);

    dim3 g_p2((12 + block.x - 1) / block.x, batch_size * 64);
    relu_maxpool1d_kernel<<<g_p2, block>>>(g_buf.d_pong, g_buf.d_ping, batch_size, 64, 25, 12);

    dim3 g_c3((12 + block.x - 1) / block.x, batch_size * 128);
    conv1d_kernel<<<g_c3, block>>>(g_buf.d_ping, w.d_w3, w.d_b3, g_buf.d_pong, batch_size, 64, 128, 12, 3, 1);

    dim3 g_p3((6 + block.x - 1) / block.x, batch_size * 128);
    relu_maxpool1d_kernel<<<g_p3, block>>>(g_buf.d_pong, g_buf.d_ping, batch_size, 128, 12, 6);

    dim3 g_fc1((64 + block.x - 1) / block.x, batch_size);
    dense_kernel<<<g_fc1, block>>>(g_buf.d_ping, w.d_w_fc1, w.d_b_fc1, g_buf.d_pong, batch_size, 128 * 6, 64, true);

    dim3 g_fc2((1 + block.x - 1) / block.x, batch_size);
    dense_kernel<<<g_fc2, block>>>(g_buf.d_pong, w.d_w_fc2, w.d_b_fc2, d_out, batch_size, 64, 1, false);
}