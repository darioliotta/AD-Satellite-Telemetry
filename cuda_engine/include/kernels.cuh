#pragma once

// --- KERNEL UNFUSED (Per v0 e v1) ---
__global__ void relu_kernel(
    float* __restrict__ data,
    int total_elements
);

__global__ void maxpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_length,
    int out_length
);

// --- KERNEL CONDIVISI ED OTTIMIZZATI (v1 e v2) ---
__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int length,
    int kernel_size,
    int pad
);

// Kernel Fuso (Solo v2)
__global__ void relu_maxpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_length,
    int out_length
);

__global__ void dense_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    bool apply_relu
);