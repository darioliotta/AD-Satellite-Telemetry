#pragma once

#include <algorithm>
#include <math.h>
#include "types.hpp"

inline CpuBuffers allocate_cpu_buffers(int max_batch) {
    CpuBuffers b;
    b.buf_c1  = (float*)malloc(max_batch * 32 * 50 * sizeof(float));
    b.buf_p1  = (float*)malloc(max_batch * 32 * 25 * sizeof(float));
    b.buf_c2  = (float*)malloc(max_batch * 64 * 25 * sizeof(float));
    b.buf_p2  = (float*)malloc(max_batch * 64 * 12 * sizeof(float));
    b.buf_c3  = (float*)malloc(max_batch * 128 * 12 * sizeof(float));
    b.buf_p3  = (float*)malloc(max_batch * 128 * 6 * sizeof(float));
    b.buf_fc1 = (float*)malloc(max_batch * 64 * sizeof(float));
    return b;
}

inline void free_cpu_buffers(CpuBuffers& b) {
    free(b.buf_c1); free(b.buf_p1); free(b.buf_c2); free(b.buf_p2);
    free(b.buf_c3); free(b.buf_p3); free(b.buf_fc1);
}

inline void conv1d_cpu(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int length,
    int kernel_size,
    int pad
) {
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float b_val = bias[oc];
            for (int t = 0; t < length; ++t) {
                float acc = b_val;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int in_t = t + k - pad;
                        if (in_t >= 0 && in_t < length) {
                            int in_idx = b * (in_channels * length) + ic * length + in_t;
                            int w_idx  = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                            acc += input[in_idx] * weights[w_idx];
                        }
                    }
                }
                output[b * (out_channels * length) + oc * length + t] = acc;
            }
        }
    }
}

inline void relu_maxpool1d_cpu(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_length,
    int out_length
) {
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int t_out = 0; t_out < out_length; ++t_out) {
                int t_in0 = t_out * 2;
                int t_in1 = t_in0 + 1;

                int base_idx = b * (channels * in_length) + c * in_length;
                float v0 = input[base_idx + t_in0];
                float v1 = (t_in1 < in_length) ? input[base_idx + t_in1] : -1e30f;

                v0 = (v0 > 0.0f) ? v0 : 0.0f;
                v1 = (v1 > 0.0f) ? v1 : 0.0f;

                output[b * (channels * out_length) + c * out_length + t_out] = (v0 > v1) ? v0 : v1;
            }
        }
    }
}

inline void dense_cpu(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    bool apply_relu
) {
    for (int b = 0; b < batch_size; ++b) {
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            float acc = bias[out_idx];
            int in_batch_offset = b * in_features;
            int w_row_offset    = out_idx * in_features;

            for (int i = 0; i < in_features; ++i) {
                acc += input[in_batch_offset + i] * weights[w_row_offset + i];
            }

            if (apply_relu) {
                acc = (acc > 0.0f) ? acc : 0.0f;
            }

            output[b * out_features + out_idx] = acc;
        }
    }
}

inline void forward_pass_cpu(
    const float* input,
    float* output,
    int batch_size,
    const ModelWeights& w,
    CpuBuffers& b
) {
    conv1d_cpu(input, w.h_w1, w.h_b1, b.buf_c1, batch_size, 1, 32, 50, 7, 3);
    relu_maxpool1d_cpu(b.buf_c1, b.buf_p1, batch_size, 32, 50, 25);

    conv1d_cpu(b.buf_p1, w.h_w2, w.h_b2, b.buf_c2, batch_size, 32, 64, 25, 5, 2);
    relu_maxpool1d_cpu(b.buf_c2, b.buf_p2, batch_size, 64, 25, 12);

    conv1d_cpu(b.buf_p2, w.h_w3, w.h_b3, b.buf_c3, batch_size, 64, 128, 12, 3, 1);
    relu_maxpool1d_cpu(b.buf_c3, b.buf_p3, batch_size, 128, 12, 6);

    dense_cpu(b.buf_p3, w.h_w_fc1, w.h_b_fc1, b.buf_fc1, batch_size, 128 * 6, 64, true);
    dense_cpu(b.buf_fc1, w.h_w_fc2, w.h_b_fc2, output, batch_size, 64, 1, false);
}