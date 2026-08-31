#include "../include/kernels.cuh"

__global__ void relu_kernel(float* __restrict__ data, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = data[idx];
        data[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

__global__ void maxpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_length,
    int out_length
) {
    int out_t = blockIdx.x * blockDim.x + threadIdx.x;
    int b     = blockIdx.y / channels;
    int c     = blockIdx.y % channels;

    if (out_t < out_length && b < batch_size) {
        int in_ch_offset  = (b * channels + c) * in_length;
        int out_ch_offset = (b * channels + c) * out_length;

        int in_t0 = out_t * 2;
        int in_t1 = in_t0 + 1;

        float val0 = input[in_ch_offset + in_t0];
        float val1 = (in_t1 < in_length) ? input[in_ch_offset + in_t1] : -1e30f;

        output[out_ch_offset + out_t] = (val0 > val1) ? val0 : val1;
    }
}

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
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int b  = blockIdx.y / out_channels;
    int oc = blockIdx.y % out_channels;

    if (t < length && b < batch_size) {
        float acc = bias[oc];
        int input_batch_offset = b * (in_channels * length);

        for (int ic = 0; ic < in_channels; ++ic) {
            int in_ch_offset = input_batch_offset + ic * length;
            int w_ch_offset  = (oc * in_channels + ic) * kernel_size;

            for (int k = 0; k < kernel_size; ++k) {
                int in_t = t + k - pad;
                if (in_t >= 0 && in_t < length) {
                    float in_val = input[in_ch_offset + in_t];
                    float w_val  = weights[w_ch_offset + k];
                    acc += in_val * w_val;
                }
            }
        }

        int out_idx = (b * out_channels + oc) * length + t;
        output[out_idx] = acc;
    }
}

__global__ void relu_maxpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_length,
    int out_length
) {
    int out_t = blockIdx.x * blockDim.x + threadIdx.x;
    int b     = blockIdx.y / channels;
    int c     = blockIdx.y % channels;

    if (out_t < out_length && b < batch_size) {
        int in_ch_offset  = (b * channels + c) * in_length;
        int out_ch_offset = (b * channels + c) * out_length;

        int in_t0 = out_t * 2;
        int in_t1 = in_t0 + 1;

        float val0 = input[in_ch_offset + in_t0];
        val0 = (val0 > 0.0f) ? val0 : 0.0f;

        float val1 = 0.0f;
        if (in_t1 < in_length) {
            val1 = input[in_ch_offset + in_t1];
            val1 = (val1 > 0.0f) ? val1 : 0.0f;
        }

        output[out_ch_offset + out_t] = (val0 > val1) ? val0 : val1;
    }
}

__global__ void dense_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    bool apply_relu
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b       = blockIdx.y;

    if (out_idx < out_features && b < batch_size) {
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