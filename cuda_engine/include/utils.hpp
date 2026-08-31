#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

inline float* load_binary_file(const std::string& path, size_t num_floats) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("Error: Failed to open file %s\n", path.c_str());
        exit(1);
    }
    float* buffer = (float*)malloc(num_floats * sizeof(float));
    if (!buffer) {
        printf("Error: Memory allocation failed for %s\n", path.c_str());
        fclose(f);
        exit(1);
    }
    
    size_t read_count = fread(buffer, sizeof(float), num_floats, f);
    if (read_count != num_floats) {
        printf("Error: Read %zu floats out of %zu expected from %s\n", read_count, num_floats, path.c_str());
        free(buffer);
        fclose(f);
        exit(1);
    }
    
    fclose(f);
    return buffer;
}

inline void load_random_samples(
    const char* data_file,
    const char* label_file,
    float* h_input,
    float* h_labels,
    int* selected_indices,
    int batch_size,
    int total_samples,
    int sample_len
) {
    FILE* f_data = fopen(data_file, "rb");
    FILE* f_lbl  = fopen(label_file, "rb");
    if (!f_data || !f_lbl) {
        printf("Error: Failed to open binary files for random sampling.\n");
        if (f_data) fclose(f_data);
        if (f_lbl) fclose(f_lbl);
        exit(1);
    }

    srand((unsigned int)time(NULL));

    for (int i = 0; i < batch_size; ++i) {
        int rand_idx = rand() % total_samples;
        selected_indices[i] = rand_idx;

        fseek(f_data, (long)rand_idx * sample_len * sizeof(float), SEEK_SET);
        size_t read_data = fread(h_input + i * sample_len, sizeof(float), sample_len, f_data);
        if (read_data != (size_t)sample_len) {
            printf("Error: Failed reading sample data at index %d\n", rand_idx);
            fclose(f_data);
            fclose(f_lbl);
            exit(1);
        }

        fseek(f_lbl, (long)rand_idx * sizeof(float), SEEK_SET);
        size_t read_lbl = fread(h_labels + i, sizeof(float), 1, f_lbl);
        if (read_lbl != 1) {
            printf("Error: Failed reading sample label at index %d\n", rand_idx);
            fclose(f_data);
            fclose(f_lbl);
            exit(1);
        }
    }

    fclose(f_data);
    fclose(f_lbl);
}