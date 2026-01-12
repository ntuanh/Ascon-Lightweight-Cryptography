#ifndef SPLIT_INFERENCE_H
#define SPLIT_INFERENCE_H

#include <cmath>
#include <stdint.h>
#include "nn_weights.h"

// Activations
inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

inline float tanh_act(float x) {
    return std::tanh(x);
}

// Split inference: 6 → 16 → 8 → 4
void forward_3layers(
    const float x[INPUT_SIZE],
    float out[4]
);

// Full inference: 6 → 2
void forward_4layers(
    const float x[INPUT_SIZE],
    float out[2]
);

// Quantization
void quantize_nbits(
    const float* x,
    int* q,
    int size,
    int n_bits,
    float& scale
);

void dequantize_nbits(
    const int* q,
    float* x,
    int size,
    float scale
);

#endif
