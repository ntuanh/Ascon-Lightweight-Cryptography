#include "split_inference.h"
#include "nn_weights.h"
#include <cmath>

// ==================================================
// Forward: 3 Linear layers (split inference)
// Input  : 6
// Output : 4
// ==================================================
void forward_3layers(const float x[INPUT_SIZE], float out[4]) {

    float h0[16];
    float h1[8];

    // -------------------------
    // Layer 0: Linear(6 → 16) + ReLU
    // -------------------------
    for (int i = 0; i < 16; i++) {
        float sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += W0[i][j] * x[j];
        }
        h0[i] = relu(sum);
    }

    // -------------------------
    // Layer 1: Linear(16 → 8) + Tanh
    // -------------------------
    for (int i = 0; i < 8; i++) {
        float sum = b1[i];
        for (int j = 0; j < 16; j++) {
            sum += W1[i][j] * h0[j];
        }
        h1[i] = tanh_act(sum);
    }

    // -------------------------
    // Layer 2: Linear(8 → 4) + Tanh
    // -------------------------
    for (int i = 0; i < 4; i++) {
        float sum = b2[i];
        for (int j = 0; j < 8; j++) {
            sum += W2[i][j] * h1[j];
        }
        out[i] = tanh_act(sum);
    }
}

// ==================================================
// Forward: FULL model (all 4 Linear layers)
// Input  : 6
// Output : 2
// ==================================================
void forward_4layers(const float x[INPUT_SIZE], float out[2]) {

    float h2[4];

    // First 3 layers
    forward_3layers(x, h2);

    // -------------------------
    // Layer 3: Linear(4 → 2)
    // -------------------------
    for (int i = 0; i < 2; i++) {
        float sum = b3[i];
        for (int j = 0; j < 4; j++) {
            sum += W3[i][j] * h2[j];
        }
        out[i] = sum;   // no activation
    }
}

// ==================================================
// Uniform symmetric quantization (n bits)
// ==================================================
void quantize_nbits(
    const float* x,
    int* q,
    int size,
    int n_bits,
    float& scale
) {
    int Qmax = (1 << (n_bits - 1)) - 1;

    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = std::fabs(x[i]);
        if (a > max_val) max_val = a;
    }

    scale = (max_val > 0.0f) ? (Qmax / max_val) : 1.0f;

    for (int i = 0; i < size; i++) {
        q[i] = (int)std::round(x[i] * scale);
    }
}

// ==================================================
// Dequantization
// ==================================================
void dequantize_nbits(
    const int* q,
    float* x,
    int size,
    float scale
) {
    for (int i = 0; i < size; i++) {
        x[i] = q[i] / scale;
    }
}
