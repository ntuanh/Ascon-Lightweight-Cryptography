#ifndef NN_WEIGHTS_H
#define NN_WEIGHTS_H

#include <stdint.h>

// input size = 6
#define INPUT_SIZE 6

// Layer 0: 6 → 16
extern const float W0[16][INPUT_SIZE];
extern const float b0[16];

// Layer 1: 16 → 8
extern const float W1[8][16];
extern const float b1[8];

// Layer 2: 8 → 4
extern const float W2[4][8];
extern const float b2[4];

// Layer 3: 4 → 2
extern const float W3[2][4];
extern const float b3[2];

#endif

