#include <stdio.h>
#include "math_nn.h"

MathStatus standard_scaler(float* out, float* in, int size, float mean, float std) {
    // Validate input pointers
    if (out == NULL || in == NULL) {
        return MATH_NULL_POINTER;
    }
    // Validate size
    if (size <= 0) {
        return MATH_INVALID_DIM;
    }
    if (size > MAX_DIM) {
        return MATH_EXCEEDS_MAX_DIM;
    }
    // Prevent division by zero
    if (std == 0.0f) {
        return MATH_OVERFLOW_RISK;
    }
    for (int i = 0; i < size; i++) {
        out[i] = (in[i] - mean) / std;
    }
    return MATH_SUCCESS;
}