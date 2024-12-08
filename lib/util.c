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

MathStatus min_max_scaler(float* out, float* in, int size, 
                         float* feature_min, float* feature_max,
                         float scale_min, float scale_max) {
    // Validate input pointers
    if (out == NULL || in == NULL || feature_min == NULL || feature_max == NULL) {
        return MATH_NULL_POINTER;
    }
    
    // Validate size
    if (size <= 0) {
        return MATH_INVALID_DIM;
    }
    if (size > MAX_DIM) {
        return MATH_EXCEEDS_MAX_DIM;
    }

    // Validate range
    if (scale_max <= scale_min) {
        return MATH_INVALID_RANGE;
    }

    float scale_range = scale_max - scale_min;
    
    for (int i = 0; i < size; i++) {
        // Prevent division by zero
        if (feature_max[i] == feature_min[i]) {
            return MATH_OVERFLOW_RISK;
        }
        
        // Apply min-max scaling formula:
        // X_scaled = (X - X_min) / (X_max - X_min) * (scale_max - scale_min) + scale_min
        float normalized = (in[i] - feature_min[i]) / (feature_max[i] - feature_min[i]);
        out[i] = normalized * scale_range + scale_min;
    }
    
    return MATH_SUCCESS;
}

