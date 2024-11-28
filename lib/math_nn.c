#include <math.h>
#include <stddef.h> // for NULL
#include <float.h>  // for FLT_MAX
#include <limits.h> // for FLT_MAX
#include "math_nn.h"


// implement the sigmoid activation function
float sigmoid_act(float x) {
    return 1.0f / (1.0f + expf(-x));
}


// implement the tanh activation function
float tanh_act(float x) {
    return tanhf(x);
}


// Implement the matrix multiplication activation function
// out[m, p] = a[m, n] * b[n, p]
// out[batch, out_dim] = a[batch, in_dim] * b[in_dim, out_dim]
MathStatus matmul(float* out, float* a, float* b, int m, int n, int p) {
    //check for null pointers
    if (out == NULL || a == NULL || b == NULL) {
        return MATH_NULL_POINTER;
    }

    // check for valid dimensions
    if (m <= 0 || n <= 0 || p <= 0 || m > MAX_DIM || n > MAX_DIM || p > MAX_DIM
    ) {
        return MATH_INVALID_DIM;
    }

    // check for potential overflow in the calculations
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            out[i * p + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                float product = a[i * n + k] * b[k * p + j];
                if (product > FLT_MAX - out[i * p + j]) {
                    return MATH_OVERFLOW_RISK;
                }
                out[i * p + j] += product;
            }
        }
    }
    return MATH_SUCCESS;
}


// implement add function at vector level 
// out[size] = a[size] + b[size]
MathStatus add(float* out, float* a, float* b, int size) {
    if (size > MAX_DIM) {
        return MATH_INVALID_DIM; // Exceeds maximum iteration limit
    }
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
    return MATH_SUCCESS;
}


// implement the element wise multiplication activation function
//out[size] = a[size] * b[size]
//          =[a1*b1 a2*b2 a3*b3 ...]
MathStatus mul(float* out, float* a, float* b, int size) {
    if (size > MAX_DIM) {
        return MATH_INVALID_DIM; // Exceeds maximum iteration limit
    }
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
    return MATH_SUCCESS;
}


// implement the sigmoid activation function at vector level 
MathStatus sigmoid_act_vec(float* out, float* x, int size) {
    if (size > MAX_DIM) {
        return MATH_INVALID_DIM; // Exceeds maximum iteration limit
    }
    for (int i = 0; i < size; i++) {
        out[i] = sigmoid_act(x[i]);
    }
    return MATH_SUCCESS;
}

// implement the tanh activation function at vector level
MathStatus tanh_act_vec(float* out, float* x, int size) {
    if (size > MAX_DIM) {
        return MATH_INVALID_DIM; // Exceeds maximum iteration limit
    }
    for (int i = 0; i < size; i++) {
        out[i] = tanh_act(x[i]);
    }
    return MATH_SUCCESS;
}