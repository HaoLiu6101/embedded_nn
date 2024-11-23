#include "math_nn.h"
#include <math.h>

// implement the sigmoid activation function
float sigmoid_act(float x) {
    return 1.0f / (1.0f + expf(-x));
}


// implement the tanh activation function
float tanh_act(float x) {
    return tanhf(x);
}


// Implement the matrix multiplication activation function
// This function takes in a vector and returns the softmax activation of
// the vector.
// out[m, p] = a[m, n] * b[n, p]
void matmul(float* out, float* a, float* b, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            out[i * p + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                out[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}


// implement add function at vector level 
// out[size] = a[size] + b[size]
void add(float* out, float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}


// implement the element wise multiplication activation function
//out[size] = a[size] * b[size]
//          =[a1*b1 a2*b2 a3*b3 ...]
void mul(float* out, float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}


// implement the sigmoid activation function at vector level 
void sigmoid_act_vec(float* out, float* x, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = sigmoid_act(x[i]);
    }
}

// implement the tanh activation function at vector level
void tanh_act_vec(float* out, float* x, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = tanh_act(x[i]);
    }
}