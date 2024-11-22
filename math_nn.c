#include "math_nn.h"
#include <math.h>

float sigmoid_act(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float tanh_act(float x) {
    return tanhf(x);
}

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

void add(float* out, float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

void mul(float* out, float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

void sigmoid_act_vec(float* out, float* x, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = sigmoid_act(x[i]);
    }
}

void tanh_act_vec(float* out, float* x, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = tanh_act(x[i]);
    }
}