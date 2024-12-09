#ifndef MATH_NN_H
#define MATH_NN_H

#define MAX_DIM 1024

// define math status struct 
typedef enum {
    MATH_SUCCESS = 0,
    MATH_NULL_POINTER = -1,
    MATH_INVALID_DIM = -2,
    MATH_OVERFLOW_RISK = -3,
    MATH_EXCEEDS_MAX_DIM = -4,
    MATH_INVALID_RANGE = -5,
} MathStatus;

// Function to compute the sigmoid activation
float sigmoid_act(float x);

// Function to compute the tanh activation
float tanh_act(float x);

// Function to perform matrix multiplication
// float out[m][p] = a[m][n] * b[n][p]
MathStatus matmul(float* out, float* a, float* b, int m, int n, int p);

// Function to perform element-wise addition
//float out[size] = a[size] + b[size]
MathStatus add(float* out, float* a, float* b, int size);

// Function to perform element-wise multiplication
//float out[size] = a[size] * b[size]
MathStatus mul(float* out, float* a, float* b, int size);

// Function to perform element-wise sigmoid activation
MathStatus sigmoid_act_vec(float* out, float* x, int size);

// Function to perform element-wise tanh activation
MathStatus tanh_act_vec(float* out, float* x, int size);

// Function to perform RMS normalization
MathStatus rms_norm(float* out, float* x, int size);

// Function to perform softmax
MathStatus softmax(float* out, float* x, int size);

#endif // MATH_NN_H