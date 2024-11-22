#ifndef MATH_NN_H
#define MATH_NN_H

// Function to compute the sigmoid activation
float sigmoid_act(float x);

// Function to compute the tanh activation
float tanh_act(float x);

// Function to perform matrix multiplication
// float out[m][p] = a[m][n] * b[n][p]
void matmul(float* out, float* a, float* b, int m, int n, int p);

// Function to perform element-wise addition
//float out[size] = a[size] + b[size]
void add(float* out, float* a, float* b, int size);

// Function to perform element-wise multiplication
//float out[size] = a[size] * b[size]
void mul(float* out, float* a, float* b, int size);

// Function to perform element-wise sigmoid activation
void sigmoid_act_vec(float* out, float* x, int size);

// Function to perform element-wise tanh activation
void tanh_act_vec(float* out, float* x, int size);

#endif // MATH_NN_H