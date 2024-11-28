
#include <stdio.h>
#include <assert.h>
#include "math_nn.h"

void test_sigmoid_act() {
    float x = 0.0f;
    float y = sigmoid_act(x);
    assert(y == 0.5f);
    printf("sigmoid_act(0.0) = %f\n", y);
}

void test_tanh_act() {
    float x = 0.5f;
    float y = tanh_act(x);
    printf("tanh_act(0.5) = %f\n", y);
}

void test_matmul() {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {7, 8, 9, 10, 11, 12};
    float out[4];
    matmul(out, a, b, 2, 3, 2);
    printf("matmul result: %f %f %f %f\n", out[0], out[1], out[2], out[3]);
}

void test_add() {
    float a[3] = {1, 2, 3};
    float b[3] = {4, 5, 6};
    float out[3];
    add(out, a, b, 3);
    printf("add result: %f %f %f\n", out[0], out[1], out[2]);
}

void test_mul() {
    float a[3] = {1, 2, 3};
    float b[3] = {4, 5, 6};
    float out[3];
    mul(out, a, b, 3);
    printf("mul result: %f %f %f\n", out[0], out[1], out[2]);
}

void test_sigmoid_act_vec() {
    float x[3] = {0.0f, 1.0f, -1.0f};
    float out[3];
    sigmoid_act_vec(out, x, 3);
    printf("sigmoid_act_vec result: %f %f %f\n", out[0], out[1], out[2]);
}

void test_tanh_act_vec() {
    float x[3] = {0.0f, 1.0f, -1.0f};
    float out[3];
    tanh_act_vec(out, x, 3);
    printf("tanh_act_vec result: %f %f %f\n", out[0], out[1], out[2]);
}

int main() {
    test_sigmoid_act();
    test_tanh_act();
    test_matmul();
    test_add();
    test_mul();
    test_sigmoid_act_vec();
    test_tanh_act_vec();
    printf("All tests passed!\n");
    return 0;
}