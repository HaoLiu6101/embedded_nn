#include <stdio.h>
#include "math_nn.h"

int main() {
    printf("Hello, World!\n");

    // create test case for tanh activation function
    float x = 0.5f;
    float y = tanh_act(x);

    // print the result
    printf("tanh(%f) = %f\n", x, y);
    return 0;
}