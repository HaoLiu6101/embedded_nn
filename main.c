#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math_nn.h"
#include "gru.h"

int main() {
    printf("Hello, World!\n");

    // init a GRU model
    GRUModel model;
    init_gru_model(&model);

    // create a test input
    float input[15] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7};

    // run the GRU model
    gru_forward(&model, input);

    // print the output
    for (int i = 0; i < model.config.output_size; i++) {
        printf("output[%d] = %f\n", i, model.state.output_buffer[i]);
    }

    // free the GRU model
    free_gru_model(&model);
    return 0;
}
