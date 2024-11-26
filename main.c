#include <stdio.h>
#include <stdlib.h>
#include "gru.h"

int main() {
    // Initialize GRU layer
    GRULayer gru_layer;
    init_gru_layer(&gru_layer);

    // Example input
    float input[15] = {0.0}; // Adjust the size according to input_size
    float h_prev[64] = {0.0}; // Adjust the size according to hidden_size

    // Forward pass
    gru_layer_forward(&gru_layer, input, h_prev);

    // Print the output hidden state
    for (int i = 0; i < gru_layer.config.hidden_size; i++) {
        printf("%f ", gru_layer.state.hidden_state_buffer[i]);
    }
    printf("\n");

    // Free resources
    free_gru_layer(&gru_layer);

    return 0;
}
