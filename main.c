#include <stdio.h>
#include <stdlib.h>
#include "gru.h"
#include "linear.h"

typedef struct {
    GRULayer gru_layers[3];
    LinearLayer output_layer;
} GRUModel;

void init_gru_model(GRUModel* model, int input_size, int hidden_size, int output_size) {
    init_gru_layer(&model->gru_layers[0], input_size, hidden_size);
    init_gru_layer(&model->gru_layers[1], hidden_size, hidden_size);
    init_gru_layer(&model->gru_layers[2], hidden_size, hidden_size);
    init_linear_layer(&model->output_layer, hidden_size, output_size);
}

void free_gru_model(GRUModel* model) {
    for (int i = 0; i < 3; i++) {
        free_gru_layer(&model->gru_layers[i]);
    }
    free_linear_layer(&model->output_layer);
}

int main() {
    // Initialize GRU model
    GRUModel model;
    init_gru_model(&model, 15, 64, 4);

    // Example input
    float input[15] = {0.0}; // Adjust the size according to input_size
    float h_prev[3][64] = {{0.0}}; // Adjust the size according to hidden_size
    float output[4] = {0.0}; // Adjust the size according to output_size

    // Forward pass through GRU layers
    gru_layer_forward(&model.gru_layers[0], input, h_prev[0]);
    gru_layer_forward(&model.gru_layers[1], model.gru_layers[0].state.hidden_state_buffer, h_prev[1]);
    gru_layer_forward(&model.gru_layers[2], model.gru_layers[1].state.hidden_state_buffer, h_prev[2]);

    // Forward pass through the output layer
    linear_layer_forward(&model.output_layer, model.gru_layers[2].state.hidden_state_buffer, output);

    // Print the output
    for (int i = 0; i < 4; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Free resources
    free_gru_model(&model);

    return 0;
}
