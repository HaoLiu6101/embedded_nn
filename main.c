#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gru.h"
#include "linear.h"

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;
}GRUModelConfig;

typedef struct {
    GRUModelConfig config;
    GRULayer* gru_layers;
    LinearLayer output_layer;
} GRUModel;

void init_gru_model(GRUModel* model) {
    //allocate memory for GRU layers
    model->gru_layers = (GRULayer*)malloc(model->config.num_layers * sizeof(GRULayer));
    int input_size = model->config.input_size;
    for (int i = 0; i < model->config.num_layers; i++) {
        init_gru_layer(&model->gru_layers[i], input_size, model->config.hidden_size);
        input_size = model->config.hidden_size;
    }
    init_linear_layer(&model->output_layer, model->config.hidden_size, model->config.output_size);
}

void free_gru_model(GRUModel* model) {
    for (int i = 0; i < model->config.num_layers; i++) {
        free_gru_layer(&model->gru_layers[i]);
    }
    free_linear_layer(&model->output_layer);
    free(model);
}

int main() {
    // Initialize GRU model
    int input_size = 15;
    int hidden_size = 64;
    int output_size = 4;
    int num_layers = 3;
    GRUModelConfig config = {input_size, hidden_size, output_size, num_layers};
    GRUModel* model = (GRUModel*)malloc(sizeof(GRUModel));
    model->config = config;
    init_gru_model(model);

// Example input
    float* input = (float*)calloc(input_size, sizeof(float)); // Adjust the size according to input_size
    float* h_prev = (float*)calloc(num_layers * hidden_size, sizeof(float)); // Allocate 3 * 64 floats and initialize to 0
    float* output = (float*)calloc(output_size,sizeof(float)); // Adjust the size according to output_size
    float* inter_input = (float*)calloc(hidden_size, sizeof(float));


    // // Forward pass through GRU layers
    // gru_layer_forward(&model->gru_layers[0], input, h_prev);
    // gru_layer_forward(&model->gru_layers[1], model->gru_layers[0].state.hidden_state_buffer, h_prev+64);
    // gru_layer_forward(&model->gru_layers[2], model->gru_layers[1].state.hidden_state_buffer, h_prev+128);

    for (int i = 0; i < num_layers; i++) {
        if (i == 0) {
            inter_input = input;
        };
        gru_layer_forward(&model->gru_layers[i], inter_input, h_prev+i*hidden_size);
        memcpy(h_prev+i*hidden_size, model->gru_layers[i].state.hidden_state_buffer, model->gru_layers->config.hidden_size * sizeof(float));
        inter_input = model->gru_layers[i].state.hidden_state_buffer;
    }

    // Forward pass through the output layer
    linear_layer_forward(&model->output_layer, model->gru_layers[2].state.hidden_state_buffer, output);

    // // pass the output buffer in gru layers into next iteration
    // memcpy(h_prev, model->gru_layers[0].state.hidden_state_buffer, model->gru_layers->config.hidden_size * sizeof(float));
    // memcpy(h_prev+64, model->gru_layers[1].state.hidden_state_buffer, model->gru_layers->config.hidden_size * sizeof(float));
    // memcpy(h_prev+128, model->gru_layers[2].state.hidden_state_buffer, model->gru_layers->config.hidden_size * sizeof(float));




    // Print the output
    for (int i = 0; i < 4; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Free resources
    free_gru_model(model);
    free(input);
    free(h_prev);
    free(output);

    return 0;
}
