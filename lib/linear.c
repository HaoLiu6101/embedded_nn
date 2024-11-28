
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "linear.h"
#include "math_nn.h"

// Initialization functions
void init_linear_layer_config(LinearLayerConfig* config, int input_size, int output_size) {
    config->input_size = input_size;
    config->output_size = output_size;
}

void init_linear_layer_weights(LinearLayerWeights* weights, LinearLayerConfig* config) {
    int input_size = config->input_size;
    int output_size = config->output_size;

    weights->weights = (float*)calloc(input_size * output_size, sizeof(float));
    weights->bias = (float*)calloc(output_size, sizeof(float));
}

void init_linear_layer(LinearLayer* layer, int input_size, int output_size) {
    init_linear_layer_config(&layer->config, input_size, output_size);
    init_linear_layer_weights(&layer->weights, &layer->config);
}

void free_linear_layer_weights(LinearLayerWeights* weights) {
    free(weights->weights);
    free(weights->bias);
}

void free_linear_layer(LinearLayer* layer) {
    free_linear_layer_weights(&layer->weights);
}

// Forward function
void linear_layer_forward(LinearLayer* layer, float* input, float* output) {
    LinearLayerConfig* config = &layer->config;
    LinearLayerWeights* weights = &layer->weights;

    int input_size = config->input_size;
    int output_size = config->output_size;

    matmul(output, input, weights->weights, 1, input_size, output_size);
    add(output, output, weights->bias, output_size);
}