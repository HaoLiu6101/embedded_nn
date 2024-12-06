#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h> // Include the stdbool.h header for bool type
#include "gru.h"
#include "math_nn.h"

// Initialization functions
void init_gru_layer_config(GRULayerConfig* config, int input_dim, int input_size, int hidden_size) {
    config->input_dim = input_dim;
    config->input_size = input_size;
    config->hidden_size = hidden_size;
}

void init_gru_layer_weights(GRULayerWeights* weights, GRULayerConfig* config) {
    int input_dim = config->input_dim;
    int input_size = config->input_size;
    int hidden_size = config->hidden_size;

    weights->W_ir = (float*)calloc(input_size * hidden_size, sizeof(float));
    weights->W_iz = (float*)calloc(input_size * hidden_size, sizeof(float));
    weights->W_in = (float*)calloc(input_size * hidden_size, sizeof(float));
    weights->W_hr = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    weights->W_hz = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    weights->W_hn = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    weights->b_ir = (float*)calloc(input_dim * hidden_size, sizeof(float));
    weights->b_iz = (float*)calloc(input_dim * hidden_size, sizeof(float));
    weights->b_in = (float*)calloc(input_dim * hidden_size, sizeof(float));
    weights->b_hr = (float*)calloc(input_dim * hidden_size, sizeof(float));
    weights->b_hz = (float*)calloc(input_dim * hidden_size, sizeof(float));
    weights->b_hn = (float*)calloc(input_dim * hidden_size, sizeof(float));
}

void init_gru_layer_run_state(GRULayerRunState* state, GRULayerConfig* config) {
    int input_dim = config->input_dim;
    int hidden_size = config->hidden_size;
    int input_size = config->input_size;

    state->hidden_state_buffer = (float*)calloc(input_dim * hidden_size, sizeof(float));
    state->input_buffer = (float*)calloc(input_dim * input_size, sizeof(float));
    state->reset_gate_buffer = (float*)calloc(input_dim * hidden_size, sizeof(float));
    state->update_gate_buffer = (float*)calloc(input_dim * hidden_size, sizeof(float));
    state->candidate_hidden_state_buffer = (float*)calloc(input_dim * hidden_size, sizeof(float));
    // Removed allocation of hidden_cell_temp
}

void init_gru_layer(GRULayer* layer, int input_dim, int input_size, int hidden_size) {
    init_gru_layer_config(&layer->config, input_dim, input_size, hidden_size);
    init_gru_layer_weights(&layer->weights, &layer->config);
    init_gru_layer_run_state(&layer->state, &layer->config);
}

void free_gru_layer_weights(GRULayerWeights* weights) {
    free(weights->W_ir);
    free(weights->W_iz);
    free(weights->W_in);
    free(weights->W_hr);
    free(weights->W_hz);
    free(weights->W_hn);
    free(weights->b_ir);
    free(weights->b_iz);
    free(weights->b_in);
    free(weights->b_hr);
    free(weights->b_hz);
    free(weights->b_hn);
}

void free_gru_layer_run_state(GRULayerRunState* state) {
    free(state->hidden_state_buffer);
    free(state->input_buffer);
    free(state->reset_gate_buffer);
    free(state->update_gate_buffer);
    free(state->candidate_hidden_state_buffer);
    // Removed freeing of hidden_cell_temp
}

void free_gru_layer(GRULayer* layer, bool free_weights) {
    printf("Freeing GRU layer...\n");
    free_gru_layer_run_state(&layer->state);

    if (free_weights) {
        free_gru_layer_weights(&layer->weights);
    }
}

// Forward function
void gru_layer_forward(GRULayer* layer, float* input, float* h_prev) {
    GRULayerConfig* config = &layer->config;
    GRULayerWeights* weights = &layer->weights;
    GRULayerRunState* state = &layer->state;

    int input_dim = config->input_dim;
    int input_size = config->input_size;
    int hidden_size = config->hidden_size;

    float* input_buffer = state->input_buffer;
    float* hidden_state_buffer = state->hidden_state_buffer;
    float* reset_gate_buffer = state->reset_gate_buffer;
    float* update_gate_buffer = state->update_gate_buffer;
    float* candidate_hidden_state_buffer = state->candidate_hidden_state_buffer;
    // Removed declaration of hidden_cell_temp

    memcpy(input_buffer, input, input_dim * input_size * sizeof(float));

    float* W_ir = weights->W_ir;
    float* W_iz = weights->W_iz;
    float* W_in = weights->W_in;
    float* W_hr = weights->W_hr;
    float* W_hz = weights->W_hz;
    float* W_hn = weights->W_hn;
    float* b_ir = weights->b_ir;
    float* b_iz = weights->b_iz;
    float* b_in = weights->b_in;
    float* b_hr = weights->b_hr;
    float* b_hz = weights->b_hz;
    float* b_hn = weights->b_hn;
    

    matmul(reset_gate_buffer, input_buffer, W_ir, input_dim, input_size, hidden_size);
    add(reset_gate_buffer, reset_gate_buffer, b_ir, input_dim * hidden_size);
    matmul(reset_gate_buffer, h_prev, W_hr, input_dim, hidden_size, hidden_size);
    add(reset_gate_buffer, reset_gate_buffer, b_hr, input_dim * hidden_size);
    sigmoid_act_vec(reset_gate_buffer, reset_gate_buffer, input_dim * hidden_size);

    matmul(update_gate_buffer, input_buffer, W_iz, input_dim, input_size, hidden_size);
    add(update_gate_buffer, update_gate_buffer, b_iz, input_dim * hidden_size);
    matmul(update_gate_buffer, h_prev, W_hz, input_dim, hidden_size, hidden_size);
    add(update_gate_buffer, update_gate_buffer, b_hz, input_dim * hidden_size);
    sigmoid_act_vec(update_gate_buffer, update_gate_buffer, input_dim * hidden_size);

    matmul(candidate_hidden_state_buffer, input_buffer, W_in, input_dim, input_size, hidden_size);
    add(candidate_hidden_state_buffer, candidate_hidden_state_buffer, b_in, input_dim * hidden_size);

    float r_h_prev[input_dim * hidden_size];
    // for (int i = 0; i < input_dim * hidden_size; i++) {
    //     r_h_prev[i] = reset_gate_buffer[i] * h_prev[i];
    // }
    mul(r_h_prev, reset_gate_buffer, h_prev, input_dim * hidden_size);

    matmul(candidate_hidden_state_buffer, r_h_prev, W_hn, input_dim, hidden_size, hidden_size);
    add(candidate_hidden_state_buffer, candidate_hidden_state_buffer, b_hn, input_dim * hidden_size);
    tanh_act_vec(candidate_hidden_state_buffer, candidate_hidden_state_buffer, input_dim * hidden_size);

    for (int i = 0; i < input_dim * hidden_size; i++) {
        hidden_state_buffer[i] = update_gate_buffer[i] * h_prev[i] + (1 - update_gate_buffer[i]) * candidate_hidden_state_buffer[i];
    }

    //below is removed for memory efficiency 
    //memcpy(hidden_state_buffer, hidden_cell_temp, input_dim * hidden_size * sizeof(float));
}