#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gru.h"
#include "math_nn.h"

// Struct definitions
// ...existing code...

// Initialization functions
void init_gru_config(GRUConfig* config) {
    config->input_size = 15;
    config->hidden_size = 64;
    config->output_size = 4;
    config->num_layers = 3;
}

void init_gru_weights(GRUWeights* weights, GRUConfig* config) {
    int input_size = config->input_size;
    int num_layers = config->num_layers;
    int hidden_size = config->hidden_size;
    int output_size = config->output_size;

    // initialize the number of weights for each layer
    int num_W_ir = 0, num_W_iz = 0, num_W_in = 0;   //weight input to hidden
    int num_b_ir = 0, num_b_iz = 0, num_b_in = 0;   //bias input to hidden
    int num_W_hr = 0, num_W_hz = 0, num_W_hn = 0;   //weight hidden to hidden
    int num_b_hr = 0, num_b_hz = 0, num_b_hn = 0;   //bias hidden to hidden

    // loop through the number of layers and calculate the total number of weights
    for(int l = 0; l < num_layers; l++){
        int cell_size = (l == 0) ? input_size : hidden_size;

        num_W_ir += cell_size * hidden_size;
        num_W_iz += cell_size * hidden_size;
        num_W_in += cell_size * hidden_size;
        num_W_hr += hidden_size * hidden_size;
        num_W_hz += hidden_size * hidden_size;
        num_W_hn += hidden_size * hidden_size;
        num_b_ir += hidden_size;
        num_b_iz += hidden_size;
        num_b_in += hidden_size;
        num_b_hr += hidden_size;
        num_b_hz += hidden_size;
        num_b_hn += hidden_size;
    }

    weights->W_ir = (float*)calloc(num_W_ir, sizeof(float));
    weights->W_iz = (float*)calloc(num_W_iz, sizeof(float));
    weights->W_in = (float*)calloc(num_W_in, sizeof(float));
    weights->W_hr = (float*)calloc(num_W_hr, sizeof(float));
    weights->W_hz = (float*)calloc(num_W_hz, sizeof(float));
    weights->W_hn = (float*)calloc(num_W_hn, sizeof(float));
    weights->b_ir = (float*)calloc(num_b_ir, sizeof(float));
    weights->b_iz = (float*)calloc(num_b_iz, sizeof(float));
    weights->b_in = (float*)calloc(num_b_in, sizeof(float));
    weights->b_hr = (float*)calloc(num_b_hr, sizeof(float));
    weights->b_hz = (float*)calloc(num_b_hz, sizeof(float));
    weights->b_hn = (float*)calloc(num_b_hn, sizeof(float));
    weights->W_out = (float*)calloc(output_size * hidden_size, sizeof(float));
    weights->b_out = (float*)calloc(output_size, sizeof(float));
}

void init_gru_run_state(GRURunState* state, GRUConfig* config) {
    int num_layers = config->num_layers;
    int hidden_size = config->hidden_size;
    int output_size = config->output_size;
    int input_size = config->input_size;

    // Set the input size to the maximum of the input and hidden sizes
    input_size = (hidden_size > input_size) ? hidden_size : input_size;

    state->hidden_state_buffer = (float*)calloc(num_layers * hidden_size, sizeof(float)); 
    state->input_buffer = (float*)calloc(input_size, sizeof(float)); 
    state->output_buffer = (float*)calloc(output_size, sizeof(float));
    state->reset_gate_buffer = (float*)calloc(hidden_size, sizeof(float));
    state->update_gate_buffer = (float*)calloc(hidden_size, sizeof(float));
    state->candidate_hidden_state_buffer = (float*)calloc(hidden_size, sizeof(float));
    state->hidden_cell_temp = (float*)calloc(hidden_size, sizeof(float));
}

void init_gru_model(GRUModel* model) {
    init_gru_config(&model->config);
    init_gru_weights(&model->weights, &model->config);
    init_gru_run_state(&model->state, &model->config);
}

void free_gru_weights(GRUWeights* weights) {
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
    free(weights->W_out);
    free(weights->b_out);
}

void free_gru_run_state(GRURunState* state) {
    free(state->hidden_state_buffer);
    free(state->input_buffer);
    free(state->output_buffer);
    free(state->reset_gate_buffer);
    free(state->update_gate_buffer);
    free(state->candidate_hidden_state_buffer);
    free(state->hidden_cell_temp);
}


void free_gru_model(GRUModel* model) {
    free_gru_weights(&model->weights);
    free_gru_run_state(&model->state);
}

// Forward function
void gru_forward(GRUModel* model, float* input) {
    GRUConfig* config = &model->config;
    GRUWeights* weights = &model->weights;
    GRURunState* state = &model->state;

    int input_size = config->input_size;
    int hidden_size = config->hidden_size;
    int output_size = config->output_size;
    int num_layers = config->num_layers;

    float* input_buffer = state->input_buffer;
    float* hidden_state_buffer = state->hidden_state_buffer;
    float* output_buffer = state->output_buffer;
    float* reset_gate_buffer = state->reset_gate_buffer;
    float* update_gate_buffer = state->update_gate_buffer;
    float* candidate_hidden_state_buffer = state->candidate_hidden_state_buffer;
    float* hidden_cell_temp = state->hidden_cell_temp;

    memcpy(input_buffer, input, input_size * sizeof(float));

    for (int l = 0; l < num_layers; l++) {
        int cell_size = (l == 0) ? input_size : hidden_size;
        float* W_ir = weights->W_ir + l * hidden_size * hidden_size;
        float* W_iz = weights->W_iz + l * hidden_size * hidden_size;
        float* W_in = weights->W_in + l * hidden_size * hidden_size;
        float* W_hr = weights->W_hr + l * hidden_size * hidden_size;
        float* W_hz = weights->W_hz + l * hidden_size * hidden_size;
        float* W_hn = weights->W_hn + l * hidden_size * hidden_size;
        float* b_ir = weights->b_ir + l * hidden_size;
        float* b_iz = weights->b_iz + l * hidden_size;
        float* b_in = weights->b_in + l * hidden_size;
        float* b_hr = weights->b_hr + l * hidden_size;
        float* b_hz = weights->b_hz + l * hidden_size;
        float* b_hn = weights->b_hn + l * hidden_size;
        float* h_prev = hidden_state_buffer + l * hidden_size;

        matmul(reset_gate_buffer, input_buffer, W_ir, 1, cell_size, hidden_size);
        add(reset_gate_buffer, reset_gate_buffer, b_ir, hidden_size);
        matmul(reset_gate_buffer, h_prev, W_hr, 1, hidden_size, hidden_size);
        add(reset_gate_buffer, reset_gate_buffer, b_hr, hidden_size);
        sigmoid_act_vec(reset_gate_buffer, reset_gate_buffer, hidden_size);

        matmul(update_gate_buffer, input_buffer, W_iz, 1, cell_size, hidden_size);
        add(update_gate_buffer, update_gate_buffer, b_iz, hidden_size);
        matmul(update_gate_buffer, h_prev, W_hz, 1, hidden_size, hidden_size);
        add(update_gate_buffer, update_gate_buffer, b_hz, hidden_size);
        sigmoid_act_vec(update_gate_buffer, update_gate_buffer, hidden_size);

        matmul(candidate_hidden_state_buffer, input_buffer, W_in, 1, cell_size, hidden_size);
        add(candidate_hidden_state_buffer, candidate_hidden_state_buffer, b_in, hidden_size);

        float r_h_prev[hidden_size];
        for (int i = 0; i < hidden_size; i++) {
            r_h_prev[i] = reset_gate_buffer[i] * h_prev[i];
        }

        matmul(candidate_hidden_state_buffer, r_h_prev, W_hn, 1, hidden_size, hidden_size);
        add(candidate_hidden_state_buffer, candidate_hidden_state_buffer, b_hn, hidden_size);
        tanh_act_vec(candidate_hidden_state_buffer, candidate_hidden_state_buffer, hidden_size);

        for (int i = 0; i < hidden_size; i++) {
            hidden_cell_temp[i] = update_gate_buffer[i] * h_prev[i] + (1 - update_gate_buffer[i]) * candidate_hidden_state_buffer[i];
        }

        memcpy(input_buffer, hidden_cell_temp, hidden_size * sizeof(float));
        memcpy(hidden_state_buffer + l * hidden_size, hidden_cell_temp, hidden_size * sizeof(float));
    }

    matmul(output_buffer, input_buffer, weights->W_out, 1, hidden_size, output_size);
    add(output_buffer, output_buffer, weights->b_out, output_size);
}