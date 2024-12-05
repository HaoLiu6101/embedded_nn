#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h> // Include the stdbool.h header for bool type
#include "lstm.h"
#include "math_nn.h"

void init_lstm_layer_config(LSTMLayerConfig* config, int input_size, int hidden_size) {
    config->input_size = input_size;
    config->hidden_size = hidden_size;
}

void init_lstm_layer_weights(LSTMLayerWeights* weights, LSTMLayerConfig* config) {
    int size = config->input_size * config->hidden_size;
    weights->W_ii = (float*)malloc(size * sizeof(float));
    weights->W_if = (float*)malloc(size * sizeof(float));
    weights->W_ig = (float*)malloc(size * sizeof(float));
    weights->W_io = (float*)malloc(size * sizeof(float));
    weights->W_hi = (float*)malloc(size * sizeof(float));
    weights->W_hf = (float*)malloc(size * sizeof(float));
    weights->W_hg = (float*)malloc(size * sizeof(float));
    weights->W_ho = (float*)malloc(size * sizeof(float));
    weights->b_ii = (float*)calloc(config->hidden_size, sizeof(float));
    weights->b_if = (float*)calloc(config->hidden_size, sizeof(float));
    weights->b_ig = (float*)calloc(config->hidden_size, sizeof(float));
    weights->b_io = (float*)calloc(config->hidden_size, sizeof(float));
    weights->b_hi = (float*)calloc(config->hidden_size, sizeof(float));
    weights->b_hf = (float*)calloc(config->hidden_size, sizeof(float));
    weights->b_hg = (float*)calloc(config->hidden_size, sizeof(float));
    weights->b_ho = (float*)calloc(config->hidden_size, sizeof(float));
}

void init_lstm_layer_run_state(LSTMLayerRunState* state, LSTMLayerConfig* config) {
    int hidden_size = config->hidden_size;
    state->input_buffer = (float*)malloc(config->input_size * sizeof(float));
    state->forget_gate_buffer = (float*)malloc(hidden_size * sizeof(float));
    state->input_gate_buffer = (float*)malloc(hidden_size * sizeof(float));
    state->output_gate_buffer = (float*)malloc(hidden_size * sizeof(float));
    state->input_node_buffer = (float*)malloc(config->input_size * sizeof(float)); // New buffer
    state->cell_state_buffer = (float*)malloc(hidden_size * sizeof(float));
    state->hidden_state_buffer = (float*)malloc(hidden_size * sizeof(float));
}

void init_lstm_layer(LSTMLayer* layer, int input_size, int hidden_size) {
    init_lstm_layer_config(&layer->config, input_size, hidden_size);
    init_lstm_layer_weights(&layer->weights, &layer->config);
    init_lstm_layer_run_state(&layer->state, &layer->config);
}

void free_lstm_layer_weights(LSTMLayerWeights* weights) {
    free(weights->W_ii);
    free(weights->W_if);
    free(weights->W_ig);
    free(weights->W_io);
    free(weights->W_hi);
    free(weights->W_hf);
    free(weights->W_hg);
    free(weights->W_ho);
    free(weights->b_ii);
    free(weights->b_if);
    free(weights->b_ig);
    free(weights->b_io);
    free(weights->b_hi);
    free(weights->b_hf);
    free(weights->b_hg);
    free(weights->b_ho);
}

void free_lstm_layer_run_state(LSTMLayerRunState* state) {
    free(state->input_buffer);
    free(state->forget_gate_buffer);
    free(state->input_gate_buffer);
    free(state->output_gate_buffer);
    free(state->input_node_buffer); // Free new buffer
    free(state->cell_state_buffer);
    free(state->hidden_state_buffer);
}

void free_lstm_layer(LSTMLayer* layer, bool free_weights) {
    free_lstm_layer_run_state(&layer->state);
    if (free_weights) {
        free_lstm_layer_weights(&layer->weights);
    }

}

void lstm_layer_forward(LSTMLayer* layer, float* input, float* h_prev, float* c_prev) {
    // get the config, weights and state
    LSTMLayerConfig* config = &layer->config;
    LSTMLayerWeights* weights = &layer->weights;
    LSTMLayerRunState* state = &layer->state;

    // get the input and hidden size
    int input_size = config->input_size;
    int hidden_size = config->hidden_size;

    // get the run state buffers
    float* input_buffer = state->input_buffer;
    float* forget_gate_buffer = state->forget_gate_buffer;
    float* input_gate_buffer = state->input_gate_buffer;
    float* output_gate_buffer = state->output_gate_buffer;
    float* input_node_buffer = state->input_node_buffer; // New buffer
    float* cell_state_buffer = state->cell_state_buffer;
    float* hidden_state_buffer = state->hidden_state_buffer;

    // map input into input buffer
    memcpy(input_buffer, input, input_size * sizeof(float));
    // get the weights and the bias 
    float* W_ii = weights->W_ii;
    float* W_if = weights->W_if;
    float* W_ig = weights->W_ig;
    float* W_io = weights->W_io;
    float* W_hi = weights->W_hi;
    float* W_hf = weights->W_hf;
    float* W_hg = weights->W_hg;
    float* W_ho = weights->W_ho;
    float* b_ii = weights->b_ii;
    float* b_if = weights->b_if;
    float* b_ig = weights->b_ig;
    float* b_io = weights->b_io;
    float* b_hi = weights->b_hi;
    float* b_hf = weights->b_hf;
    float* b_hg = weights->b_hg;
    float* b_ho = weights->b_ho;


    // Compute input gate: i_t = sigmoid(W_ii * x_t + W_hi * h_prev + b_ii + b_hi)
    matmul(input_gate_buffer, W_ii, input_buffer, 1, input_size, hidden_size);
    matmul(hidden_state_buffer, W_hi, h_prev, 1, hidden_size, hidden_size);
    add(input_gate_buffer, input_gate_buffer, hidden_state_buffer, hidden_size);
    add(input_gate_buffer, input_gate_buffer, b_ii, hidden_size);
    add(input_gate_buffer, input_gate_buffer, b_hi, hidden_size);
    sigmoid_act_vec(input_gate_buffer, input_gate_buffer, hidden_size);

    // Compute forget gate: f_t = sigmoid(W_if * x_t + W_hf * h_prev + b_if + b_hf)
    matmul(forget_gate_buffer, W_if, input_buffer, 1, input_size, hidden_size);
    matmul(hidden_state_buffer, W_hf, h_prev, 1, hidden_size, hidden_size);
    add(forget_gate_buffer, forget_gate_buffer, hidden_state_buffer, hidden_size);
    add(forget_gate_buffer, forget_gate_buffer, b_if, hidden_size);
    add(forget_gate_buffer, forget_gate_buffer, b_hf, hidden_size);
    sigmoid_act_vec(forget_gate_buffer, forget_gate_buffer, hidden_size);

    // Compute input node: g_t = tanh(W_ig * x_t + W_hg * h_prev + b_ig + b_hg)
    matmul(input_node_buffer, W_ig, input_buffer, 1, input_size, hidden_size);
    matmul(hidden_state_buffer, W_hg, h_prev, 1, hidden_size, hidden_size);
    add(input_node_buffer, input_node_buffer, hidden_state_buffer, hidden_size);
    add(input_node_buffer, input_node_buffer, b_ig, hidden_size);
    add(input_node_buffer, input_node_buffer, b_hg, hidden_size);
    tanh_act_vec(input_node_buffer, input_node_buffer, hidden_size);

    // Compute output gate: o_t = sigmoid(W_io * x_t + W_ho * h_prev + b_io + b_ho)
    matmul(output_gate_buffer, W_io, input_buffer, 1, input_size, hidden_size);
    matmul(hidden_state_buffer, W_ho, h_prev, 1, hidden_size, hidden_size);
    add(output_gate_buffer, output_gate_buffer, hidden_state_buffer, hidden_size);
    add(output_gate_buffer, output_gate_buffer, b_io, hidden_size);
    add(output_gate_buffer, output_gate_buffer, b_ho, hidden_size);
    sigmoid_act_vec(output_gate_buffer, output_gate_buffer, hidden_size);


    // Update cell state: c_t = f_t * c_prev + i_t * g_t
    mul(cell_state_buffer, forget_gate_buffer, c_prev, hidden_size);
    mul(input_node_buffer, input_gate_buffer, input_node_buffer, hidden_size);
    add(cell_state_buffer, cell_state_buffer, input_node_buffer, hidden_size);

    // Update hidden state: h_t = o_t * tanh(c_t)
    tanh_act_vec(cell_state_buffer, cell_state_buffer, hidden_size);
    mul(hidden_state_buffer, output_gate_buffer, cell_state_buffer, hidden_size);
}
