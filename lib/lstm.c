#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h> // Include the stdbool.h header for bool type
#include "lstm.h"
#include "math_nn.h"
// ...existing code...

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
    free(state->cell_state_buffer);
    free(state->hidden_state_buffer);
}

void free_lstm_layer(LSTMLayer* layer, bool free_weights) {
    free_lstm_layer_run_state(&layer->state);
    if (free_weights) {
        free_lstm_layer_weights(&layer->weights);
    }

}

void lstm_layer_forward(LSTMLayer* layer, float* input, float* h_prev) {
    int input_size = layer->config.input_size;
    int hidden_size = layer->config.hidden_size;

    // Compute input gate: i_t = sigmoid(W_ii * x_t + W_hi * h_prev + b_ii + b_hi)
    matmul(layer->state.input_buffer, layer->weights.W_ii, input, hidden_size, input_size, 1);
    matmul(layer->state.hidden_state_buffer, layer->weights.W_hi, h_prev, hidden_size, hidden_size, 1);
    add(layer->state.input_buffer, layer->state.input_buffer, layer->state.hidden_state_buffer, hidden_size);
    add(layer->state.input_buffer, layer->state.input_buffer, layer->weights.b_ii, hidden_size);
    add(layer->state.input_buffer, layer->state.input_buffer, layer->weights.b_hi, hidden_size);
    sigmoid_act_vec(layer->input_gate_buffer, layer->state.input_buffer, hidden_size);

    // Compute forget gate: f_t = sigmoid(W_if * x_t + W_hf * h_prev + b_if + b_hf)
    matmul(layer->state.forget_gate_buffer, layer->weights.W_if, input, hidden_size, input_size, 1);
    matmul(layer->state.hidden_state_buffer, layer->weights.W_hf, h_prev, hidden_size, hidden_size, 1);
    add(layer->state.forget_gate_buffer, layer->state.forget_gate_buffer, layer->state.hidden_state_buffer, hidden_size);
    add(layer->state.forget_gate_buffer, layer->state.forget_gate_buffer, layer->weights.b_if, hidden_size);
    add(layer->state.forget_gate_buffer, layer->state.forget_gate_buffer, layer->weights.b_hf, hidden_size);
    sigmoid_act_vec(layer->forget_gate_buffer, layer->state.forget_gate_buffer, hidden_size);

    // Compute cell gate: g_t = tanh(W_ig * x_t + W_hg * h_prev + b_ig + b_hg)
    matmul(layer->state.input_buffer, layer->weights.W_ig, input, hidden_size, input_size, 1);
    matmul(layer->state.hidden_state_buffer, layer->weights.W_hg, h_prev, hidden_size, hidden_size, 1);
    add(layer->state.input_buffer, layer->state.input_buffer, layer->state.hidden_state_buffer, hidden_size);
    add(layer->state.input_buffer, layer->state.input_buffer, layer->weights.b_ig, hidden_size);
    add(layer->state.input_buffer, layer->state.input_buffer, layer->weights.b_hg, hidden_size);
    tanh_act_vec(layer->input_buffer, layer->state.input_buffer, hidden_size);

    // Compute output gate: o_t = sigmoid(W_io * x_t + W_ho * h_prev + b_io + b_ho)
    matmul(layer->state.output_gate_buffer, layer->weights.W_io, input, hidden_size, input_size, 1);
    matmul(layer->state.hidden_state_buffer, layer->weights.W_ho, h_prev, hidden_size, hidden_size, 1);
    add(layer->state.output_gate_buffer, layer->state.output_gate_buffer, layer->state.hidden_state_buffer, hidden_size);
    add(layer->state.output_gate_buffer, layer->state.output_gate_buffer, layer->weights.b_io, hidden_size);
    add(layer->state.output_gate_buffer, layer->state.output_gate_buffer, layer->weights.b_ho, hidden_size);
    sigmoid_act_vec(layer->output_gate_buffer, layer->state.output_gate_buffer, hidden_size);

    // Update cell state: c_t = f_t * c_prev + i_t * g_t
    mul(layer->state.cell_state_buffer, layer->forget_gate_buffer, layer->state.cell_state_buffer, hidden_size);
    mul(layer->input_buffer, layer->input_gate_buffer, layer->input_buffer, hidden_size);
    add(layer->state.cell_state_buffer, layer->state.cell_state_buffer, layer->input_buffer, hidden_size);

    // Update hidden state: h_t = o_t * tanh(c_t)
    tanh_act_vec(layer->hidden_state_buffer, layer->state.cell_state_buffer, hidden_size);
    mul(h_prev, layer->output_gate_buffer, layer->hidden_state_buffer, hidden_size);
}