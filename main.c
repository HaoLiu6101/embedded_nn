#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math_nn.h"


//GRU model configuration 
typedef struct {
    int input_size;      // Input size
    int hidden_size;     // Hidden size
    int output_size;     // Output size
    int num_layers;      // Number of GRU layers
} GRUConfig;

// GRU model weights
typedef struct {
    float* W_ir;         // Input to reset gate weights (num_layers, hidden_size, input_size)
    float* W_iz;         // Input to update gate weights (num_layers, hidden_size, input_size)
    float* W_in;         // Input to candidate hidden gate weights (num_layers, hidden_size, input_size)
    float* W_hr;         // Hidden to reset gate weights (num_layers, hidden_size, hidden_size)
    float* W_hz;         // Hidden to update gate weights (num_layers, hidden_size, hidden_size)
    float* W_hn;         // Hidden to candidate hidden gate weights (num_layers, hidden_size, hidden_size)
    float* b_ir;         // Input to reset gate biases (num_layers, hidden_size)
    float* b_iz;         // Input to update gate biases (num_layers, hidden_size)
    float* b_in;         // Input to candidate hidden gate biases (num_layers, hidden_size)
    float* b_hr;         // Hidden to reset gate biases (num_layers, hidden_size)
    float* b_hz;         // Hidden to update gate biases (num_layers, hidden_size)
    float* b_hn;         // Hidden to candidate hidden gate biases (num_layers, hidden_size)
    float* W_out;        // Output weights (output_size, hidden_size)
    float* b_out;        // Output biases (output_size)
} GRUWeights;

// GRU model run state
typedef struct {
    float* hidden_state_buffer;            // Hidden state (num_layers * hidden_size)
    float* input_buffer;            // Input buffer
    float* output_buffer;            // Output buffer (output_size)
    float* reset_gate_buffer;            // Reset gate buffer (hidden_size)
    float* update_gate_buffer;            // Update gate buffer (hidden_size)
    float* candidate_hidden_state_buffer;            // New gate buffer (hidden_size)
} GRURunState;

/// GRU model
typedef struct {
    GRUConfig config;    // GRU model configuration
    GRUWeights weights;  // GRU model weights
    GRURunState state;   // GRU model run state
} GRUModel;


void init_gru_config(GRUConfig* config) {
    config->input_size = 15;    // Set the input size
    config->hidden_size = 64;   // Set the hidden layer size
    config->output_size = 4;    // Set the output size
    config->num_layers = 3;     // Set the number of GRU layers
}

void init_gru_weights(GRUWeights* weights, GRUConfig* config) {
    // initialize the weights of GRU
    // allocate memory for all the weights 

    int num_layers = config->num_layers;
    int hidden_size = config->hidden_size;
    // int input_size = config->input_size;
    int output_size = config->output_size;
    weights->W_ir = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));    //allocate more memory
    weights->W_iz = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));    //allocate more memory
    weights->W_in = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));    //allocate more memory
    weights->W_hr = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));
    weights->W_hz = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));
    weights->W_hn = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));
    weights->b_ir = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights->b_iz = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights->b_in = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights->b_hr = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights->b_hz = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights->b_hn = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights->W_out = (float*)calloc(output_size * hidden_size, sizeof(float));
    weights->b_out = (float*)calloc(output_size, sizeof(float));
}

// Function to initialize GRU model run state
void init_gru_run_state(GRURunState* state, GRUConfig* config) {
    int num_layers = config->num_layers;
    int hidden_size = config->hidden_size;
    //int input_size = config->input_size;    //considering hidden size is greated than input size, allocate more memory
    int output_size = config->output_size;

    state->hidden_state_buffer = (float*)calloc(num_layers * hidden_size, sizeof(float));
    state->input_buffer = (float*)calloc(hidden_size, sizeof(float));                               //allocate more memory
    state->output_buffer = (float*)calloc(output_size, sizeof(float));
    state->reset_gate_buffer = (float*)calloc(hidden_size, sizeof(float));
    state->update_gate_buffer = (float*)calloc(hidden_size, sizeof(float));
    state->candidate_hidden_state_buffer = (float*)calloc(hidden_size, sizeof(float));
}

// Function to free GRU model weights
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

// Function to free GRU model run state
void free_gru_run_state(GRURunState* state) {
    free(state->hidden_state_buffer);
    free(state->input_buffer);
    free(state->output_buffer);
    free(state->reset_gate_buffer);
    free(state->update_gate_buffer);
    free(state->candidate_hidden_state_buffer);
}

void init_gru_model(GRUModel* model) {
    init_gru_config(&model->config);
    init_gru_weights(&model->weights, &model->config);
    init_gru_run_state(&model->state, &model->config);
}

// Function to free GRU model
void free_gru_model(GRUModel* model) {
    free_gru_weights(&model->weights);
    free_gru_run_state(&model->state);
}

void gru_forward(GRUModel* model, float* input) {
    GRUConfig* config = &model->config;
    GRUWeights* weights = &model->weights;
    GRURunState* state = &model->state;

    int input_size = config->input_size;
    int hidden_size = config->hidden_size;
    int output_size = config->output_size;
    int num_layers = config->num_layers;

    float* input_buffer = state->input_buffer;  // bear in mind, input buffer size is hidden size
    float* hidden_state_buffer = state->hidden_state_buffer;
    float* output_buffer = state->output_buffer;
    float* reset_gate_buffer = state->reset_gate_buffer;
    float* update_gate_buffer = state->update_gate_buffer;
    float* candidate_hidden_state_buffer = state->candidate_hidden_state_buffer;

    // Copy input to input buffer
    memcpy(input_buffer, input, input_size * sizeof(float)); // for the first layer, input buffer is the input

    // loop over layers
    for (int l = 0; l < num_layers; l++) {
        // define a cell size for each layer
        int cell_size = hidden_size;

        if (l == 0) {
            cell_size = input_size;
        }

        // get the memory address of weights
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

        float* h_prev = hidden_state_buffer + l * hidden_size;   // this is the hidden state of the previous layer
        float* h_next = hidden_state_buffer + (l + 1) * hidden_size;  // this is the hidden state of this layer, used in two places, one is as cell input for next cell, one is h_prev of next layer

        // rest gate 
        matmul(reset_gate_buffer, input_buffer, W_ir, 1, cell_size, hidden_size);
        add(reset_gate_buffer, reset_gate_buffer, b_ir, hidden_size);
        matmul(reset_gate_buffer, h_prev, W_hr, 1, hidden_size, hidden_size);
        add(reset_gate_buffer, reset_gate_buffer, b_hr, hidden_size);
        sigmoid_act_vec(reset_gate_buffer, reset_gate_buffer, hidden_size);

        // update gate
        matmul(update_gate_buffer, input_buffer, W_iz, 1, cell_size, hidden_size);
        add(update_gate_buffer, update_gate_buffer, b_iz, hidden_size);
        matmul(update_gate_buffer, h_prev, W_hz, 1, hidden_size, hidden_size);
        add(update_gate_buffer, update_gate_buffer, b_hz, hidden_size);
        sigmoid_act_vec(update_gate_buffer, update_gate_buffer, hidden_size);

        // candidate hidden state
        matmul(candidate_hidden_state_buffer, input_buffer, W_in, 1, cell_size, hidden_size);
        add(candidate_hidden_state_buffer, candidate_hidden_state_buffer, b_in, hidden_size);

        //hadamard product between rest gate and h_prev
        float r_h_prev[hidden_size];
        for (int i = 0; i < hidden_size; i++) {
            r_h_prev[i] = reset_gate_buffer[i] * h_prev[i];
        }

        matmul(candidate_hidden_state_buffer, r_h_prev, W_hn, 1, hidden_size, hidden_size);
        add(candidate_hidden_state_buffer, candidate_hidden_state_buffer, b_hn, hidden_size);
        tanh_act_vec(candidate_hidden_state_buffer, candidate_hidden_state_buffer, hidden_size);

        // update hidden state
        for (int i = 0; i < hidden_size; i++) {
            h_next[i] = update_gate_buffer[i] * h_prev[i] + (1 - update_gate_buffer[i]) * candidate_hidden_state_buffer[i];
        }

        // copy hidden state to output buffer
        memcpy(input_buffer, h_next, hidden_size * sizeof(float));

        // copy hidden state in h_next to h_prev for the next layer
        memcpy(hidden_state_buffer + l * hidden_size, h_next, hidden_size * sizeof(float));

    }
    //compute output
    // output = W_out * hidden_state + b_out, here input buffer = hidden state because of the last loop
    matmul(output_buffer, input_buffer, weights->W_out, 1, hidden_size, output_size);
    add(output_buffer, output_buffer, weights->b_out, output_size);

}

int main() {
    
    printf("Hello, World!\n");

    // create test case for tanh activation function
    float x = 0.5f;
    float y = tanh_act(x);

    // print the result
    printf("tanh(%f) = %f\n", x, y);
    return 0;
}
