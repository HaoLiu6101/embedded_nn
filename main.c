#include <stdio.h>
#include <stdlib.h>
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
    float* hidden_state_buffer;            // Hidden state (num_layers, hidden_size)
    float* input_buffer;            // Input buffer (input_size)
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
    int input_size = config->input_size;
    int output_size = config->output_size;
    weights->W_ir = (float*)calloc(num_layers * hidden_size * input_size, sizeof(float));
    weights->W_iz = (float*)calloc(num_layers * hidden_size * input_size, sizeof(float));
    weights->W_in = (float*)calloc(num_layers * hidden_size * input_size, sizeof(float));
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
    int input_size = config->input_size;
    int output_size = config->output_size;

    state->hidden_state_buffer = (float*)calloc(num_layers * hidden_size, sizeof(float));
    state->input_buffer = (float*)calloc(input_size, sizeof(float));
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

void gru_forware(GRUModel* model, float* input) {
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

    // Copy input to input buffer
    memcpy(input_buffer, input, input_size * sizeof(float));

    // loop over layers
    for (int l = 0; l < num_layers; l++) {
        // get the memory address of weights
        float* W_ir = weights->W_ir + l * hidden_size * input_size;
        float* W_iz = weights->W_iz + l * hidden_size * input_size;
        float* W_in = weights->W_in + l * hidden_size * input_size;
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
        float* h_next = hidden_state_buffer + (l + 1) * hidden_size;
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