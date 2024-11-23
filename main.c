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
    float* candidate_hidden_buffer;            // New gate buffer (hidden_size)
} GRURunState;

/// GRU model
typedef struct {
    GRUConfig config;    // GRU model configuration
    GRUWeights weights;  // GRU model weights
    GRURunState state;   // GRU model run state
} GRUModel;


void init_gru_config(GRUConfig* config_ptr) {
    config_ptr->input_size = 15;    // Set the input size
    config_ptr->hidden_size = 64;   // Set the hidden layer size
    config_ptr->output_size = 4;    // Set the output size
    config_ptr->num_layers = 3;     // Set the number of GRU layers
}

void init_gru_weights(GRUWeights* weights_ptr, GRUConfig* config_ptr) {
    // initialize the weights of GRU
    // allocate memory for all the weights 

    int num_layers = config_ptr->num_layers;
    int hidden_size = config_ptr->hidden_size;
    int input_size = config_ptr->input_size;
    int output_size = config_ptr->output_size;
    weights_ptr->W_ir = (float*)calloc(num_layers * hidden_size * input_size, sizeof(float));
    weights_ptr->W_iz = (float*)calloc(num_layers * hidden_size * input_size, sizeof(float));
    weights_ptr->W_in = (float*)calloc(num_layers * hidden_size * input_size, sizeof(float));
    weights_ptr->W_hr = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));
    weights_ptr->W_hz = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));
    weights_ptr->W_hn = (float*)calloc(num_layers * hidden_size * hidden_size, sizeof(float));
    weights_ptr->b_ir = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights_ptr->b_iz = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights_ptr->b_in = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights_ptr->b_hr = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights_ptr->b_hz = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights_ptr->b_hn = (float*)calloc(num_layers * hidden_size, sizeof(float));
    weights_ptr->W_out = (float*)calloc(output_size * hidden_size, sizeof(float));
    weights_ptr->b_out = (float*)calloc(output_size, sizeof(float));
}

// Function to initialize GRU model run state
void init_gru_run_state(GRURunState* state_ptr, GRUConfig* config_ptr) {
    int num_layers = config_ptr->num_layers;
    int hidden_size = config_ptr->hidden_size;
    int input_size = config_ptr->input_size;
    int output_size = config_ptr->output_size;

    state_ptr->hidden_state_buffer = (float*)calloc(num_layers * hidden_size, sizeof(float));
    state_ptr->input_buffer = (float*)calloc(input_size, sizeof(float));
    state_ptr->output_buffer = (float*)calloc(output_size, sizeof(float));
    state_ptr->reset_gate_buffer = (float*)calloc(hidden_size, sizeof(float));
    state_ptr->update_gate_buffer = (float*)calloc(hidden_size, sizeof(float));
    state_ptr->candidate_hidden_buffer = (float*)calloc(hidden_size, sizeof(float));
}

// Function to free GRU model weights
void free_gru_weights(GRUWeights* weights_ptr) {
    free(weights_ptr->W_ir);
    free(weights_ptr->W_iz);
    free(weights_ptr->W_in);
    free(weights_ptr->W_hr);
    free(weights_ptr->W_hz);
    free(weights_ptr->W_hn);
    free(weights_ptr->b_ir);
    free(weights_ptr->b_iz);
    free(weights_ptr->b_in);
    free(weights_ptr->b_hr);
    free(weights_ptr->b_hz);
    free(weights_ptr->b_hn);
    free(weights_ptr->W_out);
    free(weights_ptr->b_out);
}

// Function to free GRU model run state
void free_gru_run_state(GRURunState* state_ptr) {
    free(state_ptr->hidden_state_buffer);
    free(state_ptr->input_buffer);
    free(state_ptr->output_buffer);
    free(state_ptr->reset_gate_buffer);
    free(state_ptr->update_gate_buffer);
    free(state_ptr->candidate_hidden_buffer);
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

int main() {
    printf("Hello, World!\n");

    // create test case for tanh activation function
    float x = 0.5f;
    float y = tanh_act(x);

    // print the result
    printf("tanh(%f) = %f\n", x, y);
    return 0;
}