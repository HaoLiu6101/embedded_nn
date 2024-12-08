#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "lstm.h"
#include "linear.h"
#include "util.h"

typedef struct {
    int input_dim;
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;
} LSTMModelConfig;

typedef struct {
    LSTMModelConfig config;
    LSTMLayer* lstm_layers;
    LinearLayer output_layer;
} LSTMModel;

void init_lstm_model(LSTMModel* model, LSTMModelConfig config) {
    model->config = config;
    printf("Initializing LSTM model...\n");
    
    // Allocate memory for LSTM layers
    model->lstm_layers = (LSTMLayer*)malloc(model->config.num_layers * sizeof(LSTMLayer));
    
    int input_dim = model->config.input_dim;
    int input_size = model->config.input_size;
    
    // Initialize each LSTM layer
    for (int i = 0; i < model->config.num_layers; i++) {
        init_lstm_layer(&model->lstm_layers[i], input_dim, input_size, model->config.hidden_size);
        input_size = model->config.hidden_size; // Next layer's input size is current layer's hidden size
    }
    
    // Initialize the final linear layer
    init_linear_layer(&model->output_layer, input_dim * model->config.hidden_size, model->config.output_size);
    printf("LSTM model initialized.\n");
}

void free_lstm_model(LSTMModel* model, bool free_weights) {
    printf("Freeing LSTM model...\n");
    for (int i = 0; i < model->config.num_layers; i++) {
        free_lstm_layer(&model->lstm_layers[i], free_weights);
    }
    free(model->lstm_layers);
    free_linear_layer(&model->output_layer);
    printf("LSTM model freed.\n");
}

void memory_map_weights(LSTMModel* model, float* data_ptr) {
    printf("Mapping weights...\n");
    int input_size = model->config.input_size;
    int hidden_size = model->config.hidden_size;
    int num_layers = model->config.num_layers;
    int output_size = model->config.output_size;
    
    int ptr_offset = 0;
    
    for (int l = 0; l < num_layers; l++) {
        int cell_size = (l == 0) ? input_size : hidden_size;
        
        // Map weights and biases for each LSTM layer
        model->lstm_layers[l].weights.W_ii = data_ptr + ptr_offset;
        ptr_offset += cell_size * hidden_size;
        
        model->lstm_layers[l].weights.W_if = data_ptr + ptr_offset;
        ptr_offset += cell_size * hidden_size;
        
        model->lstm_layers[l].weights.W_ig = data_ptr + ptr_offset;
        ptr_offset += cell_size * hidden_size;
        
        model->lstm_layers[l].weights.W_io = data_ptr + ptr_offset;
        ptr_offset += cell_size * hidden_size;
        
        model->lstm_layers[l].weights.W_hi = data_ptr + ptr_offset;
        ptr_offset += hidden_size * hidden_size;
        
        model->lstm_layers[l].weights.W_hf = data_ptr + ptr_offset;
        ptr_offset += hidden_size * hidden_size;
        
        model->lstm_layers[l].weights.W_hg = data_ptr + ptr_offset;
        ptr_offset += hidden_size * hidden_size;
        
        model->lstm_layers[l].weights.W_ho = data_ptr + ptr_offset;
        ptr_offset += hidden_size * hidden_size;
        
        // Map biases
        model->lstm_layers[l].weights.b_ii = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->lstm_layers[l].weights.b_if = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->lstm_layers[l].weights.b_ig = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->lstm_layers[l].weights.b_io = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->lstm_layers[l].weights.b_hi = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->lstm_layers[l].weights.b_hf = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->lstm_layers[l].weights.b_hg = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->lstm_layers[l].weights.b_ho = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
    }
    
    // Map output layer weights and biases
    model->output_layer.weights.weights = data_ptr + ptr_offset;
    ptr_offset += output_size * hidden_size;
    model->output_layer.weights.bias = data_ptr + ptr_offset;
}

int main() {
    printf("Starting LSTM model...\n");
    
    // Initialize model configuration
    int input_dim = 1;
    int input_size = 20;
    int hidden_size = 64;
    int output_size = 4;
    int num_layers = 3;
    
    LSTMModelConfig model_config = {input_dim, input_size, hidden_size, output_size, num_layers};
    LSTMModel* model = (LSTMModel*)malloc(sizeof(LSTMModel));
    init_lstm_model(model, model_config);
    
    // Create sample input
    float* input = (float*)calloc(input_dim * input_size, sizeof(float));
    for (int i = 0; i < input_dim * input_size; i++) {
        input[i] = 0.3f;
    }
    
    // Initialize hidden states and cell states for all layers
    float* h_prev = (float*)calloc(num_layers * input_dim * hidden_size, sizeof(float));
    float* c_prev = (float*)calloc(num_layers * input_dim * hidden_size, sizeof(float));
    float* output = (float*)calloc(input_dim * output_size, sizeof(float));
    float* inter_input = input;
    
    printf("Running forward pass through LSTM layers...\n");
    for (int i = 0; i < num_layers; i++) {
        printf("Layer %d input: ", i);
        for (int j = 0; j < 5; j++) {  // Print first 5 values
            printf("%f ", inter_input[j]);
        }
        printf("...\n");
        
        lstm_layer_forward(&model->lstm_layers[i], 
                          inter_input, 
                          h_prev + i * hidden_size,
                          c_prev + i * hidden_size);
                          
        // Update hidden state for next layer
        memcpy(h_prev + i * hidden_size, 
               model->lstm_layers[i].state.hidden_state_buffer, 
               hidden_size * sizeof(float));
               
        // Update cell state for next layer
        memcpy(c_prev + i * hidden_size, 
               model->lstm_layers[i].state.cell_state_buffer, 
               hidden_size * sizeof(float));
               
        inter_input = model->lstm_layers[i].state.hidden_state_buffer;
    }
    
    printf("Running forward pass through the output layer...\n");
    linear_layer_forward(&model->output_layer, 
                        model->lstm_layers[num_layers-1].state.hidden_state_buffer, 
                        output);
    
    // Print output
    printf("Output: ");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");
    
    // Cleanup
    free(input);
    free(h_prev);
    free(c_prev);
    free(output);
    free_lstm_model(model, true);
    free(model);
    
    printf("LSTM model finished.\n");
    return 0;
} 