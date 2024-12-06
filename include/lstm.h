#ifndef LSTM_H
#define LSTM_H

typedef struct {
    int input_dim;
    int input_size;
    int hidden_size;
} LSTMLayerConfig;


typedef struct {
    float* W_ii;    //input to hidden weights 
    float* W_if;    //forget to hidden weights
    float* W_ig;    //input node or cell to hidden gate 
    float* W_io;    //output to hidden gate
    float* W_hi;    //hidden to hidden gate
    float* W_hf;    //forget to hidden gate 
    float* W_hg;    //cell to hidden gate
    float* W_ho;    //output to hidden gate
    float* b_ii;    //bias for input gate
    float* b_if;    //bias for forget gate
    float* b_ig;    //bias for cell gate
    float* b_io;    //bias for output gate
    float* b_hi;    //bias for hidden input gate 
    float* b_hf;    //bias for hidden forget gate
    float* b_hg;    //bias for hidden cell gate
    float* b_ho;    //bias for hidden output gate
} LSTMLayerWeights;


typedef struct {
    float* input_buffer;
    float* forget_gate_buffer;
    float* input_gate_buffer;
    float* output_gate_buffer;
    float* input_node_buffer; // New buffer
    float* cell_state_buffer;
    float* hidden_state_buffer;
} LSTMLayerRunState;


typedef struct {
    LSTMLayerConfig config;
    LSTMLayerWeights weights;
    LSTMLayerRunState state;
} LSTMLayer;


void init_lstm_layer_config(LSTMLayerConfig* config, int input_dim, int input_size, int hidden_size);
void init_lstm_layer_weights(LSTMLayerWeights* weights, LSTMLayerConfig* config);
void init_lstm_layer_run_state(LSTMLayerRunState* state, LSTMLayerConfig* config);
void init_lstm_layer(LSTMLayer* layer, int input_dim, int input_size, int hidden_size);
void free_lstm_layer_weights(LSTMLayerWeights* weights);
void free_lstm_layer_run_state(LSTMLayerRunState* state);
void free_lstm_layer(LSTMLayer* layer, bool free_weights);
void lstm_layer_forward(LSTMLayer* layer, float* input, float* h_prev, float* c_prev);

#endif // LSTM_H
