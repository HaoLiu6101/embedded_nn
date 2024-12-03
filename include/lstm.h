#ifndef LSTM_H
#define LSTM_H

typedef struct {
    int input_size;
    int hidden_size;
} LSTMLayerConfig;


typedef struct {
    float* W_ii;
    float* W_if;
    float* W_ig;
    float* W_io;
    float* W_hi;
    float* W_hf;
    float* W_hg;
    float* W_ho;
    float* b_ii;
    float* b_if;
    float* b_ig;
    float* b_io;
    float* b_hi;
    float* b_hf;
    float* b_hg;
    float* b_ho;
} LSTMLayerWeights;


typedef struct {
    float* input_buffer;
    float* forget_gate_buffer;
    float* input_gate_buffer;
    float* output_gate_buffer;
    float* cell_state_buffer;
    float* hidden_state_buffer;
} LSTMLayerRunState;


typedef struct {
    LSTMLayerConfig config;
    LSTMLayerWeights weights;
    LSTMLayerRunState state;
} LSTMLayer;


void init_lstm_layer_config(LSTMLayerConfig* config, int input_size, int hidden_size);
void init_lstm_layer_weights(LSTMLayerWeights* weights, LSTMLayerConfig* config);
void init_lstm_layer_run_state(LSTMLayerRunState* state, LSTMLayerConfig* config);
void init_lstm_layer(LSTMLayer* layer, int input_size, int hidden_size);
void free_lstm_layer_weights(LSTMLayerWeights* weights);
void free_lstm_layer_run_state(LSTMLayerRunState* state);
void free_lstm_layer(LSTMLayer* layer, bool free_weights);
void lstm_layer_forward(LSTMLayer* layer, float* input, float* h_prev);

#endif // LSTM_H
