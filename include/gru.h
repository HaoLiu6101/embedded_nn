#ifndef GRU_H
#define GRU_H

typedef struct {
    int input_dim;
    int input_size;
    int hidden_size;
} GRULayerConfig;

typedef struct {
    float* W_ir;
    float* W_iz;
    float* W_in;
    float* W_hr;
    float* W_hz;
    float* W_hn;
    float* b_ir;
    float* b_iz;
    float* b_in;
    float* b_hr;
    float* b_hz;
    float* b_hn;
} GRULayerWeights;

typedef struct {
    float* hidden_state_buffer;
    float* input_buffer;
    float* reset_gate_buffer;
    float* update_gate_buffer;
    float* candidate_hidden_state_buffer;
} GRULayerRunState;

typedef struct {
    GRULayerConfig config;
    GRULayerWeights weights;
    GRULayerRunState state;
} GRULayer;

void init_gru_layer_config(GRULayerConfig* config, int input_dim, int input_size, int hidden_size);
void init_gru_layer_weights(GRULayerWeights* weights, GRULayerConfig* config);
void init_gru_layer_run_state(GRULayerRunState* state, GRULayerConfig* config);
void init_gru_layer(GRULayer* layer, int input_dim, int input_size, int hidden_size);
void free_gru_layer_weights(GRULayerWeights* weights);
void free_gru_layer_run_state(GRULayerRunState* state);
void free_gru_layer(GRULayer* layer, bool free_weights);
void gru_layer_forward(GRULayer* layer, float* input, float* h_prev);

#endif // GRU_H