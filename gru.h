
#ifndef GRU_H
#define GRU_H

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;
} GRUConfig;

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
    float* W_out;
    float* b_out;
} GRUWeights;

typedef struct {
    float* hidden_state_buffer;
    float* input_buffer;
    float* output_buffer;
    float* reset_gate_buffer;
    float* update_gate_buffer;
    float* candidate_hidden_state_buffer;
    float* hidden_cell_temp;
} GRURunState;

typedef struct {
    GRUConfig config;
    GRUWeights weights;
    GRURunState state;
} GRUModel;

void init_gru_config(GRUConfig* config);
void init_gru_weights(GRUWeights* weights, GRUConfig* config);
void init_gru_run_state(GRURunState* state, GRUConfig* config);
void free_gru_weights(GRUWeights* weights);
void free_gru_run_state(GRURunState* state);
void init_gru_model(GRUModel* model);
void free_gru_model(GRUModel* model);
void gru_forward(GRUModel* model, float* input);

#endif // GRU_H