#include "Rte_FITin_TMCAIShadow.h"
#include "math_nn.h"  // Include the new header file

#define FITin_TMCAIShadow_START_SEC_CODE                   
#include "FITin_TMCAIShadow_MemMap.h"

// GRU model configuration
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
    float* W_in;         // Input to new gate weights (num_layers, hidden_size, input_size)
    float* W_hr;         // Hidden to reset gate weights (num_layers, hidden_size, hidden_size)
    float* W_hz;         // Hidden to update gate weights (num_layers, hidden_size, hidden_size)
    float* W_hn;         // Hidden to new gate weights (num_layers, hidden_size, hidden_size)
    float* b_ir;         // Input to reset gate biases (num_layers, hidden_size)
    float* b_iz;         // Input to update gate biases (num_layers, hidden_size)
    float* b_in;         // Input to new gate biases (num_layers, hidden_size)
    float* b_hr;         // Hidden to reset gate biases (num_layers, hidden_size)
    float* b_hz;         // Hidden to update gate biases (num_layers, hidden_size)
    float* b_hn;         // Hidden to new gate biases (num_layers, hidden_size)
    float* W_out;        // Output weights (output_size, hidden_size)
    float* b_out;        // Output biases (output_size)
} GRUWeights;

// GRU model run state
typedef struct {
    float* h;            // Hidden state (num_layers, hidden_size)
    float* x;            // Input buffer (input_size)
    float* y;            // Output buffer (output_size)
    float* r;            // Reset gate buffer (hidden_size)
    float* z;            // Update gate buffer (hidden_size)
    float* n;            // New gate buffer (hidden_size)
} GRURunState;

// GRU model
typedef struct {
    GRUConfig config;    // GRU model configuration
    GRUWeights weights;  // GRU model weights
    GRURunState state;   // GRU model run state
} GRUModel;

// Function to initialize GRU model configuration
void init_gru_config(GRUConfig* config) {
    config->input_size = 15;
    config->hidden_size = 64;
    config->output_size = 4;
    config->num_layers = 3;
}

// Function to initialize GRU model weights
void init_gru_weights(GRUWeights* weights, GRUConfig* config) {
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

    state->h = (float*)calloc(num_layers * hidden_size, sizeof(float));
    state->x = (float*)calloc(input_size, sizeof(float));
    state->y = (float*)calloc(output_size, sizeof(float));
    state->r = (float*)calloc(hidden_size, sizeof(float));
    state->z = (float*)calloc(hidden_size, sizeof(float));
    state->n = (float*)calloc(hidden_size, sizeof(float));
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
    free(state->h);
    free(state->x);
    free(state->y);
    free(state->r);
    free(state->z);
    free(state->n);
}

// Function to initialize GRU model
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

// Function to perform GRU forward pass
void gru_forward(GRUModel* model, float* input) {
    GRUConfig* config = &model->config;
    GRUWeights* weights = &model->weights;
    GRURunState* state = &model->state;

    int num_layers = config->num_layers;
    int hidden_size = config->hidden_size;
    int input_size = config->input_size;
    int output_size = config->output_size;

    float* h = state->h;
    float* x = state->x;
    float* y = state->y;
    float* r = state->r;
    float* z = state->z;
    float* n = state->n;

    memcpy(x, input, input_size * sizeof(float));

    for (int l = 0; l < num_layers; l++) {
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

        float* h_prev = h + l * hidden_size;
        float* h_next = h + (l + 1) * hidden_size;

        // Reset gate
        matmul(r, x, W_ir, 1, input_size, hidden_size);
        add(r, r, b_ir, hidden_size);
        matmul(r, h_prev, W_hr, 1, hidden_size, hidden_size);
        add(r, r, b_hr, hidden_size);
        sigmoid_act_vec(r, r, hidden_size);

        // Update gate
        matmul(z, x, W_iz, 1, input_size, hidden_size);
        add(z, z, b_iz, hidden_size);
        matmul(z, h_prev, W_hz, 1, hidden_size, hidden_size);
        add(z, z, b_hz, hidden_size);
        sigmoid_act_vec(z, z, hidden_size);

        // New gate or candidate hidden state
        matmul(n, x, W_in, 1, input_size, hidden_size);
        add(n, n, b_in, hidden_size);

        // Hadamard product between reset gate r_t and h_prev
        float r_h_prev[hidden_size];
        for (int i = 0; i < hidden_size; i++) {
            r_h_prev[i] = r[i] * h_prev[i];
        }

        matmul(n, r_h_prev, W_hn, 1, hidden_size, hidden_size);
        add(n, n, b_hn, hidden_size);
        tanh_act_vec(n, n, hidden_size);

        // Update hidden state
        for (int i = 0; i < hidden_size; i++) {
            h_next[i] = (1.0f - z[i]) * n[i] + z[i] * h_prev[i];
        }

        // Update input for the next layer
        memcpy(x, h_next, hidden_size * sizeof(float));
    }

    // Compute output
    matmul(y, h_next, weights->W_out, 1, hidden_size, output_size);
    add(y, y, weights->b_out, output_size);
}

FUNC (void, FITin_TMCAIShadow_CODE) FITin_TMCAIShadow_Init/* return value & FctID */
(
    void
)
{
    // Initialize GRU model
    GRUModel model;
    init_gru_model(&model);

    // Free GRU model
    free_gru_model(&model);
}

FUNC (void, FITin_TMCAIShadow_CODE) FITin_TMCAIShadow_Main/* return value & FctID */
(
    void
)
{
    // Initialize GRU model
    GRUModel model;
    init_gru_model(&model);

    // Input data
    float input[15] = {
        DefMotVolFb,
        BlowerPwmSetVal,
        AirDisMotVolFb,
        AirIntakeMotVolFb,
        AM_VolFb_Right,
        AM_VolFb_Left,
        HeatCoreInWatTemp,
        SolarLeft,
        SolarRight,
        EvaporatorTemp,
        CorrectedExterTemp,
        AcPumpSpd,
        BatPmpActSpdRatio,
        AC_CDV1Pos,
        AmbTempFildForAirCon
    };

    // Forward pass through GRU model
    gru_forward(&model, input);

    // Get output
    float* output = model.state.y;

    // Write output to RTE
    DuctTempFootRight_Predicted = output[0];
    DuctTempFaceRight_Predicted = output[1];
    DuctTempFootLeft_Predicted = output[2];
    DuctTempFaceLeft_Predicted = output[3];

    Rte_IWrite_FITin_TMCAIShadow_Main_DuctTempFootRight_Predicted_gdf32(DuctTempFootRight_Predicted);
    Rte_IWrite_FITin_TMCAIShadow_Main_DuctTempFaceRight_Predicted_gdf32(DuctTempFaceRight_Predicted);
    Rte_IWrite_FITin_TMCAIShadow_Main_DuctTempFootLeft_Predicted_gdf32(DuctTempFootLeft_Predicted);
    Rte_IWrite_FITin_TMCAIShadow_Main_DuctTempFaceLeft_Predicted_gdf32(DuctTempFaceLeft_Predicted);

    // Free GRU model
    free_gru_model(&model);
}

#define FITin_TMCAIShadow_STOP_SEC_CODE  
#include "FITin_TMCAIShadow_MemMap.h" 

/*PROTECTED REGION ID(FileHeaderUserDefinedFunctions :FITin_TMCAIShadow) ENABLED START */
/* Start of user defined functions  - Do not remove this comment */
/* End of user defined functions - Do not remove this comment */
/*PROTECTED REGION END */