#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdbool.h>

#include "gru.h"
#include "linear.h"
#include "util.h"
#if defined _WIN32
    #include "win.h"
#else
    #include <sys/mman.h>
#endif

typedef struct {
    int input_dim;
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;
} GRUModelConfig;

typedef struct {
    GRUModelConfig config;
    GRULayer* gru_layers;
    LinearLayer output_layer;
} GRUModel;

//define a checkpoint config struct to capture checkpoint information
typedef struct {
    char* checkpoint;
    size_t file_size;
    float* data;
} CheckpointConfig;

void init_gru_model(GRUModel* model, GRUModelConfig config) {
    model->config = config;    printf("Initializing GRU model...\n");
    //allocate memory for GRU layers
    model->gru_layers = (GRULayer*)malloc(model->config.num_layers * sizeof(GRULayer));
    int input_dim = model->config.input_dim;
    int input_size = model->config.input_size;
    for (int i = 0; i < model->config.num_layers; i++) {
        init_gru_layer(&model->gru_layers[i], input_dim, input_size, model->config.hidden_size);
        input_size = model->config.hidden_size;
    }
    init_linear_layer(&model->output_layer, input_dim * model->config.hidden_size, model->config.output_size); // the linear layer requires a flattening beforehand
    printf("GRU model initialized.\n");
}

void free_gru_model(GRUModel* model, bool free_weights) {
    printf("Freeing GRU model...\n");
    for (int i = 0; i < model->config.num_layers; i++) {
        free_gru_layer(&model->gru_layers[i], free_weights); // Pass false to not free weights
    }
    free(model->gru_layers); // Free the array of GRU layers
    printf("GRU model freed.\n");
}

void memory_map_weights(GRUModel* model, float *data_ptr) {
    printf("Mapping weights...\n");
    int input_size = model->config.input_size;
    int hidden_size = model->config.hidden_size;
    int num_layers = model->config.num_layers;
    int output_size = model->config.output_size;

    // Map weights and biases from the pointer
    int ptr_offset = 0;
    
    for (int l = 0; l < num_layers; l++) {
        int cell_size = (l == 0) ? input_size : hidden_size;

        //map the weights and biases for the current layer
        model->gru_layers[l].weights.W_ir = data_ptr + ptr_offset;
        ptr_offset += cell_size * hidden_size;

        model->gru_layers[l].weights.W_iz = data_ptr + ptr_offset;
        ptr_offset += cell_size * hidden_size;

        model->gru_layers[l].weights.W_in = data_ptr + ptr_offset;
        ptr_offset += cell_size * hidden_size;

        model->gru_layers[l].weights.W_hr = data_ptr + ptr_offset;
        ptr_offset += hidden_size * hidden_size;

        model->gru_layers[l].weights.W_hz = data_ptr + ptr_offset;
        ptr_offset += hidden_size * hidden_size;

        model->gru_layers[l].weights.W_hn = data_ptr + ptr_offset;
        ptr_offset += hidden_size * hidden_size;

        model->gru_layers[l].weights.b_ir = data_ptr + ptr_offset;
        ptr_offset += hidden_size;

        model->gru_layers[l].weights.b_iz = data_ptr + ptr_offset;
        ptr_offset += hidden_size;

        model->gru_layers[l].weights.b_in = data_ptr + ptr_offset;
        ptr_offset += hidden_size;

        model->gru_layers[l].weights.b_hr = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
        
        model->gru_layers[l].weights.b_hz = data_ptr + ptr_offset;
        ptr_offset += hidden_size;

        model->gru_layers[l].weights.b_hn = data_ptr + ptr_offset;
        ptr_offset += hidden_size;
    }

    // Output Layer Weights and Biases
    model->output_layer.weights.weights = data_ptr + ptr_offset;
    ptr_offset += output_size * hidden_size;

    model->output_layer.weights.bias = data_ptr + ptr_offset;

    //print out the output layer weights and biases
    printf("first Layer Weights: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", model->gru_layers[0].weights.W_ir[i]);
    }
    printf("\n");

    printf("Output Layer Biases: ");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", model->output_layer.weights.bias[i]);
    }

    printf("Weights mapped.\n");
}

void read_checkpoint(CheckpointConfig* checkpoint_config, GRUModel* model) {
    printf("Reading checkpoint from %s...\n", checkpoint_config->checkpoint);
    FILE *file = fopen(checkpoint_config->checkpoint, "rb");
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", checkpoint_config->checkpoint); 
        exit(EXIT_FAILURE); 
    }

    // Figure out the file size
    fseek(file, 0, SEEK_END); // Move file pointer to end of file
    checkpoint_config->file_size = ftell(file); // Get the file size in bytes
    fclose(file);
    printf("File size: %ld bytes\n", checkpoint_config->file_size);

    // Memory map the GRU weights into the data pointer
    int fd = open(checkpoint_config->checkpoint, O_RDONLY); // Open in read-only mode
    if (fd == -1) { 
        fprintf(stderr, "open failed!\n"); 
        exit(EXIT_FAILURE); 
    }

    checkpoint_config->data = mmap(NULL, checkpoint_config->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    if (checkpoint_config->data == MAP_FAILED) { 
        fprintf(stderr, "mmap failed!\n"); 
        exit(EXIT_FAILURE); 
    }

    memory_map_weights(model, checkpoint_config->data);
    printf("Checkpoint read and weights mapped.\n");
}

GRUModel* load_gru_model(CheckpointConfig* checkpoint_config, GRUModelConfig model_config) {
    printf("Loading GRU model from checkpoint...\n");

    GRUModel* model = (GRUModel*)malloc(sizeof(GRUModel));
    init_gru_model(model, model_config);

    // Read the checkpoint and map the weights
    read_checkpoint(checkpoint_config, model);

    return model;
}

void gru_model_forward(GRUModel* model, float* input, float* h_prev, float* output, float* inter_input) {
    printf("Running forward pass through GRU layers...\n");
    for (int i = 0; i < model->config.num_layers; i++) {
        if (i == 0) {
            inter_input = input;
        }
        gru_layer_forward(&model->gru_layers[i], inter_input, h_prev + i * model->config.hidden_size);
        memcpy(h_prev + i * model->config.hidden_size, model->gru_layers[i].state.hidden_state_buffer, model->gru_layers->config.hidden_size * sizeof(float));
        inter_input = model->gru_layers[i].state.hidden_state_buffer; // inter_input points to hidden_state_buffer
    }

    printf("Running forward pass through the output layer...\n");
    linear_layer_forward(&model->output_layer, inter_input, output);
}

int main() { 
    printf("Starting main...\n");
    // // Initialize GRU model
    int input_dim = 1;
    int input_size = 15;
    int hidden_size = 64;
    int output_size = 4;
    int num_layers = 5;

    GRUModelConfig model_config = {input_dim, input_size, hidden_size, output_size, num_layers};
    CheckpointConfig checkpoint_config = {"GRUModel_5_64_1_para.bin", 0, NULL};
    GRUModel* model = load_gru_model(&checkpoint_config, model_config);

    // Example input
    float* input = (float*)calloc(input_dim * input_size, sizeof(float)); // Adjust the size according to input_size
    float* h_prev = (float*)calloc( num_layers * input_dim * hidden_size, sizeof(float)); // Allocate 3 * 64 floats
    float* output = (float*)calloc(input_dim * output_size, sizeof(float)); // Adjust the size according to output_size
    float* inter_input = (float*)calloc(input_dim * hidden_size, sizeof(float)); // Allocate memory for inter_input
    
    for (int i = 0; i < input_dim * input_size; i++) {
        input[i] = 0.3f; // Initialize to 1
    }

    for (int i = 0; i < num_layers * input_dim * hidden_size; i++) {
        h_prev[i] = 0.5f; // Initialize to 1
    }

    // input scaling
    float* in_mean = (float[]){1.62f, 22.25f, 3.83f, 3.90f, 3.91f,
                                3.8886f, 40.52f, 45.20f, 35.51f, 11.53f,
                                25.10f, 29.79f, 99.71f, 2.086f, 13.39f}; 
    float* in_std = (float[]){0.39f, 1.53f, 7.27f, 1.43f, 0.63f,
                                0.68f, 8.35f, 148.63f, 127.37f, 7.53f,
                                3.23f, 4.32f, 3.85f, 14.11f, 27.65f}; 
    standard_scaler(inter_input, input, input_size, in_mean, in_std);

    gru_model_forward(model, inter_input, h_prev, output, inter_input);


    // Print the output
    printf("Output: ");
    for (int i = 0; i < 4; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Free resources
    free(input);
    free(h_prev);
    free(output);
    // free(inter_input); // Free inter_input
    free_gru_model(model, false); // Free the model and its internal memory
    free(model); // Free the model itself

    munmap(checkpoint_config.data, checkpoint_config.file_size);

    printf("Main finished.\n");
    return 0;
}
