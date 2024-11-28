#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdbool.h>
#include <sys/mman.h>
#include "gru.h"
#include "linear.h"
#include "util.h"

typedef struct {
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

void init_gru_model(GRUModel* model, GRUModelConfig config) {
    model->config = config;
    printf("Initializing GRU model...\n");
    //allocate memory for GRU layers
    model->gru_layers = (GRULayer*)malloc(model->config.num_layers * sizeof(GRULayer));
    int input_size = model->config.input_size;
    for (int i = 0; i < model->config.num_layers; i++) {
        init_gru_layer(&model->gru_layers[i], input_size, model->config.hidden_size);
        input_size = model->config.hidden_size;
    }
    init_linear_layer(&model->output_layer, model->config.hidden_size, model->config.output_size);
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

void read_checkpoint(char *checkpoint, float **data, size_t *file_size, GRUModel* model) {
    printf("Reading checkpoint from %s...\n", checkpoint);
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { 
        fprintf(stderr, "Couldn't open file %s\n", checkpoint); 
        exit(EXIT_FAILURE); 
    }

    // Figure out the file size
    fseek(file, 0, SEEK_END); // Move file pointer to end of file
    *file_size = ftell(file); // Get the file size in bytes
    fclose(file);
    printf("File size: %ld bytes\n", *file_size);

    // Memory map the GRU weights into the data pointer
    int fd = open(checkpoint, O_RDONLY); // Open in read-only mode
    if (fd == -1) { 
        fprintf(stderr, "open failed!\n"); 
        exit(EXIT_FAILURE); 
    }

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    if (*data == MAP_FAILED) { 
        fprintf(stderr, "mmap failed!\n"); 
        exit(EXIT_FAILURE); 
    }

    memory_map_weights(model, *data);
    printf("Checkpoint read and weights mapped.\n");
}

int main() { 
    printf("Starting main...\n");
    // Initialize GRU model
    int input_size = 15;
    int hidden_size = 64;
    int output_size = 4;
    int num_layers = 5;

    float* data;
    size_t file_size;

    GRUModelConfig model_config = {input_size, hidden_size, output_size, num_layers};
    GRUModel* model = (GRUModel*)malloc(sizeof(GRUModel));
    init_gru_model(model, model_config);

    read_checkpoint("GRUModel_5_64_1_para.bin", &data, &file_size, model);



    // Example input
    float* input = (float*)calloc(input_size, sizeof(float)); // Adjust the size according to input_size
    for (int i = 0; i < input_size; i++) {
        input[i] = 0.3f; // Initialize to 1
    }
    float* h_prev = (float*)malloc(num_layers * hidden_size * sizeof(float)); // Allocate 3 * 64 floats
    for (int i = 0; i < num_layers * hidden_size; i++) {
        h_prev[i] = 0.5f; // Initialize to 1
    }
    float* output = (float*)calloc(output_size, sizeof(float)); // Adjust the size according to output_size
    float* inter_input = NULL; // Initialize inter_input to NULL


    // input scaling
    float* in_mean = (float[]){1.62f, 22.25f, 3.83f, 3.90f, 3.91f,
                                3.8886f, 40.52f, 45.20f, 35.51f, 11.53f,
                                25.10f, 29.79f, 99.71f, 2.086f, 13.39f}; 
    float* in_std = (float[]){0.39f, 1.53f, 7.27f, 1.43f, 0.63f,
                                0.68f, 8.35f, 148.63f, 127.37f, 7.53f,
                                3.23f, 4.32f, 3.85f, 14.11f, 27.65f}; 
    standard_scaler(inter_input, input, input_size, in_mean, in_std);

    printf("Running forward pass through GRU layers...\n");
    for (int i = 0; i < num_layers; i++) {
        if (i == 0) {
            inter_input = input;
        }
        //print out the input 
        printf("Input: ");
        for (int j = 0; j < input_size; j++) {
            printf("%f ", inter_input[j]);
        }
        printf("\n");
        gru_layer_forward(&model->gru_layers[i], inter_input, h_prev + i * hidden_size);
        memcpy(h_prev + i * hidden_size, model->gru_layers[i].state.hidden_state_buffer, model->gru_layers->config.hidden_size * sizeof(float));
        inter_input = model->gru_layers[i].state.hidden_state_buffer; // inter_input points to hidden_state_buffer
    }

    printf("Running forward pass through the output layer...\n");
    linear_layer_forward(&model->output_layer, model->gru_layers[2].state.hidden_state_buffer, output);

    // Print the output
    printf("Output: ");
    for (int i = 0; i < 4; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Free resources
    //free(input);
    free(h_prev);
    free(output);
    // Do not free inter_input as it points to memory managed by the model

    munmap(data, file_size);
    free_gru_model(model, false); // Free the model and its internal memory
    free(model); // Free the model itself

    printf("Main finished.\n");
    return 0;
}
