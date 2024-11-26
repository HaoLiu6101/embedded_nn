
#ifndef LINEAR_H
#define LINEAR_H

typedef struct {
    int input_size;
    int output_size;
} LinearLayerConfig;

typedef struct {
    float* weights;
    float* bias;
} LinearLayerWeights;

typedef struct {
    LinearLayerConfig config;
    LinearLayerWeights weights;
} LinearLayer;

void init_linear_layer_config(LinearLayerConfig* config, int input_size, int output_size);
void init_linear_layer_weights(LinearLayerWeights* weights, LinearLayerConfig* config);
void init_linear_layer(LinearLayer* layer, int input_size, int output_size);
void free_linear_layer_weights(LinearLayerWeights* weights);
void free_linear_layer(LinearLayer* layer);
void linear_layer_forward(LinearLayer* layer, float* input, float* output);

#endif // LINEAR_H