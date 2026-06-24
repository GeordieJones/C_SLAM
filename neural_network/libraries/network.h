#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "tensor.h"

typedef enum {
    LAYER_DENSE,
    LAYER_CONVOLUTION,
    LAYER_POOLING,
    LAYER_ADD,
    LAYER_UPSAMPLE
} LayerType;

typedef struct {
    LayerType type;
    void* layer_data;
    int skip_source_idx;
} NetworkLayer;

typedef struct {

    int num_layers;
    NetworkLayer* layers;    
    Tensor** Layer_outputs; 
    Tensor** Layer_gradients; 
 
    float current_loss;
} Network;

typedef enum {
    LOSS_MSE,
    LOSS_CROSS_ENTROPY
} LossType;


Network* Network_make();


void Network_add_dense(Network* net, Layer* dense_layer);

void Network_add_convolution(Network* net, Layer* convolution_layer, int out_height, int out_width);

void Network_add_pooling(Network* net, Layer* pooling_layer);

void Network_add_addition(Network* net, int skip_source_idx);

void Network_add_upsample(Network* net, int scale_factor);


Tensor* Network_forward(Network* net, const Tensor* input);


void Network_backward(Network* net, const Tensor* loss_gradient, Tensor* grad_input);


void Network_update(Network* net, float learning_rate);

float Network_compute_loss(Network* net, const Tensor* predictions, const Tensor* targets, LossType loss_type, Tensor* loss_gradient_dest);


void Network_add_addition(Network* net, int skip_source_idx);


void Network_free(Network* net);

#endif
