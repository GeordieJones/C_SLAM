#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"



typedef enum {
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_SOFTMAX
} ActivationType;

typedef struct { 
    int input_dim;
    int output_dim;

    ActivationType activation;


    //varibles for convolution and pooling
    int kernel_size;
    int stride;
    int padding;
    int output_channels;
    int pool_size;


    struct Layer* skip_source_layer;
 
    Tensor* weights;           // Shape: [input_dim, output_dim] (Contiguous for X * W)
    Tensor* biases;            // Shape: [output_dim] (1D Tensor for easy broadcasting)

   
    Tensor* inputs_cache;      // Caches incoming 'X' (Shape: [batch_size, input_dim])
    Tensor* z_cache;           // Caches raw linear output: Z = X*W + b (Shape: [batch_size, output_dim])
    Tensor* a_cache;           // Caches activated output: A = activation(Z) (Shape: [batch_size, output_dim])

    Tensor* x_cache;            // this is for convolution

    Tensor* d_weights;         // Gradient of loss w.r.t weights (Shape: [input_dim, output_dim])
    Tensor* d_biases;          // Gradient of loss w.r.t biases (Shape: [output_dim])
                               //
    Tensor* pooling_indices;   // keeping track of pooling locations

} Layer;




Layer* Layer_make_dense(int input_dim, int output_dim, ActivationType activation);

Layer* Layer_make_convolution(int in_channels, int out_channels, int in_height, int in_width, int kernel_size, int stride, int padding, ActivationType activation);

Layer* Layer_make_pooling(int in_channels, int in_height, int in_width, int pool_size, int stride);

void Layer_free(Layer* self);

void DenseLayer_forward(Layer* layer, const Tensor* inputs, Tensor* outputs);

void DenseLayer_backward(Layer* layer, const Tensor* grad_output, Tensor* grad_input);

void Layer_update(Layer* layer, float learning_rate);



void ConvolutionLayer_forward(Layer* layer, const Tensor* input, Tensor* output);

void ConvolutionLayer_backward(Layer* layer, const Tensor* output_gradient, Tensor* input_gradient);


void PoolingLayer_forward(Layer* layer, const Tensor* input, Tensor* output);


void PoolingLayer_backward(Layer* layer, const Tensor* output_gradient, Tensor* input_gradient);


void Layer_forward_add(Tensor* inputA, Tensor* inputB, Tensor* output);

void Layer_forward_upsample(const Tensor* input, Tensor* output, int scale_factor); 

void Layer_backward_upsample(const Tensor* output_gradient, Tensor* input_gradient, int scale_factor);

#endif 
