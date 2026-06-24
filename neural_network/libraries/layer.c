#include "layer.h"
#include "activation.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>







//two helper funcitons need to be here because in computer_vision is caused a circular dependancy

void img2col(const Tensor* input,int batch_idx, int channels, int height, int width, int kernel_size, int stride, int padding, Tensor* output){

    int output_height = (int)((height - kernel_size + (2*padding)) / stride) + 1;
    int output_width = (int)((width - kernel_size + (2*padding)) / stride) + 1;
    
    int cols = output_height * output_width;
    
    int col_idx = 0;
    

    for(int y = 0; y < output_height; y++){
        for(int x = 0; x < output_width; x++){
           
            int row_idx = 0;

            for(int c = 0; c < channels; c++){

                int base_input_y = (y * stride) - padding;
                int base_input_x = (x * stride) - padding;

                for(int ky = 0; ky < kernel_size; ky++){

                    int input_y = base_input_y + ky;
                    
                    if(input_y < 0 || input_y >= height){
                        //dont check rest of row just fill zero
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int dest_idx = row_idx * cols + col_idx;
                            output->data[dest_idx] = 0.0f;
                            row_idx++;
                        }
                        continue;  
                    }

                    for(int kx = 0; kx < kernel_size; kx++){
    
                        int input_x = base_input_x + kx;
                        int dest_idx = row_idx * cols + col_idx;

                        if(input_x < 0 || input_x >= width){
                            output->data[dest_idx] = 0.0f;
                        }else{
                            int src_idx =   (batch_idx * input->strides[0]) +
                                            (c * input->strides[1]) +
                                            (input_y * input->strides[2]) +
                                            (input_x * input->strides[3]);
                            output->data[dest_idx] = input->data[src_idx];

                        }
                        row_idx++;
                    }
                }
            }
            col_idx++;
        }
    }
}





void col2img(const Tensor* input,int batch_idx, int channels, int height, int width, int kernel_size, int stride, int padding, Tensor* output){

    int output_height = (int)((height - kernel_size + (2*padding)) / stride) + 1;
    int output_width = (int)((width - kernel_size + (2*padding)) / stride) + 1;
    
    int cols = output_height * output_width;
    
    int col_idx = 0;

    Tensor_zero(output);
    

    for(int y = 0; y < output_height; y++){
        for(int x = 0; x < output_width; x++){
           
            int row_idx = 0;

            for(int c = 0; c < channels; c++){

                int base_input_y = (y * stride) - padding;
                int base_input_x = (x * stride) - padding;

                for(int ky = 0; ky < kernel_size; ky++){

                    int input_y = base_input_y + ky;
                    
                    if(input_y < 0 || input_y >= height){
                        row_idx += kernel_size;
                        continue;  
                    }

                    for(int kx = 0; kx < kernel_size; kx++){
    
                        int input_x = base_input_x + kx;
                        int src_idx = row_idx * cols + col_idx;

                        if(!(input_x < 0 || input_x >= width)){
                            int dest_idx =  (batch_idx * output->strides[0]) +
                                            (c * input->strides[1]) +
                                            (input_y * output->strides[2]) +
                                            (input_x * output->strides[3]);
                            output->data[dest_idx] += input->data[src_idx];
                        }
                        row_idx++;
                    }
                }
            }
            col_idx++;
        }
    }
}











Layer* Layer_make_dense(int input_dim, int output_dim, ActivationType activation) {
    Layer* layer = (Layer*)calloc(1, sizeof(Layer));
    if (!layer) return NULL;

    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->activation = activation;

    int w_shape[2] = {input_dim, output_dim};
    int b_shape[1] = {output_dim};
    
    layer->weights = Tensor_make(2, w_shape);
    layer->d_weights = Tensor_make(2, w_shape);
    layer->biases = Tensor_make(1, b_shape);
    layer->d_biases = Tensor_make(1, b_shape);

    float scale = sqrtf(6.0f / (float)input_dim);
    for(int i = 0; i < layer->weights->total_elements; i++) {
        float uniform_rand = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // [-1.0, 1.0]
        layer->weights->data[i] = uniform_rand * scale;
    }
    
    return layer;
}




Layer* Layer_make_convolution(int in_channels, int out_channels, int in_height, int in_width, int kernel_size, int stride, int padding, ActivationType activation) {
    Layer* layer = (Layer*)calloc(1, sizeof(Layer));
    if (!layer) return NULL;

    layer->input_dim = in_channels;

    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->output_channels = out_channels;
    layer->activation = activation;

    int out_height = ((in_height - kernel_size + (2 * padding)) / stride) + 1;
    int out_width  = ((in_width - kernel_size + (2 * padding)) / stride) + 1;

    int flat_kernel_features = in_channels * kernel_size * kernel_size;
    int w_shape[2] = {out_channels, flat_kernel_features};
    int b_shape[1] = {out_channels};

 
    layer->weights = Tensor_make(2, w_shape);
    layer->d_weights = Tensor_make(2, w_shape);
    layer->biases = Tensor_make(1, b_shape);
    layer->d_biases = Tensor_make(1, b_shape);

 
    int cache_shape[2] = {flat_kernel_features, out_height * out_width};
    layer->inputs_cache = Tensor_make(2, cache_shape);

    float scale = sqrtf(6.0f / (float)flat_kernel_features);
    for(int i = 0; i < layer->weights->total_elements; i++) {
        float uniform_rand = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // [-1.0, 1.0]
        layer->weights->data[i] = uniform_rand * scale;
    }


    layer->x_cache = NULL;
    layer->z_cache = NULL;
    layer->a_cache = NULL;

    return layer;
}





Layer* Layer_make_pooling(int in_channels, int in_height, int in_width, int pool_size, int stride) {
    Layer* layer = (Layer*)calloc(1, sizeof(Layer));
    if (!layer) return NULL;

    layer->input_dim = in_channels;

    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->activation = ACTIVATION_NONE;

    int out_height = ((in_height - pool_size) / stride) + 1;
    int out_width  = ((in_width - pool_size) / stride) + 1;

    // Shape: [1, In_Channels, Out_Height, Out_Width] 
    int pool_cache_shape[4] = {1, in_channels, out_height, out_width};
    layer->pooling_indices = Tensor_make(4, pool_cache_shape); 

    return layer;
}




void Layer_free(Layer* self){
    
    Tensor_free(self->weights);    
    Tensor_free(self->biases);
    Tensor_free(self->inputs_cache);
    Tensor_free(self->z_cache);
    Tensor_free(self->a_cache);
    Tensor_free(self->d_weights);
    Tensor_free(self->d_biases);


    //only used in convolution
    if(self->x_cache){
        Tensor_free(self->x_cache);
    }

    //only used if pooling is used
    if(self->pooling_indices){
        Tensor_free(self->pooling_indices);
    }


    free(self);
}




void DenseLayer_forward(Layer* layer, const Tensor* inputs, Tensor* outputs){

    int batch_size = inputs->shape[0];

    if (layer->inputs_cache == NULL || layer->inputs_cache->shape[0] != batch_size) {
        if (layer->inputs_cache) Tensor_free(layer->inputs_cache);
        if (layer->z_cache) Tensor_free(layer->z_cache);
        if (layer->a_cache) Tensor_free(layer->a_cache);

        int z_dims[2] = {batch_size, layer->output_dim};
        layer->z_cache = Tensor_make(2, z_dims);
        layer->a_cache = Tensor_make(2, z_dims);
        
        int x_dims[2] = {batch_size, layer->input_dim};
        layer->inputs_cache = Tensor_make(2, x_dims);
    }

    Tensor_copy(inputs, layer->inputs_cache);
    Tensor_matmul(layer->inputs_cache, layer->weights, layer->z_cache);
    Tensor_vector_addition(layer->z_cache, layer->biases);

    switch(layer->activation){ 
        case ACTIVATION_NONE:
            Tensor_copy(layer->z_cache, layer->a_cache);
            break;
        case ACTIVATION_RELU:
            relu(layer->z_cache, layer->a_cache); 
            break;
        case ACTIVATION_LEAKY_RELU:
            leaky_relu(layer->z_cache, layer->a_cache);
            break;
        case ACTIVATION_SIGMOID:
            sigmoid(layer->z_cache, layer->a_cache);
            break;
        case ACTIVATION_SOFTMAX:
            softmax(layer->z_cache, layer->a_cache, 1);
            break;
    }
    
    Tensor_copy(layer->a_cache, outputs);
}



void DenseLayer_backward(Layer* layer, const Tensor* grad_output, Tensor* grad_input){
    int dz_dims[2] = {grad_output->shape[0], grad_output->shape[1]};
    Tensor* dZ = Tensor_make(2, dz_dims);

    switch(layer->activation){ 
        case ACTIVATION_NONE: 
            Tensor_copy(grad_output, dZ);
            break;
        case ACTIVATION_RELU:
            relu_derivative(layer->a_cache, grad_output, dZ); 
            break;
        case ACTIVATION_LEAKY_RELU:
            leaky_relu_derivative(layer->a_cache, grad_output, dZ);
            break;
        case ACTIVATION_SIGMOID:
            sigmoid_derivative(layer->a_cache, grad_output, dZ);
            break;
        case ACTIVATION_SOFTMAX:
            //softmax_cross_entropy_derivative(layer->z_cache, layer->a_cache, dZ);
            Tensor_copy(grad_output, dZ);
            break;
    }

    Tensor_sum_axis(dZ, layer->d_biases, 0);
    
    Tensor_lazy_transpose(layer->inputs_cache, 0, 1);
    Tensor_matmul(layer->inputs_cache, dZ, layer->d_weights);
    Tensor_lazy_transpose(layer->inputs_cache, 0, 1);

    Tensor_lazy_transpose(layer->weights, 0, 1);
    Tensor_matmul(dZ, layer->weights, grad_input);
    Tensor_lazy_transpose(layer->weights, 0, 1);


    Tensor_free(dZ);
}



void Layer_update(Layer* layer, float learning_rate){
    if(!layer) return;

    for(int i = 0; i < layer->weights->total_elements; i++){
        layer->weights->data[i] -= learning_rate * layer->d_weights->data[i];
    }

    for(int i = 0; i < layer->biases->total_elements; i++){
        layer->biases->data[i] -= learning_rate * layer->d_biases->data[i];
    }
}






void ConvolutionLayer_forward(Layer* layer, const Tensor* input, Tensor* output){
    
    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];

    int out_height = ((in_height - layer->kernel_size +(2 * layer->padding)) / layer->stride) +1;
    int out_width = ((in_width - layer->kernel_size +(2 * layer->padding)) / layer->stride) +1;
    int out_elements = out_height * out_width;


    if (layer->z_cache == NULL || layer->z_cache->shape[0] != batch_size) {
        if (layer->z_cache) Tensor_free(layer->z_cache);
        if (layer->a_cache) Tensor_free(layer->a_cache);
        if (layer->x_cache) Tensor_free(layer->x_cache);
        if (layer->inputs_cache) Tensor_free(layer->inputs_cache);

        int z_dims[4] = {batch_size, layer->output_channels, out_height, out_width};
        layer->z_cache = Tensor_make(4, z_dims);
        layer->a_cache = Tensor_make(4, z_dims);

        int x_dims[4] = {batch_size, in_channels, in_height, in_width};
        layer->x_cache = Tensor_make(4, x_dims);
    
        int cache_shape[2] = {in_channels * layer->kernel_size * layer->kernel_size, out_height * out_width};
        layer->inputs_cache = Tensor_make(2, cache_shape);
    }

    Tensor_copy(input, layer->x_cache);



    Tensor output_slice;
    output_slice.ndim = 2;
    output_slice.shape[0] = layer->output_channels;
    output_slice.shape[1] = out_elements;

    output_slice.strides[0] = out_elements;
    output_slice.strides[1] = 1;
    output_slice.total_elements = layer->output_channels * out_elements;


    for(int b = 0; b < batch_size; b++){

        img2col(layer->x_cache, b, in_channels, in_height, in_width, layer->kernel_size, layer->stride, layer->padding, layer->inputs_cache);

        int batch_offset = b * layer->output_channels * out_elements;
        output_slice.data = &layer->z_cache->data[batch_offset];

        Tensor_matmul(layer->weights, layer->inputs_cache, &output_slice);

        int idx = 0;
        for(int oc = 0; oc < layer->output_channels; oc++){
            float bias_val = layer->biases->data[oc];
            for(int s = 0; s < out_elements; s++){
                output_slice.data[idx] += bias_val;
                idx++;
            }
        }
    }

    switch(layer->activation) {
        case ACTIVATION_NONE:
            Tensor_copy(layer->z_cache, layer->a_cache);
            break;
        case ACTIVATION_RELU:
            relu(layer->z_cache, layer->a_cache);
            break;
        case ACTIVATION_LEAKY_RELU:
            leaky_relu(layer->z_cache, layer->a_cache);
            break;
        case ACTIVATION_SIGMOID:
            sigmoid(layer->z_cache, layer->a_cache);
            break;
        case ACTIVATION_SOFTMAX: 
            softmax(layer->z_cache, layer->a_cache, 1);
            break;
    }


    Tensor_copy(layer->a_cache, output);
}




/*
void ConvolutionLayer_backward(Layer* layer, const Tensor* output_gradient, Tensor* input_gradient) {
    int batch_size = output_gradient->shape[0];
    int out_channels = output_gradient->shape[1];
    int out_height = output_gradient->shape[2];
    int out_width = output_gradient->shape[3];
    int out_elements = out_height * out_width;

    int in_channels = layer->x_cache->shape[1];
    int in_height = layer->x_cache->shape[2];
    int in_width = layer->x_cache->shape[3];
    int flat_kernel_features = in_channels * layer->kernel_size * layer->kernel_size;
 
    int dz_dims[4] = {batch_size, out_channels, out_height, out_width};
    Tensor* dZ = Tensor_make(4, dz_dims);

    // 1. Activation Backprop
    switch(layer->activation) {
        case ACTIVATION_NONE:
            Tensor_copy(output_gradient, dZ);
            break;
        case ACTIVATION_RELU:        
            relu_derivative(layer->a_cache, output_gradient, dZ); 
            break;
        case ACTIVATION_LEAKY_RELU:  
            leaky_relu_derivative(layer->a_cache, output_gradient, dZ); 
            break;
        case ACTIVATION_SIGMOID:     
            sigmoid_derivative(layer->a_cache, output_gradient, dZ);
            break;
        case ACTIVATION_SOFTMAX:
            Tensor_copy(output_gradient, dZ); 
            break;
    }


    Tensor_zero(layer->d_biases);
    Tensor_zero(layer->d_weights);
    Tensor_zero(input_gradient);


    int total_spatial_elements = dZ->total_elements;
    for(int i = 0; i < total_spatial_elements; i++) {
        int oc = (i / out_elements) % out_channels;
        layer->d_biases->data[oc] += dZ->data[i];
    }

    // explicit 2D views for Tensor_matmul compliance
    Tensor dZ_slice;
    dZ_slice.ndim = 2;
    dZ_slice.shape[0] = out_channels;
    dZ_slice.shape[1] = out_elements;
    dZ_slice.strides[0] = out_elements;
    dZ_slice.strides[1] = 1;

    // Force a 2D view over the inputs_cache block populated by img2col
    Tensor inputs_cache_2d;
    inputs_cache_2d.data = layer->inputs_cache->data;
    inputs_cache_2d.ndim = 2;
    inputs_cache_2d.shape[0] = flat_kernel_features;
    inputs_cache_2d.shape[1] = out_elements;
    inputs_cache_2d.strides[0] = out_elements;
    inputs_cache_2d.strides[1] = 1;

    //make temporary 2D tensor for batch weight-gradient collection
    int dW_shape[2] = {out_channels, flat_kernel_features};
    Tensor* dW_batch = Tensor_make(2, dW_shape);

    // Force a 2D view over global layer weights
    Tensor weights_2d;
    weights_2d.data = layer->weights->data;
    weights_2d.ndim = 2;
    weights_2d.shape[0] = layer->weights->shape[0]; // out_channels
    weights_2d.shape[1] = layer->weights->shape[1]; // flat_kernel_features
    weights_2d.strides[0] = layer->weights->strides[0];
    weights_2d.strides[1] = layer->weights->strides[1];

    // Setup grad_col strictly as a 2D target matrix
    
    int grad_col_shape[2] = {flat_kernel_features, out_elements};
    Tensor* grad_col_tensor = Tensor_make(2, grad_col_shape);

    Tensor grad_col;
    grad_col.ndim = 2;
    grad_col.shape[0] = flat_kernel_features;
    grad_col.shape[1] = out_elements;
    grad_col.strides[0] = out_elements;
    grad_col.strides[1] = 1;
    grad_col.data = grad_col_tensor->data;//(float*)malloc(grad_col.shape[0] * out_elements * sizeof(float));


    for (int b = 0; b < batch_size; b++) {
 
        img2col(layer->x_cache, b, in_channels, in_height, in_width, layer->kernel_size, layer->stride, layer->padding, layer->inputs_cache);

        // Update active slice base offset pointer
        int batch_offset = b * out_channels * out_elements;
        dZ_slice.data = &dZ->data[batch_offset];


        Tensor_lazy_transpose(&inputs_cache_2d, 0, 1);
        
        Tensor_matmul(&dZ_slice, &inputs_cache_2d, dW_batch); 
        
        Tensor_lazy_transpose(&inputs_cache_2d, 0, 1);

    
        for(int i = 0; i < layer->d_weights->total_elements; i++) {
            layer->d_weights->data[i] += dW_batch->data[i];
        }

        Tensor_lazy_transpose(&weights_2d, 0, 1);
        
        Tensor_matmul(&weights_2d, &dZ_slice, &grad_col);
        
        Tensor_lazy_transpose(&weights_2d, 0, 1); 


        col2img(&grad_col, b, in_channels, in_height, in_width, layer->kernel_size, layer->stride, layer->padding, input_gradient);
    }

  
    //free(grad_col.data);
    Tensor_free(grad_col_tensor);
    Tensor_free(dW_batch);
    Tensor_free(dZ);
}
*/

void ConvolutionLayer_backward(Layer* layer, const Tensor* output_gradient, Tensor* input_gradient) {
    printf("\n  [CONV BACKWARD] === Entering Convolution Layer Backward ===\n");

    int batch_size = output_gradient->shape[0];
    int out_channels = output_gradient->shape[1];
    int out_height = output_gradient->shape[2];
    int out_width = output_gradient->shape[3];
    int out_elements = out_height * out_width;

    int in_channels = layer->x_cache->shape[1];
    int in_height = layer->x_cache->shape[2];
    int in_width = layer->x_cache->shape[3];
    int flat_kernel_features = in_channels * layer->kernel_size * layer->kernel_size;

    printf("  [CONV BACKWARD] Shapes: Batch=%d, InChannels=%d, OutChannels=%d\n", batch_size, in_channels, out_channels);
    printf("  [CONV BACKWARD] InSpatial=%dx%d | OutSpatial=%dx%d (out_elements=%d)\n", in_height, in_width, out_height, out_width, out_elements);
    printf("  [CONV BACKWARD] flat_kernel_features calculated: %d\n", flat_kernel_features);

    int dz_dims[4] = {batch_size, out_channels, out_height, out_width};
    printf("  [CONV BACKWARD] Allocating dZ tensor...\n");
    Tensor* dZ = Tensor_make(4, dz_dims);
    printf("  [CONV BACKWARD] dZ Allocated successfully. Total elements: %d\n", dZ->total_elements);

    // 1. Activation Backprop
    printf("  [CONV BACKWARD] Executing activation derivative (Type: %d)...\n", layer->activation);
    switch(layer->activation) {
        case ACTIVATION_NONE:
            Tensor_copy(output_gradient, dZ);
            break;
        case ACTIVATION_RELU:
            relu_derivative(layer->a_cache, output_gradient, dZ);
            break;
        case ACTIVATION_LEAKY_RELU:
            leaky_relu_derivative(layer->a_cache, output_gradient, dZ);
            break;
        case ACTIVATION_SIGMOID:
            sigmoid_derivative(layer->a_cache, output_gradient, dZ);
            break;
        case ACTIVATION_SOFTMAX:
            Tensor_copy(output_gradient, dZ);
            break;
    }
    printf("  [CONV BACKWARD] Activation derivative applied cleanly.\n");

    printf("  [CONV BACKWARD] Zeroing gradient tracking arrays...\n");
    Tensor_zero(layer->d_biases);
    Tensor_zero(layer->d_weights);
    Tensor_zero(input_gradient);

    int total_spatial_elements = dZ->total_elements;
    //printf("  [CONV BACKWARD] Accumulating biases. Total spatial iterations: %d\n", total_spatial_elements);
    for(int i = 0; i < total_spatial_elements; i++) {
        int oc = (i / out_elements) % out_channels;
        layer->d_biases->data[oc] += dZ->data[i];
    }
    //printf("  [CONV BACKWARD] Biases accumulated.\n");

    // Explicit 2D views for Tensor_matmul compliance
    Tensor dZ_slice;
    dZ_slice.ndim = 2;
    dZ_slice.shape[0] = out_channels;
    dZ_slice.shape[1] = out_elements;
    dZ_slice.strides[0] = out_elements;
    dZ_slice.strides[1] = 1;

    Tensor inputs_cache_2d;
    inputs_cache_2d.data = layer->inputs_cache->data;
    inputs_cache_2d.ndim = 2;
    inputs_cache_2d.shape[0] = flat_kernel_features;
    inputs_cache_2d.shape[1] = out_elements;
    inputs_cache_2d.strides[0] = out_elements;
    inputs_cache_2d.strides[1] = 1;

    int dW_shape[2] = {out_channels, flat_kernel_features};
    //printf("  [CONV BACKWARD] Allocating dW_batch tensor (Shape: %dx%d)...\n", out_channels, flat_kernel_features);
    Tensor* dW_batch = Tensor_make(2, dW_shape);

    Tensor weights_2d;
    weights_2d.data = layer->weights->data;
    weights_2d.ndim = 2;
    weights_2d.shape[0] = layer->weights->shape[0];
    weights_2d.shape[1] = layer->weights->shape[1];
    weights_2d.strides[0] = layer->weights->strides[0];
    weights_2d.strides[1] = layer->weights->strides[1];

    //printf("  [CONV BACKWARD] Allocating grad_col_tensor framework block...\n");
    int grad_col_shape[2] = {flat_kernel_features, out_elements};
    Tensor* grad_col_tensor = Tensor_make(2, grad_col_shape);

    Tensor grad_col;
    grad_col.ndim = 2;
    grad_col.shape[0] = flat_kernel_features;
    grad_col.shape[1] = out_elements;
    grad_col.strides[0] = out_elements;
    grad_col.strides[1] = 1;
    grad_col.data = grad_col_tensor->data;

    printf("  [CONV BACKWARD] Entering batch loop execution (Total runs: %d)...\n", batch_size);

    // Add this check right before the batch loop in ConvolutionLayer_backward:
    int expected_input_elements = batch_size * in_channels * in_height * in_width;
    if (input_gradient->total_elements < expected_input_elements) {
        printf("\n[CRITICAL SHAPE MISMATCH] input_gradient has %d elements, but col2img expects %d!\n",
               input_gradient->total_elements, expected_input_elements);
        printf("Check Layer 3 output dimension compilation calculations.\n");
        exit(1);
    }


    for (int b = 0; b < batch_size; b++) {
        //printf("    [BATCH LOOP %d] Executing img2col...\n", b);
        img2col(layer->x_cache, b, in_channels, in_height, in_width, layer->kernel_size, layer->stride, layer->padding, layer->inputs_cache);

        int batch_offset = b * out_channels * out_elements;
        dZ_slice.data = &dZ->data[batch_offset];

        //printf("    [BATCH LOOP %d] Running dW weight-gradient Tensor_matmul...\n", b);
        Tensor_lazy_transpose(&inputs_cache_2d, 0, 1);
        Tensor_matmul(&dZ_slice, &inputs_cache_2d, dW_batch);
        Tensor_lazy_transpose(&inputs_cache_2d, 0, 1);

        //printf("    [BATCH LOOP %d] Accumulating dW_batch into global layer d_weights...\n", b);
        for(int i = 0; i < layer->d_weights->total_elements; i++) {
            layer->d_weights->data[i] += dW_batch->data[i];
        }

        //printf("    [BATCH LOOP %d] Running grad_col Tensor_matmul...\n", b);
        Tensor_lazy_transpose(&weights_2d, 0, 1);
        Tensor_matmul(&weights_2d, &dZ_slice, &grad_col);
        Tensor_lazy_transpose(&weights_2d, 0, 1);

        //printf("    [BATCH LOOP %d] Running col2img into input_gradient output...\n", b);
        col2img(&grad_col, b, in_channels, in_height, in_width, layer->kernel_size, layer->stride, layer->padding, input_gradient);
    }
    printf("  [CONV BACKWARD] Batch loop finished completely.\n");

    //printf("  [CONV BACKWARD] Cleaning up tracking structures...\n");
    Tensor_free(grad_col_tensor);
    Tensor_free(dW_batch);
    Tensor_free(dZ);

    printf("  [CONV BACKWARD] === Exiting Convolution Layer Backward Cleanly ===\n");
}



void PoolingLayer_forward(Layer* layer, const Tensor* input, Tensor* output) {
    int batch_size = input->shape[0];
    int channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];

    int out_height = ((in_height - layer->pool_size) / layer->stride) + 1;
    int out_width  = ((in_width - layer->pool_size) / layer->stride) + 1;


    if (layer->pooling_indices == NULL || layer->pooling_indices->shape[0] != batch_size) {
        if (layer->pooling_indices) Tensor_free(layer->pooling_indices);
        int pool_cache_shape[4] = {batch_size, channels, out_height, out_width};
        layer->pooling_indices = Tensor_make(4, pool_cache_shape);
    }

 
    int total_outputs = batch_size * channels * out_height * out_width;
    int spatial_out_area = out_height * out_width;

 
    for (int i = 0; i < total_outputs; i++) {
  
        int b_c_idx = i / spatial_out_area; 
        int out_spatial_idx = i % spatial_out_area;
        
        int oh = out_spatial_idx / out_width;
        int ow = out_spatial_idx % out_width;

        int h_start = oh * layer->stride;
        int w_start = ow * layer->stride;
         
        int input_base_offset = b_c_idx * in_height * in_width;

        float max_val = -3.40282347e+38F; // -FLT_MAX
        int max_idx = -1;


        for (int ph = 0; ph < layer->pool_size; ph++) {
            int in_y = h_start + ph;
            int row_offset = input_base_offset + (in_y * in_width);

            for (int pw = 0; pw < layer->pool_size; pw++) {
                int in_x = w_start + pw;
                int input_idx = row_offset + in_x;
                
                float val = input->data[input_idx];

                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }

  
        output->data[i] = max_val;
        layer->pooling_indices->data[i] = (float)max_idx;
    }
}






void PoolingLayer_backward(Layer* layer, const Tensor* output_gradient, Tensor* input_gradient){
    
    Tensor_zero(input_gradient);

    int total_gradients = output_gradient->total_elements;

    for(int i = 0; i < total_gradients; i++){
        float grad_val = output_gradient->data[i];
        int target_input_idx = (int)layer->pooling_indices->data[i];

        input_gradient->data[target_input_idx] += grad_val;
    }    
}



void Layer_forward_add(Tensor* inputA, Tensor* inputB, Tensor* output) {
    int total = output->total_elements;
    float* out_d = output->data;
    float* a_d = inputA->data;
    float* b_d = inputB->data;

    for (int i = 0; i < total; i++) {
        out_d[i] = a_d[i] + b_d[i];
    }
}


void Layer_forward_upsample(const Tensor* input, Tensor* output, int scale_factor) {
    int batch_size = input->shape[0];
    int channels = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];
    int out_h = output->shape[2];
    int out_w = output->shape[3];

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                int ih = oh / scale_factor;
                if (ih >= in_h) ih = in_h - 1;

                for (int ow = 0; ow < out_w; ow++) {
                    int iw = ow / scale_factor;
                    if (iw >= in_w) iw = in_w - 1;
 
                    int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    int in_idx  = ((b * channels + c) * in_h + ih) * in_w + iw;

                    output->data[out_idx] = input->data[in_idx];
                }
            }
        }
    }
}



void Layer_backward_upsample(const Tensor* output_gradient, Tensor* input_gradient, int scale_factor) {
    int batch_size = input_gradient->shape[0];
    int channels = input_gradient->shape[1];
    int in_h = input_gradient->shape[2];
    int in_w = input_gradient->shape[3];
    int out_h = output_gradient->shape[2];
    int out_w = output_gradient->shape[3];

    for (int i = 0; i < input_gradient->total_elements; i++) {
        input_gradient->data[i] = 0.0f;
    }

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                int ih = oh / scale_factor;
                if (ih >= in_h) ih = in_h - 1;

                for (int ow = 0; ow < out_w; ow++) {
                    int iw = ow / scale_factor;
                    if (iw >= in_w) iw = in_w - 1;

                    int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    int in_idx  = ((b * channels + c) * in_h + ih) * in_w + iw;

                    input_gradient->data[in_idx] += output_gradient->data[out_idx];
                }
            }
        }
    }
}

