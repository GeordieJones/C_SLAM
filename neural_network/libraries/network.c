#include "network.h"
#include "layer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>



Network* Network_make(){
    Network* network = (Network*)malloc(sizeof(Network));
    if(!network) return NULL;
    
    network->num_layers = 0;
    network->layers = NULL;
    network->Layer_outputs = NULL;
    network->Layer_gradients = NULL;
    network->current_loss = 0.0f;
    
    return network;
}


void Network_add_dense(Network* net, Layer* layer){
    int new_layers = net->num_layers +1;

    NetworkLayer* temp_layers  = (NetworkLayer*)realloc(net->layers, sizeof(NetworkLayer) * new_layers);
    Tensor** temp_outputs = (Tensor**)realloc(net->Layer_outputs, sizeof(Tensor*) * new_layers);
    Tensor** temp_gradients = (Tensor**)realloc(net->Layer_gradients, sizeof(Tensor*) * new_layers);


    if(temp_layers == NULL || temp_outputs == NULL || temp_gradients == NULL){
        printf("ERROR out of memory when expanding\n");
        return;
    }


    if(net->num_layers > 0){
        Layer* prev_layer = (Layer*)temp_layers[net->num_layers - 1].layer_data;
        printf("DEBUG: Prev output_dim: %d, New input_dim: %d\n", prev_layer->output_dim, layer->input_dim); // If it dies here, your Layer struct properties don't match!
        if(prev_layer->output_dim != layer->input_dim){
            printf("ERROR: dimesions of new layer do not mathc\n");
            return;
}
    }


    net->layers = temp_layers;
    net->Layer_outputs = temp_outputs;
    net->Layer_gradients = temp_gradients;

    int target_idx = net->num_layers;
    net->layers[target_idx].type = LAYER_DENSE;
    net->layers[target_idx].layer_data = (void*)layer;

    
    int temp_size[2] = {1, layer->output_dim};
    Tensor* temp_tensor = Tensor_make(2, temp_size);
    Tensor* grad_temp_tensor = Tensor_make(2, temp_size);

    net->Layer_outputs[target_idx] = temp_tensor;
    net->Layer_gradients[target_idx] = grad_temp_tensor;


    net->num_layers = new_layers;

}


void Network_add_convolution(Network* net, Layer* layer) {
    int new_layers = net->num_layers + 1;

    NetworkLayer* temp_layers  = (NetworkLayer*)realloc(net->layers, sizeof(NetworkLayer) * new_layers);
    Tensor** temp_outputs = (Tensor**)realloc(net->Layer_outputs, sizeof(Tensor*) * new_layers);
    Tensor** temp_gradients = (Tensor**)realloc(net->Layer_gradients, sizeof(Tensor*) * new_layers);

    if(temp_layers == NULL || temp_outputs == NULL || temp_gradients == NULL){
        printf("ERROR out of memory when expanding network for convolution\n");
        return;
    }

    net->layers = temp_layers;
    net->Layer_outputs = temp_outputs;
    net->Layer_gradients = temp_gradients;

    int target_idx = net->num_layers;
    net->layers[target_idx].type = LAYER_CONVOLUTION;
    net->layers[target_idx].layer_data = (void*)layer;
    net->layers[target_idx].skip_source_idx = -1; 

    // layer->inputs_cache shape is [flat_kernel_features, out_height * out_width]
    int out_spatial_elements = layer->inputs_cache->shape[1];

    // Network_forward will dynamically scale this buffer later if batch size changes
    int temp_size[4] = {1, layer->output_channels, 4, 4};

    // For safety with get_layer_output_shape, we assign a placeholder that matches element boundaries
    net->Layer_outputs[target_idx] = Tensor_make(4, temp_size);
    net->Layer_gradients[target_idx] = Tensor_make(4, temp_size);

    net->num_layers = new_layers;
}

void Network_add_pooling(Network* net, Layer* layer) {
    int new_layers = net->num_layers + 1;

    NetworkLayer* temp_layers  = (NetworkLayer*)realloc(net->layers, sizeof(NetworkLayer) * new_layers);
    Tensor** temp_outputs = (Tensor**)realloc(net->Layer_outputs, sizeof(Tensor*) * new_layers);
    Tensor** temp_gradients = (Tensor**)realloc(net->Layer_gradients, sizeof(Tensor*) * new_layers);

    if(temp_layers == NULL || temp_outputs == NULL || temp_gradients == NULL){
        printf("ERROR out of memory when expanding network for pooling\n");
        return;
    }

    net->layers = temp_layers;
    net->Layer_outputs = temp_outputs;
    net->Layer_gradients = temp_gradients;

    int target_idx = net->num_layers;
    net->layers[target_idx].type = LAYER_POOLING;
    net->layers[target_idx].layer_data = (void*)layer;
    net->layers[target_idx].skip_source_idx = -1;

    int temp_size[4] = {
        1,
        layer->pooling_indices->shape[1], // channels
        layer->pooling_indices->shape[2], // out_height
        layer->pooling_indices->shape[3]  // out_width
    };

    net->Layer_outputs[target_idx] = Tensor_make(4, temp_size);
    net->Layer_gradients[target_idx] = Tensor_make(4, temp_size);

    net->num_layers = new_layers;
}


void Network_add_addition(Network* net, int skip_source_idx) {
    net->num_layers++;
    net->layers = realloc(net->layers, net->num_layers * sizeof(NetworkLayer));
    net->Layer_outputs = realloc(net->Layer_outputs, net->num_layers * sizeof(Tensor*));
    net->Layer_gradients = realloc(net->Layer_gradients, net->num_layers * sizeof(Tensor*));

    int new_idx = net->num_layers - 1;
    net->layers[new_idx].type = LAYER_ADD;
    net->layers[new_idx].layer_data = NULL;
    net->layers[new_idx].skip_source_idx = skip_source_idx;

    net->Layer_outputs[new_idx] = NULL;
    net->Layer_gradients[new_idx] = NULL;
}


void Network_add_upsample(Network* net, int scale_factor) {
    net->num_layers++;
    net->layers = realloc(net->layers, net->num_layers * sizeof(NetworkLayer));
    net->Layer_outputs = realloc(net->Layer_outputs, net->num_layers * sizeof(Tensor*));
    net->Layer_gradients = realloc(net->Layer_gradients, net->num_layers * sizeof(Tensor*));

    int new_idx = net->num_layers - 1;
    net->layers[new_idx].type = LAYER_UPSAMPLE;
    net->layers[new_idx].layer_data = NULL;
    net->layers[new_idx].skip_source_idx = scale_factor; 

    net->Layer_outputs[new_idx] = NULL;
    net->Layer_gradients[new_idx] = NULL;
}



// Static local helper to resolve the output shape of a layer dynamically
static void get_layer_output_shape(const Network* net, int layer_idx, const Tensor* input, int* out_ndim, int* out_shape) {
    NetworkLayer* nl = &net->layers[layer_idx];

    if (nl->type == LAYER_DENSE) {
      // [Batch, Output_Dim]
        Layer* dense = (Layer*)nl->layer_data;
        *out_ndim = 2;
        out_shape[0] = input->shape[0]; // Batch size
        out_shape[1] = dense->output_dim;

    } else if (nl->type == LAYER_CONVOLUTION) {
        //[Batch, Channels, Height, Width]
        Layer* conv = (Layer*)nl->layer_data;
        *out_ndim = 4;
        out_shape[0] = input->shape[0]; // Batch 
        out_shape[1] = conv->output_channels;

        //using CNN
        out_shape[2] = ((input->shape[2] - conv->kernel_size + 2 * conv->padding) / conv->stride) + 1;
        out_shape[3] = ((input->shape[3] - conv->kernel_size + 2 * conv->padding) / conv->stride) + 1;

    } else if (nl->type == LAYER_POOLING) {
        Layer* pool = (Layer*)nl->layer_data;
        *out_ndim = 4;
        out_shape[0] = input->shape[0]; // Batch size
        out_shape[1] = input->shape[1]; // Channels stay the same

        out_shape[2] = ((input->shape[2] - pool->pool_size) / pool->stride) + 1;
        out_shape[3] = ((input->shape[3] - pool->pool_size) / pool->stride) + 1;
    } else if (net->layers[layer_idx].type == LAYER_ADD) {

        *out_ndim = input->ndim;

        for (int d = 0; d < input->ndim; d++) {
            out_shape[d] = input->shape[d];
        }
    }else if (nl->type == LAYER_UPSAMPLE) {
        int scale_factor = nl->skip_source_idx; 
        *out_ndim = 4;
        out_shape[0] = input->shape[0]; 
        out_shape[1] = input->shape[1];
        out_shape[2] = input->shape[2] * scale_factor; 
        out_shape[3] = input->shape[3] * scale_factor;
    }

    // needed to inspect the NEXT layer to adjust shapes

    if (layer_idx + 1 < net->num_layers) {
        NetworkLayer* next_nl = &net->layers[layer_idx + 1];
        if (next_nl->type == LAYER_DENSE && *out_ndim == 4) {            
            int flat_features = out_shape[1] * out_shape[2] * out_shape[3];
            *out_ndim = 2;
            out_shape[1] = flat_features;
        }
    }
   
}










Tensor* Network_forward(Network* net, const Tensor* input){
    if(!net || !input) return NULL;    
    int batch_size = input->shape[0];

    for(int i = 0; i < net->num_layers; i++){
        Tensor* current_input = (i == 0) ? (Tensor*)input : net->Layer_outputs[i-1];
        int target_ndim = 0;
        int target_shape[4] = {0};
        get_layer_output_shape(net, i, current_input, &target_ndim, target_shape);

        if(!Tensor_matches_shape(net->Layer_outputs[i], target_ndim, target_shape)){
            Tensor_free(net->Layer_outputs[i]);
            net->Layer_outputs[i] = Tensor_make(target_ndim,target_shape); 
        }

        if(!Tensor_matches_shape(net->Layer_gradients[i], target_ndim, target_shape)){            
            Tensor_free(net->Layer_gradients[i]);
            net->Layer_gradients[i] = Tensor_make(target_ndim, target_shape); 
        }


        Tensor* current_output = net->Layer_outputs[i];
        Layer* layer = (Layer*)net->layers[i].layer_data;

        
        //later make case for more types
        if(net->layers[i].type == LAYER_DENSE){
            DenseLayer_forward(layer, current_input, current_output);
        }else if(net->layers[i].type == LAYER_CONVOLUTION){
            ConvolutionLayer_forward(layer, current_input, current_output);     
        }else if(net->layers[i].type == LAYER_POOLING){
            PoolingLayer_forward(layer, current_input, current_output);
        }else if(net->layers[i].type == LAYER_ADD){ 
            int skip_idx = net->layers[i].skip_source_idx;
            Tensor* skip_tensor = net->Layer_outputs[skip_idx];
            Layer_forward_add(current_input, skip_tensor, current_output);
        }else if(net->layers[i].type == LAYER_UPSAMPLE){
            int scale_factor = net->layers[i].skip_source_idx;
            Layer_forward_upsample(current_input, current_output, scale_factor);
        }

    }
    
    return net->Layer_outputs[net->num_layers-1];
}



void Network_backward(Network* net, const Tensor* loss_gradient, Tensor* grad_input){
    
    for(int i = net->num_layers -1; i>= 0; i--){
        printf("DEBUG: Processing Layer %d (Type Enum Value: %d)\n", i, net->layers[i].type);

        Tensor* current_output_gradient = NULL;
        if(i == net->num_layers -1){
            current_output_gradient = (Tensor*)loss_gradient;
        }else{
            current_output_gradient = net->Layer_gradients[i];
        }

        printf("  -> current_output_gradient: %s (Elements: %d)\n",current_output_gradient ? "VALID" : "NULL",current_output_gradient ? current_output_gradient->total_elements : 0);

        Tensor* current_input_gradient = NULL;
        if(i == 0){ 
            current_input_gradient = grad_input;
        }else{ 
            current_input_gradient = net->Layer_gradients[i-1];
        }

        printf("  -> current_input_gradient: %s\n", current_input_gradient ? "VALID" : "NULL");

        Layer* layer = (Layer*)net->layers[i].layer_data;


        if(net->layers[i].type == LAYER_DENSE){ 
            printf("  -> Executing Dense Layer Backward...\n");
            DenseLayer_backward(layer, current_output_gradient, current_input_gradient);
        }else if(net->layers[i].type == LAYER_CONVOLUTION){
            printf("  -> Executing Convolution Layer Backward...\n");
            if(!layer->x_cache) printf("     WARNING: layer->x_cache is NULL!\n");
            ConvolutionLayer_backward(layer, current_output_gradient, current_input_gradient);     
        }else if(net->layers[i].type == LAYER_POOLING){
            printf("  -> Executing Pooling Layer Backward...\n");
            PoolingLayer_backward(layer, current_output_gradient, current_input_gradient);
        }else if(net->layers[i].type == LAYER_ADD){
            printf("  -> Executing Add Layer Backward (Skip Index: %d)...\n", net->layers[i].skip_source_idx);
            if(i > 0){
                for(int g = 0; g < current_output_gradient->total_elements; g++){
                    current_input_gradient->data[g] = current_output_gradient->data[g];
                }
            }
                //check you added skip_source_idx to the struct later
            int skip_idx = net->layers[i].skip_source_idx;
            Tensor* skip_grad = net->Layer_gradients[skip_idx];
            printf("     -> skip_grad pointer: %p\n", (void*)skip_grad);

            if (!skip_grad) {
                printf("     -> ERROR: skip_grad is NULL at index %d!\n", skip_idx);
            } else {
                printf("     -> skip_grad elements: %d\n", skip_grad->total_elements);
                for(int g = 0; g < current_output_gradient->total_elements; g++){
                    skip_grad->data[g] += current_output_gradient->data[g];
                }
            }         
        }else if(net->layers[i].type == LAYER_UPSAMPLE){
            printf("  -> Executing Upsample Layer Backward...\n");
            int scale_factor = net->layers[i].skip_source_idx;
            Layer_backward_upsample(current_output_gradient, current_input_gradient, scale_factor);
        }
        printf("DEBUG: Layer %d Complete.\n\n", i);
    }
}




void Network_update(Network* net, float learning_rate){
    for(int i = 0; i < net->num_layers; i++){
        int type = net->layers[i].type;
        void* data = net->layers[i].layer_data;

         if(type == LAYER_DENSE || type == LAYER_CONVOLUTION) {
            if (data != NULL) {
                printf("  -> Calling Layer_update on Layer %d...\n", i);
                Layer* layer = (Layer*)data;
                Layer_update(layer, learning_rate);
                printf("  -> Layer %d updated successfully.\n", i);
            } else {
                printf("  -> WARNING: Layer %d is weights-based but data pointer is NULL!\n", i);
            }
        }
    } 
}

/*
void Network_update(Network* net, float learning_rate){
    printf("\n=== STARTING NETWORK PARAMETER UPDATE DIAGNOSTICS ===\n");
    for(int i = 0; i < net->num_layers; i++){
        int type = net->layers[i].type;
        void* data = net->layers[i].layer_data;
        
        printf("DEBUG: Updating Layer %d | Type Enum Value: %d | Data Pointer: %p\n", i, type, data);
        
        // Explicitly only let DENSE and CONVOLUTION pass to Layer_update!
        if(type == LAYER_DENSE || type == LAYER_CONVOLUTION) {
            if (data != NULL) {
                printf("  -> Calling Layer_update on Layer %d...\n", i);
                Layer* layer = (Layer*)data;
                Layer_update(layer, learning_rate);
                printf("  -> Layer %d updated successfully.\n", i);
            } else {
                printf("  -> WARNING: Layer %d is weights-based but data pointer is NULL!\n", i);
            }
        } else {
            printf("  -> Skipping non-parametric structural layer (Type %d)\n", type);
        }
    }
    printf("=== PARAMETER UPDATE COMPLETED CLEANLY ===\n");
}

*/


float Network_compute_loss(Network* net, const Tensor* predictions, const Tensor* targets, LossType loss_type, Tensor* loss_gradient_dest){

    if(predictions->total_elements !=  targets->total_elements){
        printf("ERROR: total elements do not match for pred and targets");
        return 0.0f;
    }

    if(loss_gradient_dest->shape[0] != predictions->shape[0] || loss_gradient_dest->shape[1] != predictions->shape[1]){
        printf("ERROR: loss_gradient_dest shape mismatch!\n");
        return 0.0f;
    }
    
    float avg_loss = 0.0f;

    switch(loss_type){
        case LOSS_MSE:{
            float l_squared = 0.0;
            for(int i = 0; i < predictions->total_elements; i++){
                float diff = predictions->data[i] - targets->data[i];
                l_squared += diff * diff;
                loss_gradient_dest->data[i] = (2.0f*(diff)) / (float)(predictions->total_elements);
            }
            avg_loss = l_squared / (float)(predictions->total_elements);
            break;
        }

        case LOSS_CROSS_ENTROPY:{
            
            float l_sum = 0.0f;
            for(int i = 0; i < predictions->total_elements; i++){
                float pred_val = predictions->data[i] < 1e-15f ? 1e-15f: predictions->data[i];
                l_sum += (targets->data[i] * logf(pred_val));
                loss_gradient_dest->data[i] = -(predictions->data[i] - targets->data[i]) / (float)predictions->shape[0];
            }
            avg_loss = -(l_sum / (float)predictions->shape[0]);
            break;
        }    


    }

    net->current_loss = avg_loss;
    return avg_loss;    
}










void Network_free(Network* net){
    for(int i = 0; i < net->num_layers; i++){
        if(net->layers[i].layer_data){
            Layer* layer = (Layer*)net->layers[i].layer_data;
            Layer_free(layer);
        }
        
        if(net->Layer_outputs[i]){
            Tensor_free(net->Layer_outputs[i]);
        }
        if(net->Layer_gradients[i]){
            Tensor_free(net->Layer_gradients[i]);
        }
    }

    if(net->layers) free(net->layers);
    if(net->Layer_gradients) free(net->Layer_gradients);
    if(net->Layer_outputs) free(net->Layer_outputs);

    free(net);
}




