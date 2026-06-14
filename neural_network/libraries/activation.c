#include "tensor.h"
#include "dataset.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "activation.h"


// ReLU 
//
static float relu_map(float x){
    if(x <= 0.0f){
        return 0.0f;
    }
    return x;
}

void relu_inplace(Tensor* x){
    Tensor_map(x, relu_map);
}



void relu(const Tensor* src, Tensor* dest){
    for(int i = 0; i < src->total_elements; i++){
        dest->data[i] = src->data[i] > 0.0 ? src->data[i] : 0.0f;
    }
}


void relu_derivative(const Tensor* src, const Tensor* grad_out, Tensor* grad_in){
    for(int i = 0; i< src->total_elements; i++){
        grad_in->data[i] = src->data[i] > 0.0f ? grad_out->data[i] : 0.0f;
    }
}

//leaky ReLU tiny slope to negitives
void leaky_relu_inplace(Tensor* x){
    for(int i = 0; i < x->total_elements; i++){
        x->data[i] = x->data[i] > 0.0 ? x->data[i] : x->data[i] * 0.01f;
    }
}
void leaky_relu(const Tensor* src, Tensor* dest){
    for(int i = 0; i < src->total_elements; i++){
        dest->data[i] = src->data[i] > 0.0 ? src->data[i] : src->data[i] * 0.01f;
    }
}


void leaky_relu_derivative(const Tensor* src, const Tensor* grad_out, Tensor* grad_in){
    for(int i = 0; i< src->total_elements; i++){
        grad_in->data[i] = src->data[i] > 0.0f ? grad_out->data[i] : grad_out->data[i] * 0.01f;
    }
}







// sigmoid funcitons
void sigmoid_inplace(Tensor* x){
    for(int i = 0; i < x->total_elements; i++){
        x->data[i] = 1/(1+(expf(-x->data[i])));
    }
}
void sigmoid(const Tensor* src, Tensor* dest){  
    for(int i = 0; i < src->total_elements; i++){
        dest->data[i] = 1/(1+(expf(-src->data[i])));
    }
}

void sigmoid_derivative(const Tensor* activated_out, const Tensor* grad_out, Tensor* grad_in){
    for(int i = 0; i < activated_out->total_elements; i++){
        float a = activated_out->data[i];
        grad_in->data[i] = grad_out->data[i] * (a*(1.0f - a));
    }
}





// softmax
void softmax(const Tensor* logits, Tensor* probs, int axis) {
    
    int axis_size = logits->shape[axis];
    int axis_stride = logits->strides[axis];
    int num_slices = logits->total_elements / axis_size;

   
    for (int s = 0; s < num_slices; s++) {
        
        int coords[MAX_DIMS] = {0};
        int rem_idx = s;
        for (int j = 0; j < logits->ndim; j++) {
            if (j == axis) continue;
            coords[j] = rem_idx / logits->strides[j];
            rem_idx %= logits->strides[j];
        }
        
        int base_idx = 0;
        for (int j = 0; j < logits->ndim; j++) {
            base_idx += (coords[j] * logits->strides[j]);
        }

        float max_logit = -FLT_MAX;
        for (int k = 0; k < axis_size; k++) {
            int idx = base_idx + (k * axis_stride);
            if (logits->data[idx] > max_logit) {
                max_logit = logits->data[idx];
            }
        }

        float sum = 0.0f;
        for (int k = 0; k < axis_size; k++) {
            int idx = base_idx + (k * axis_stride);
            probs->data[idx] = expf(logits->data[idx] - max_logit);
            sum += probs->data[idx];
        }

  
        for (int k = 0; k < axis_size; k++) {
            int idx = base_idx + (k * axis_stride);
            probs->data[idx] /= sum;
        }
    }
}


void softmax_cross_entropy_derivative(const Tensor* probs, const Tensor* targets, Tensor* grad_in){
    for(int i  = 0; i < probs->total_elements; i++){
        grad_in->data[i] = probs->data[i] - targets->data[i];
    }
}
