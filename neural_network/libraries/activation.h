#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"



//ReLU 
void relu_inplace(Tensor* x);
void relu(const Tensor* src, Tensor* dest);
void relu_derivative(const Tensor* src, const Tensor* grad_out, Tensor* grad_in);

//leaky ReLU tiny slope to negitives
void leaky_relu_inplace(Tensor* x);
void leaky_relu(const Tensor* src, Tensor* dest);
void leaky_relu_derivative(const Tensor* src, const Tensor* grad_out, Tensor* grad_in);

// sigmoid funcitons
void sigmoid_inplace(Tensor* x);
void sigmoid(const Tensor* src, Tensor* dest);
void sigmoid_derivative(const Tensor* activated_out, const Tensor* grad_out, Tensor* grad_in);

// softmax
void softmax(const Tensor* logits, Tensor* probs, int axis);
void softmax_cross_entropy_derivative(const Tensor* probs, const Tensor* targets, Tensor* grad_in);







#endif
