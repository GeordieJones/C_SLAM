#include "layer.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f

// Helper to manually check if two floats are close enough
int is_close(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

// Minimal mock tensor implementations if your tensor library names vary
void print_test_result(const char* test_name, int success) {
    if (success) {
        printf("[ PASS ] %s\n", test_name);
    } else {
        printf("[ FAIL ] %s\n", test_name);
        exit(1);
    }
}

void test_creation_and_destruction() {
    int in_dim = 4;
    int out_dim = 3;
    
    Layer* layer = Layer_make(in_dim, out_dim, ACTIVATION_RELU);
    
    assert(layer != NULL);
    assert(layer->input_dim == in_dim);
    assert(layer->output_dim == out_dim);
    assert(layer->activation == ACTIVATION_RELU);
    assert(layer->weights != NULL);
    assert(layer->biases != NULL);
    assert(layer->d_weights != NULL);
    assert(layer->d_biases != NULL);
    
    // Caches must start as NULL
    assert(layer->inputs_cache == NULL);
    assert(layer->z_cache == NULL);
    assert(layer->a_cache == NULL);
    
    Layer_free(layer);
    print_test_result("Creation & Destruction Test", 1);
}

void test_forward_pass_none_activation() {
    // Setup a 2x3 input batch and a 3x2 layer
    int batch_size = 2;
    int in_dim = 3;
    int out_dim = 2;
    
    Layer* layer = Layer_make(in_dim, out_dim, ACTIVATION_NONE);
    
    // Setup deterministic weights manually
    // Weights shape: [3, 2]
    layer->weights->data[0] = 0.1f; layer->weights->data[1] = 0.2f;
    layer->weights->data[2] = 0.3f; layer->weights->data[3] = 0.4f;
    layer->weights->data[4] = 0.5f; layer->weights->data[5] = 0.6f;
    
    // Setup deterministic biases: [2]
    layer->biases->data[0] = 0.5f;
    layer->biases->data[1] = -0.5f;
    
    // Create inputs tensor shape: [2, 3]
    int in_shape[2] = {batch_size, in_dim};
    Tensor* inputs = Tensor_make(2, in_shape);
    inputs->data[0] = 1.0f; inputs->data[1] = 2.0f; inputs->data[2] = 3.0f;
    inputs->data[3] = 4.0f; inputs->data[4] = 5.0f; inputs->data[5] = 6.0f;
    
    // Create outputs placeholder destination shape: [2, 2]
    int out_shape[2] = {batch_size, out_dim};
    Tensor* outputs = Tensor_make(2, out_shape);
    
    // Execute Forward
    Layer_forward(layer, inputs, outputs);
    
    // Check dynamic allocations occurred inside forward
    assert(layer->inputs_cache != NULL);
    assert(layer->z_cache != NULL);
    assert(layer->a_cache != NULL);
    
    /* Expected values calculation:
       Row 0: 
         z0 = (1*0.1 + 2*0.3 + 3*0.5) + 0.5  = 2.2 + 0.5  = 2.7
         z1 = (1*0.2 + 2*0.4 + 3*0.6) - 0.5  = 2.8 - 0.5  = 2.3
       Row 1:
         z0 = (4*0.1 + 5*0.3 + 6*0.5) + 0.5  = 4.9 + 0.5  = 5.4
         z1 = (4*0.2 + 5*0.4 + 6*0.6) - 0.5  = 6.4 - 0.5  = 5.9
       Since Activation is NONE, outputs == Z
    */
    int success = 1;
    if (!is_close(outputs->data[0], 2.7f)) success = 0;
    if (!is_close(outputs->data[1], 2.3f)) success = 0;
    if (!is_close(outputs->data[2], 5.4f)) success = 0;
    if (!is_close(outputs->data[3], 5.9f)) success = 0;
    
    Tensor_free(inputs);
    Tensor_free(outputs);
    Layer_free(layer);
    
    print_test_result("Forward Pass (Linear/None) Numerical Verification", success);
}

void test_forward_pass_relu_activation() {
    int batch_size = 1;
    int in_dim = 2;
    int out_dim = 2;
    
    Layer* layer = Layer_make(in_dim, out_dim, ACTIVATION_RELU);
    layer->weights->data[0] = 1.0f; layer->weights->data[1] = -2.0f;
    layer->weights->data[2] = 3.0f; layer->weights->data[3] = 4.0f;
    layer->biases->data[0] = -10.0f; // Force this neuron output negative
    layer->biases->data[1] = 1.0f;
    
    int in_shape[2] = {batch_size, in_dim};
    Tensor* inputs = Tensor_make(2, in_shape);
    inputs->data[0] = 1.0f; inputs->data[1] = 1.0f;
    
    int out_shape[2] = {batch_size, out_dim};
    Tensor* outputs = Tensor_make(2, out_shape);
    
    Layer_forward(layer, inputs, outputs);
    
    /*
       Calculation:
       z0 = (1*1 + 1*3) - 10 = -6.0  --> ReLU(-6.0) -> 0.0
       z1 = (1*-2 + 1*4) + 1 = 3.0   --> ReLU(3.0)  -> 3.0
    */
    int success = 1;
    if (!is_close(outputs->data[0], 0.0f)) success = 0;
    if (!is_close(outputs->data[1], 3.0f)) success = 0;
    
    Tensor_free(inputs);
    Tensor_free(outputs);
    Layer_free(layer);
    
    print_test_result("Forward Pass (ReLU) Activation Verification", success);
}

void test_backward_pass_and_update() {
    int batch_size = 2;
    int in_dim = 2;
    int out_dim = 2;
    
    Layer* layer = Layer_make(in_dim, out_dim, ACTIVATION_NONE);
    
    // Identity-like setup
    layer->weights->data[0] = 1.0f; layer->weights->data[1] = 0.0f;
    layer->weights->data[2] = 0.0f; layer->weights->data[3] = 1.0f;
    Tensor_zero(layer->biases);
    
    int dims[2] = {batch_size, in_dim};
    Tensor* inputs = Tensor_make(2, dims);
    inputs->data[0] = 1.0f; inputs->data[1] = 2.0f;
    inputs->data[2] = 3.0f; inputs->data[3] = 4.0f;
    
    Tensor* outputs = Tensor_make(2, dims);
    Layer_forward(layer, inputs, outputs);
    
    // Simulate incoming gradient from upper layer (e.g., Loss gradient)
    Tensor* grad_output = Tensor_make(2, dims);
    grad_output->data[0] = 0.1f; grad_output->data[1] = 0.2f;
    grad_output->data[2] = 0.3f; grad_output->data[3] = 0.4f;
    
    Tensor* grad_input = Tensor_make(2, dims);
    
    // Run backward
    Layer_backward(layer, grad_output, grad_input);
    
    // Verify gradients are no longer strictly zero
    int grads_computed = 0;
    for(int i = 0; i < layer->d_weights->total_elements; i++) {
        if (layer->d_weights->data[i] != 0.0f) grads_computed = 1;
    }
    for(int i = 0; i < layer->d_biases->total_elements; i++) {
        if (layer->d_biases->data[i] != 0.0f) grads_computed = 1;
    }
    
    // Test parameters update function
    float old_w0 = layer->weights->data[0];
    Layer_update(layer, 0.1f);
    int weights_updated = (layer->weights->data[0] != old_w0);
    
    Tensor_free(inputs);
    Tensor_free(outputs);
    Tensor_free(grad_output);
    Tensor_free(grad_input);
    Layer_free(layer);
    
    print_test_result("Backward Pass & Parameter Update Structural Execution", grads_computed && weights_updated);
}

int main() {
    // Seed random state
    srand(42);
    
    printf("=== RUNNING LAYER UNIT TESTS ===\n");
    
    test_creation_and_destruction();
    test_forward_pass_none_activation();
    test_forward_pass_relu_activation();
    test_backward_pass_and_update();
    
    printf("=== ALL TESTS PASSED SUCCESSFULLY ===\n");
    return 0;
}
