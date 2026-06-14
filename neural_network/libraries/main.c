#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "layer.h"
#include "tensor.h"

// Helper to print basic tensor metadata and data values
void print_tensor_summary(const char* label, const Tensor* t) {
    if (!t) {
        printf("%s: NULL\n", label);
        return;
    }
    printf("%s: Shape [", label);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], (i == t->ndim - 1) ? "" : ", ");
    }
    printf("], Total Elements: %d\n  Data: ", t->total_elements);
    
    // Print up to the first 8 elements so it doesn't flood the terminal
    int print_limit = t->total_elements > 8 ? 8 : t->total_elements;
    for (int i = 0; i < print_limit; i++) {
        printf("%.4f ", t->data[i]);
    }
    if (t->total_elements > 8) printf("... ");
    printf("\n\n");
}

int main() {
    printf("=== INITIALIZING MINI-NETWORK TEST ===\n\n");

    // 1. Create the central network manager
    Network* net = Network_make();
    if (!net) {
        printf("Failed to create network.\n");
        return 1;
    }

    // 2. Build the structural layers using our framework constructors
    // Layer 0: Convolution (1 Input Channel, 1 Output Channel, Kernel=3, Stride=1, Padding=1)
    // Using padding=1 keeps our spatial output dimensions at a stable 4x4
     Layer* conv_layer = Layer_make_convolution(1, 1, 4, 4, 3, 1, 1, ACTIVATION_NONE);
Network_add_convolution(net, conv_layer);// <-- Make sure this matches your framework's exact conv builder name

    // Layer 1: Max Pooling (Pool size=2, Stride=2) -> Drops 4x4 down to 2x2
Layer* pool_layer = Layer_make_pooling(1, 4, 4, 2, 2);
Network_add_pooling(net, pool_layer);      // Make sure this matches your pooling builder name

    // Layer 2: Upsample by a factor of 2 -> Stretches 2x2 back out to 4x4
    Network_add_upsample(net, 2);

    // Layer 3: Skip Addition -> Adds current input (Layer 2 output) to Layer 0's output
    Network_add_addition(net, 0); 

    printf("Network structure assembled successfully with %d layers.\n\n", net->num_layers);

    // 3. Prepare Dummy Input Data (A 1x1x4x4 Tensor)
    int input_shape[4] = {1, 1, 4, 4};
    Tensor* input = Tensor_make(4, input_shape);
    
    // Fill the input tensor with predictable values (1.0, 2.0, 3.0, ...)
    for (int i = 0; i < input->total_elements; i++) {
        input->data[i] = (float)(i + 1);
    }
    print_tensor_summary("Network Input Tensor", input);

    // 4. Run the Forward Pass
    printf("--- Executing Network_forward ---\n");
    Tensor* output = Network_forward(net, input);
    print_tensor_summary("Network Output (Prediction)", output);

    // 5. Generate Artificial Targets and Compute Loss
    // Let's assume the ideal target values are half of our index sequence
    Tensor* targets = Tensor_make(4, input_shape);
    for (int i = 0; i < targets->total_elements; i++) {
        targets->data[i] = (float)(i + 1) * 0.5f;
    }

    Tensor* loss_gradient = Tensor_make(4, input_shape);
    float loss = Network_compute_loss(net, output, targets, LOSS_MSE, loss_gradient);
    printf("Computed MSE Loss: %.6f\n", loss);
    print_tensor_summary("Loss Gradient (Initial Upstream)", loss_gradient);

    // 6. Run the Backward Pass
    printf("--- Executing Network_backward ---\n");
    Tensor* grad_input = Tensor_make(4, input_shape); // To hold derivative with respect to original input
    Network_backward(net, loss_gradient, grad_input);
    
    print_tensor_summary("Gradient backpropagated all the way to Input", grad_input);
    print_tensor_summary("Gradients stored at Layer 0 (Conv)", net->Layer_gradients[0]);

    // 7. Update the Network Parameters
    printf("--- Executing Network_update ---\n");
    float learning_rate = 0.01f;
    Network_update(net, learning_rate);
    printf("Weights updated successfully without crashing!\n\n");

    // 8. Clean up all allocated memory safely
    printf("=== CLEANING UP MEMORY ===\n");
    Tensor_free(input);
    Tensor_free(targets);
    Tensor_free(loss_gradient);
    Tensor_free(grad_input);
    Network_free(net);

    printf("Test complete. Memory freed perfectly.\n");
    return 0;
}
