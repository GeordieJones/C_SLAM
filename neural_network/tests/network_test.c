#include "network.h"
#include "layer.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Simple helper to print success banners
void print_suite_result(const char* test_name, int success) {
    if (success) {
        printf("[ PASS ] %s\n", test_name);
    } else {
        printf("[ FAIL ] %s\n", test_name);
        exit(1);
    }
}

void test_network_lifecycle_and_learning() {
    printf("--- Test Scenario: Network Building, Training Loop, and Convergence ---\n");

    // 1. Initialize an empty network shell
    Network* net = Network_make();
    assert(net != NULL);
    assert(net->num_layers == 0);
    printf("Network container successfully created.\n");

    // 2. Build Architecture: Input(2) -> Hidden Expansion(4) -> Output Contraction(2)
    Layer* hidden_layer = Layer_make(2, 4, ACTIVATION_RELU);
    Layer* output_layer = Layer_make(4, 2, ACTIVATION_SOFTMAX);

    Network_add_dense(net, hidden_layer);
    assert(net->num_layers == 1);

    // This second add will trigger the internal dimension/shape validation code
    Network_add_dense(net, output_layer);
    assert(net->num_layers == 2);
    printf("Layer structures dynamically appended and verified.\n");

    // 3. Create a static dummy mini-batch dataset (Batch size = 3, Features = 2)
    int batch_size = 3;
    int input_shape[2] = {batch_size, 2};
    Tensor* inputs = Tensor_make(2, input_shape);
    
    // Fill inputs with arbitrary non-zero features
    inputs->data[0] = 0.5f;  inputs->data[1] = -0.2f;
    inputs->data[2] = 0.8f;  inputs->data[3] = 0.9f;
    inputs->data[4] = -0.1f; inputs->data[5] = 0.4f;

    // Create One-Hot encoded Target classifications (Batch size = 3, Categories = 2)
    int target_shape[2] = {batch_size, 2};
    Tensor* targets = Tensor_make(2, target_shape);
    
    targets->data[0] = 1.0f; targets->data[1] = 0.0f; // Sample 0: Class 0
    targets->data[2] = 0.0f; targets->data[3] = 1.0f; // Sample 1: Class 1
    targets->data[4] = 1.0f; targets->data[5] = 0.0f; // Sample 2: Class 0

    // Pre-allocate the external loss gradient tracking buffer
    Tensor* loss_gradient_buffer = Tensor_make(2, target_shape);
    
    // Pre-allocate a final placeholder tensor to capture input gradients if needed
    Tensor* raw_input_gradients = Tensor_make(2, input_shape);

    // 4. Run an iterative optimizations sequence loop to check convergence
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    float learning_rate = 0.1f;
    int steps = 5;

    printf("\nBeginning Optimization Step Testing Loops:\n");
    for (int step = 0; step < steps; step++) {
        // A. Forward Propagate
        Tensor* predictions = Network_forward(net, inputs);
        assert(predictions != NULL);
        assert(predictions->shape[0] == batch_size);
        assert(predictions->shape[1] == 2);

        // B. Compute Loss and Fill Initial Gradients
        float loss = Network_compute_loss(net, predictions, targets, LOSS_CROSS_ENTROPY, loss_gradient_buffer);
        printf("   Step %d -> Categorical Cross Entropy Loss: %f\n", step, loss);

        if (step == 0) initial_loss = loss;
        final_loss = loss;

        // C. Backward Propagate
        Network_backward(net, loss_gradient_buffer, raw_input_gradients);

        // D. Update weights via Gradient Descent
        Network_update(net, learning_rate);
    }

    // 5. Assertions and Verifications
    int metrics_passed = 1;
    
    // Mathematical Convergence Verification: The loss MUST decrease over time if weights are updating!
    if (final_loss >= initial_loss) {
        printf("ERROR: Network parameters are not optimizing. Loss stalled or increased.\n");
        metrics_passed = 0;
    } else {
        printf("\nMathematical Optimization Verified: Loss decreased smoothly from %f down to %f.\n", initial_loss, final_loss);
    }

    // 6. Memory Clean up teardown check
    Tensor_free(inputs);
    Tensor_free(targets);
    Tensor_free(loss_gradient_buffer);
    Tensor_free(raw_input_gradients);
    
    // Destroys arrays, layers, intermediate caches, and the network struct shell
    Network_free(net);
    printf("Network structure safely deallocated.\n");

    print_suite_result("Full Network Integration Optimization Suite", metrics_passed);
}

int main() {
    // Seed the random state generator for deterministic weights initialization
    srand(1337);

    printf("===========================================\n");
    printf("      RUNNING NETWORK GRAPH INTEGRATION     \n");
    printf("===========================================\n");

    test_network_lifecycle_and_learning();

    printf("===========================================\n");
    printf("       ALL NETWORK SYSTEM TESTS PASSED     \n");
    printf("===========================================\n");
    return 0;
}
