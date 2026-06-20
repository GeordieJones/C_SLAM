#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "../libaries/tensor.h"
#include "../libaries/activation.h"

// Helper function to print tensors nicely during testing
void print_test_tensor(const char* name, const Tensor* t) {
    int rows = t->shape[0];
    int cols = (t->ndim > 1) ? t->shape[1] : 1;
    
    printf("--- %s (Dims: %d, Total Elements: %d) ---\n", name, t->ndim, t->total_elements);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%7.4f ", t->data[r * cols + c]);
        }
        printf("\n");
    }
    printf("\n");
}

int approx_equal(float a, float b) {
    return fabsf(a - b) < 0.0001f;
}

int main() {
    printf("=========================================\n");
    printf("   RUNNING ACTIVATION LIBRARY TESTS      \n");
    printf("=========================================\n\n");

    int dims[2] = {1, 4};
    
    Tensor* input    = Tensor_make(2, dims);
    Tensor* output   = Tensor_make(2, dims);
    Tensor* grad_out = Tensor_make(2, dims);
    Tensor* grad_in  = Tensor_make(2, dims);

    assert(input != NULL && output != NULL && grad_out != NULL && grad_in != NULL);

    input->data[0] = -2.0f;
    input->data[1] =  0.0f;
    input->data[2] =  1.0f;
    input->data[3] =  4.0f;

    grad_out->data[0] = 1.0f; 
    grad_out->data[1] = 1.0f; 
    grad_out->data[2] = 1.0f; 
    grad_out->data[3] = 1.0f;

    print_test_tensor("Initial Input Vector", input);

    // -------------------------------------------------------------------------
    // 2. TEST ReLU & ReLU DERIVATIVE
    // -------------------------------------------------------------------------
    printf("[TEST] Testing ReLU Forward and Derivative...\n");
    relu(input, output);
    print_test_tensor("ReLU Output (Expected: 0, 0, 1, 4)", output);
    assert(approx_equal(output->data[0], 0.0f));
    assert(approx_equal(output->data[2], 1.0f));

    relu_derivative(input, grad_out, grad_in);
    print_test_tensor("ReLU Derivative (Expected: 0, 0, 1, 1)", grad_in);
    assert(approx_equal(grad_in->data[0], 0.0f));
    assert(approx_equal(grad_in->data[3], 1.0f));

    // -------------------------------------------------------------------------
    // 3. TEST LEAKY ReLU & DERIVATIVE
    // -------------------------------------------------------------------------
    printf("[TEST] Testing Leaky ReLU Forward and Derivative...\n");
    leaky_relu(input, output);
    print_test_tensor("Leaky ReLU Output (Expected: -0.02, 0, 1, 4)", output);
    assert(approx_equal(output->data[0], -0.02f));

    leaky_relu_derivative(input, grad_out, grad_in);
    print_test_tensor("Leaky ReLU Derivative (Expected: 0.01, 0.01, 1, 1)", grad_in);
    assert(approx_equal(grad_in->data[0], 0.01f));

    // -------------------------------------------------------------------------
    // 4. TEST SIGMOID & DERIVATIVE
    // -------------------------------------------------------------------------
    printf("[TEST] Testing Sigmoid Forward and Derivative...\n");
    sigmoid(input, output);
    print_test_tensor("Sigmoid Output (Expected values between 0 and 1)", output);
    assert(approx_equal(output->data[1], 0.5f)); 

    sigmoid_derivative(output, grad_out, grad_in);
    print_test_tensor("Sigmoid Derivative (Expected: output * (1 - output))", grad_in);
    assert(approx_equal(grad_in->data[1], 0.25f));

    // -------------------------------------------------------------------------
    // 5. TEST SOFTMAX & COMBINED CROSS-ENTROPY DERIVATIVE
    // -------------------------------------------------------------------------
    printf("[TEST] Testing Softmax and Cross-Entropy Shortcut...\n");
    
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    input->data[3] = 4.0f;
    print_test_tensor("New Softmax Logits Input", input);

    softmax(input, output);
    print_test_tensor("Softmax Probabilities", output);
    
    float sum = output->data[0] + output->data[1] + output->data[2] + output->data[3];
    printf("Softmax Probabilities Sum: %f (Expected: 1.0000)\n\n", sum);
    assert(approx_equal(sum, 1.0f));

    Tensor* targets = Tensor_make(2, dims);
    assert(targets != NULL);
    Tensor_zero(targets); 
    targets->data[2] = 1.0f; 
    print_test_tensor("One-Hot Target Vector (Class Index 2)", targets);

    softmax_cross_entropy_derivative(output, targets, grad_in);
    print_test_tensor("Softmax Loss Derivative (Expected: Probs - Targets)", grad_in);
    
    assert(grad_in->data[2] < 0.0f);
    assert(grad_in->data[3] > 0.0f);

    // -------------------------------------------------------------------------
    // CLEANUP
    // -------------------------------------------------------------------------
    Tensor_free(input);
    Tensor_free(output);
    Tensor_free(grad_out);
    Tensor_free(grad_in);
    Tensor_free(targets);

    printf("=========================================\n");
    printf("   ALL TESTS PASSED SUCCESSFULLY!        \n");
    printf("=========================================\n");

    return 0;
}
