#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Helper to print test headers with timing info
void print_timing(const char* test_name, clock_t start, clock_t end) {
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf(">>> %s completed in %.4f ms\n\n", test_name, cpu_time_used);
}

// Simple dummy operation for testing Tensor_map
float square_element(float val) {
    return val * val;
}

int main() {
    // Seed random number generator for Tensor_rand
    srand((unsigned int)time(NULL));
    
    clock_t start, end;
    printf("=========================================\n");
    printf("        STARTING TENSOR SYSTEM TESTS      \n");
    printf("=========================================\n\n");

    // -------------------------------------------------------------
    // Test 1: Tensor Creation and Allocation
    // -------------------------------------------------------------
    printf("--- Test 1: Tensor_make (3D Tensor 2x3x4) ---\n");
    int shape_3d[] = {2, 3, 4};
    
    start = clock();
    Tensor* t1 = Tensor_make(3, shape_3d);
    end = clock();
    
    if (t1 == NULL) {
        printf("ERROR: Tensor_make returned NULL!\n");
        return 1;
    }
    Tensor_print(t1);
    print_timing("Tensor_make", start, end);

    // -------------------------------------------------------------
    // Test 2: Basic Initialization (Zero & Random)
    // -------------------------------------------------------------
    printf("--- Test 2: Tensor_zero and Tensor_rand ---\n");
    start = clock();
    Tensor_zero(t1);
    end = clock();
    printf("Tensor after Tensor_zero (First element: %.1f)\n", t1->data[0]);
    print_timing("Tensor_zero", start, end);

    start = clock();
    Tensor_rand(t1);
    end = clock();
    printf("Tensor after Tensor_rand (First 3 elements: %.3f, %.3f, %.3f)\n", 
           t1->data[0], t1->data[1], t1->data[2]);
    print_timing("Tensor_rand", start, end);

    // -------------------------------------------------------------
    // Test 3: Coordinate Access Math
    // -------------------------------------------------------------
    printf("--- Test 3: Tensor_at (Coordinate Access) ---\n");
    int coords[] = {1, 2, 3}; // Max index boundaries for 2x3x4
    start = clock();
    float val = Tensor_at(t1, coords);
    end = clock();
    printf("Value at coordinate [1, 2, 3]: %.4f\n", val);
    print_timing("Tensor_at", start, end);

    // -------------------------------------------------------------
    // Test 4: Math Reductions (Sum and Scaling)
    // -------------------------------------------------------------
    printf("--- Test 4: Tensor_scale & Tensor_sum ---\n");
    start = clock();
    Tensor_scale(t1, 2.0f);
    end = clock();
    print_timing("Tensor_scale", start, end);

    start = clock();
    float total_sum = Tensor_sum(t1);
    end = clock();
    printf("Total integrated sum of all elements: %.4f\n", total_sum);
    print_timing("Tensor_sum", start, end);

    // -------------------------------------------------------------
    // Test 5: Structural Operations (Cloning and Copying)
    // -------------------------------------------------------------
    printf("--- Test 5: Tensor_clone & Tensor_copy ---\n");
    start = clock();
    Tensor* t1_clone = Tensor_clone(t1);
    end = clock();
    print_timing("Tensor_clone", start, end);

    start = clock();
    Tensor_copy(t1, t1_clone);
    end = clock();
    print_timing("Tensor_copy", start, end);
    Tensor_free(t1_clone);

    // -------------------------------------------------------------
    // Test 6: Statistical Extrema Finding
    // -------------------------------------------------------------
    printf("--- Test 6: Tensor_min_max ---\n");
    // Explicitly seed extrema values to ensure accurate location tracking
    t1->data[5] = -999.0f;
    t1->data[12] = 999.0f;
    
    start = clock();
    MinMax mm = Tensor_min_max(t1);
    end = clock();
    printf("Min Found: %.1f at position [%d]\n", mm.min, mm.min_loc);
    printf("Max Found: %.1f at position [%d]\n", mm.max, mm.max_loc);
    print_timing("Tensor_min_max", start, end);

    // -------------------------------------------------------------
    // Test 7: Pointwise Transforms (Mapping Engine)
    // -------------------------------------------------------------
    printf("--- Test 7: Tensor_map ---\n");
    start = clock();
    Tensor_map(t1, square_element);
    end = clock();
    printf("First mapped squared element check: %.4f\n", t1->data[5]);
    print_timing("Tensor_map", start, end);

    // -------------------------------------------------------------
    // Test 8: Dimensional Reshaping
    // -------------------------------------------------------------
    printf("--- Test 8: Tensor_reshape (2x3x4 -> 4x6) ---\n");
    int new_shape[] = {4, 6};
    start = clock();
    Tensor_reshape(t1, new_shape, 2);
    end = clock();
    Tensor_print(t1);
    print_timing("Tensor_reshape", start, end);

    // -------------------------------------------------------------
    // Test 9: Transposition Engines (Eager & Lazy)
    // -------------------------------------------------------------
    printf("--- Test 9: Tensor Transpositions ---\n");
    start = clock();
    Tensor_lazy_transpose(t1, 0, 1);
    end = clock();
    printf("Lazy transpose structure output:\n");
    Tensor_print(t1);
    print_timing("Tensor_lazy_transpose", start, end);

    start = clock();
    Tensor* t_eager = Tensor_eager_transpose(t1, 0, 1);
    end = clock();
    print_timing("Tensor_eager_transpose", start, end);
    Tensor_free(t_eager);

    // -------------------------------------------------------------
    // Test 10: Matrix Multiplication and Linear Algebra
    // -------------------------------------------------------------
    printf("--- Test 10: Tensor_matmul (2D Matrix Multiply) ---\n");
    int shape_a[] = {2, 3};
    int shape_b[] = {3, 2};
    Tensor* mat_a = Tensor_make(2, shape_a);
    Tensor* mat_b = Tensor_make(2, shape_b);
    
    // Fill sample uniform values
    for(int i = 0; i < mat_a->total_elements; i++) mat_a->data[i] = 2.0f;
    for(int i = 0; i < mat_b->total_elements; i++) mat_b->data[i] = 3.0f;

    start = clock();
    Tensor* mat_c = Tensor_matmul(mat_a, mat_b);
    end = clock();
    
    if (mat_c) {
        printf("Output matrix dimensions expected: [%d, %d]\n", mat_c->shape[0], mat_c->shape[1]);
        printf("Calculated index [0]: %.2f (Expected: 18.00)\n", mat_c->data[0]);
        Tensor_free(mat_c);
    }
    print_timing("Tensor_matmul", start, end);
    
    // -------------------------------------------------------------
    // Test 11: Vector Operations (Dot Product and Hadamard)
    // -------------------------------------------------------------
    printf("--- Test 11: Vector Dot and Hadamard ---\n");
    start = clock();
    float dot_prod = Tensor_vector_dot(mat_a, mat_a);
    end = clock();
    printf("Vector Dot Product result: %.2f\n", dot_prod);
    print_timing("Tensor_vector_dot", start, end);

    start = clock();
    Tensor_hadamard(mat_a, mat_a);
    end = clock();
    printf("Hadamard output index [0]: %.2f\n", mat_a->data[0]);
    print_timing("Tensor_hadamard", start, end);

    Tensor_free(mat_a);
    Tensor_free(mat_b);

    // -------------------------------------------------------------
    // Test 12: Tensor Slicing & Merging (Tensor_addon)
    // -------------------------------------------------------------
    printf("--- Test 12: Tensor_addon (Concatenation) ---\n");
    int base_shape[] = {2, 2};
    Tensor* base_t = Tensor_make(2, base_shape);
    Tensor* addon_t = Tensor_make(2, base_shape);
    
    for(int i=0; i<4; i++) {
        base_t->data[i] = 1.0f;
        addon_t->data[i] = 5.0f;
    }

    start = clock();
    Tensor* concat_t = Tensor_addon(base_t, addon_t, 0); // Concat along rows
    end = clock();

    if (concat_t) {
        printf("Concatenated Shape: [%d, %d] (Expected [4,2])\n", concat_t->shape[0], concat_t->shape[1]);
        printf("Element block verification: Top Row=%.1f, Bottom Row=%.1f\n", concat_t->data[0], concat_t->data[6]);
        Tensor_free(concat_t);
    }
    print_timing("Tensor_addon", start, end);

    Tensor_free(base_t);
    Tensor_free(addon_t);
    Tensor_free(t1);

    printf("=========================================\n");
    printf("      ALL RUNTIME TESTS EXECUTED         \n");
    printf("=========================================\n");

    return 0;
}
