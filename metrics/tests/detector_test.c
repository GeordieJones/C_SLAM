#include "../libraries/detector_bridge.h"
#include "../../neural_network/libraries/tensor.h" // Import your true project tensor library
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

/* ========================================================================= */
/* MOCK IMPLEMENTATIONS FOR ISOLATED TESTING                                 */
/* ========================================================================= */

// Mock AutoLabeler call since we are verifying the standard path mechanics
uint32_t AutoLabeler_generate_autoregressive_targets(AutoLabeler* labeler, Tensor** imgs, BoundingBox2D* boxes, uint32_t* counts, uint32_t cap) {
    (void)labeler; (void)imgs; (void)boxes; (void)counts; (void)cap;
    return 0; // Stubbed out for isolation
}

/* ========================================================================= */
/* CORE UNIT TEST RUNNER                                                     */
/* ========================================================================= */
void verify_detector_bridge_matrices() {
    printf("[START] Beginning DetectorBridge validation tests...\n");

    // 1. Construct explicit configurations
    BridgeConfig config = {
        .batch_size = 2,
        .max_targets_per_img = 4, 
        .net_input_channels = 3,
        .net_input_width = 8,
        .net_input_height = 8,
        .class_loss_weight = 1.0f,
        .box_loss_weight = 1.0f,
        .apply_data_augmentation = false
    };
    
    CameraIntrinsics intrinsics = {
        .fx = 320.0f, .fy = 240.0f,
        .cx = 320.0f, .cy = 240.0f // Explicitly maps to a 640x480 virtual viewport
    };

    // Run constructor instantiations
    DetectorBridge* bridge = DetectorBridge_make(config, config.net_input_height, config.net_input_width, config.net_input_channels);
    assert(bridge != NULL);
    printf("[PASS] Lifecycle instantiation allocations successfully completed.\n");

    // 2. Mock raw camera unformatted streams (640x480 resolution input) using your native Tensor library signatures
    int raw_cam_shape[3] = {3, 480, 640}; // Changed to signed int to match Tensor_make expectancies
    Tensor* raw_frame_0 = Tensor_make(3, raw_cam_shape);
    Tensor* raw_frame_1 = Tensor_make(3, raw_cam_shape);

    // Calculate element bounds based on shapes directly to fill elements safely
    uint32_t raw_element_count = 3 * 480 * 640;
    for (uint32_t i = 0; i < raw_element_count; i++) {
        raw_frame_0->data[i] = 255.0f;
        raw_frame_1->data[i] = 255.0f;
    }

    Tensor* batch_input_images[2] = { raw_frame_0, raw_frame_1 };

    // 3. Formulate uneven bounding box target data blocks
    uint32_t mock_box_counts[2] = { 2, 1 };
    BoundingBox2D mock_ground_truths[3] = {
        // Frame 0 Targets
        { .class_id = 12, .x_min = 0.0f,   .y_min = 0.0f,   .x_max = 320.0f, .y_max = 240.0f },
        { .class_id = 7,  .x_min = 320.0f, .y_min = 240.0f, .x_max = 640.0f, .y_max = 480.0f },
        // Frame 1 Target
        { .class_id = 99, .x_min = 160.0f, .y_min = 120.0f, .x_max = 480.0f, .y_max = 360.0f }
    };

    // 4. Run the structural compilation pipeline standard training pass
    int structural_network_mock = 42; 
    printf("[EXEC] Processing image translation and normalization channels...\n");
    DetectorBridge_train_step_standard(bridge, batch_input_images, mock_ground_truths, mock_box_counts, intrinsics, &structural_network_mock);

    /* ========================================================================= */
    /* MATRIX VERIFICATION ASSERTIONS                                            */
    /* ========================================================================= */
    
    // Check pixel normalizations
    uint32_t total_image_elements = config.batch_size * config.net_input_channels * config.net_input_height * config.net_input_width;
    assert(fabs(bridge->batch_images->data[0] - 1.0f) < 1e-5);
    assert(fabs(bridge->batch_images->data[total_image_elements - 1] - 1.0f) < 1e-5);
    printf("[PASS] Image memory formatting transformations successfully scaled down to [0.0, 1.0].\n");

    // Frame 0, Bounding Box 0 checking validations
    assert((uint32_t)bridge->batch_targets->data[0] == 12);
    assert(fabs(bridge->batch_targets->data[1] - 0.00f) < 1e-4); 
    assert(fabs(bridge->batch_targets->data[3] - 0.50f) < 1e-4); 
    assert(fabs(bridge->batch_targets->data[5] - 1.00f) < 1e-4); 

    // Frame 0, Bounding Box 1 checking validations
    assert((uint32_t)bridge->batch_targets->data[6] == 7);
    assert(fabs(bridge->batch_targets->data[7] - 0.50f) < 1e-4); 
    assert(fabs(bridge->batch_targets->data[9] - 1.00f) < 1e-4); 
    assert(fabs(bridge->batch_targets->data[11] - 1.00f) < 1e-4); 

    // Frame 0, Bounding Box 2 checking validations (Should be zero-padded space)
    assert(fabs(bridge->batch_targets->data[12]) < 1e-5);
    assert(fabs(bridge->batch_targets->data[17]) < 1e-5); 
    printf("[PASS] Structural padding boundary constraints verified for Frame 0 trailing empty blocks.\n");

    // Frame 1, Bounding Box 0 checking validations
    uint32_t frame_1_box_0_base = 24; // (b=1 * max_targets=4 * 6 metrics)
    assert((uint32_t)bridge->batch_targets->data[frame_1_box_0_base + 0] == 99);
    assert(fabs(bridge->batch_targets->data[frame_1_box_0_base + 1] - 0.25f) < 1e-4); 
    assert(fabs(bridge->batch_targets->data[frame_1_box_0_base + 3] - 0.75f) < 1e-4); 
    assert(fabs(bridge->batch_targets->data[frame_1_box_0_base + 5] - 1.00f) < 1e-4); 
    printf("[PASS] Batch index 1 structural targets successfully processed with zero alignment collisions.\n");

    // Clean up targets via project native tensor methods
    Tensor_free(raw_frame_0);
    Tensor_free(raw_frame_1);
    DetectorBridge_free(bridge);

    printf("[SUCCESS] All DetectorBridge internal translation validation operations passed flawlessly!\n");
}

int main(void) {
    verify_detector_bridge_matrices();
    return 0;
}
