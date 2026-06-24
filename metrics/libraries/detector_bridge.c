#include "detector_bridge.h"

#include "../../neural_network/libraries/network.h"
#include "../../neural_network/libraries/tensor.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

DetectorBridge* DetectorBridge_make(BridgeConfig config, uint32_t img_h, uint32_t img_w, uint32_t img_c) {
    DetectorBridge* bridge = (DetectorBridge*)malloc(sizeof(DetectorBridge));
    if (!bridge) return NULL;

    bridge->config = config;

    // Shape: [Batch_Size, Channels, Height, Width]
    int img_shape[4] = {(int)config.batch_size, (int)img_c, (int)img_h, (int)img_w};
    bridge->batch_images = Tensor_make(4, img_shape);

    // Target Matrix shape layout: [Batch_Size, Max_Targets, 6]
    // Each target item details: [class_id, x_min, y_min, x_max, y_max, valid_flag]
    int target_shape[3] = {(int)config.batch_size, (int)config.max_targets_per_img, 6};
    bridge->batch_targets = Tensor_make(3, target_shape);

    return bridge;
}



void DetectorBridge_free(DetectorBridge* bridge) {
    if (!bridge) return;
    if (bridge->batch_images) Tensor_free(bridge->batch_images);
    if (bridge->batch_targets) Tensor_free(bridge->batch_targets);
    free(bridge);
}






void DetectorBridge_prepare_input(DetectorBridge* bridge, Tensor* raw_frame_tensor, uint32_t batch_idx) {
    if (!bridge || !raw_frame_tensor) return;
    if (batch_idx >= bridge->config.batch_size) return; // Prevent out-of-bounds heap corruption

    uint32_t out_c = bridge->batch_images->shape[1];
    uint32_t out_h = bridge->batch_images->shape[2];
    uint32_t out_w = bridge->batch_images->shape[3];

    uint32_t in_h = raw_frame_tensor->shape[1];
    uint32_t in_w = raw_frame_tensor->shape[2];

    for (uint32_t c = 0; c < out_c; c++) {
        for (uint32_t y = 0; y < out_h; y++) {
            for (uint32_t x = 0; x < out_w; x++) {
                uint32_t src_x = (x * in_w) / out_w;
                uint32_t src_y = (y * in_h) / out_h;

                uint32_t src_idx = (c * in_h * in_w) + (src_y * in_w) + src_x;
                
                uint32_t dst_idx = (batch_idx * out_c * out_h * out_w) + (c * out_h * out_w) + (y * out_w) + x;

                bridge->batch_images->data[dst_idx] = raw_frame_tensor->data[src_idx] / 255.0f;
            }
        }
    }
}




//this needs working on:
    //1. I think it should be a Network* neural_network
    //2. needs to call the Network API if it wants to train I think it should go in the other program function
float DetectorBridge_train_step_standard(DetectorBridge* bridge, Tensor** input_images, BoundingBox2D* ground_truth_boxes, uint32_t* box_counts_per_img, CameraIntrinsics intrinsics, void* neural_network_instance) {
    if (!bridge || !input_images || !box_counts_per_img  || !ground_truth_boxes || !neural_network_instance) return 0.0f;

    uint32_t total_target_elements = bridge->config.batch_size * bridge->config.max_targets_per_img * 6;

    // Reset target padding matrix
    memset(bridge->batch_targets->data, 0, total_target_elements * sizeof(float));

    for (uint32_t b = 0; b < bridge->config.batch_size; b++) {
        DetectorBridge_prepare_input(bridge, input_images[b], b);
    }

    uint32_t global_box_idx = 0;
    float img_w = intrinsics.cx * 2.0f;
    float img_h = intrinsics.cy * 2.0f;

    // Outer Loop: Step through each batch index slot
    for (uint32_t b = 0; b < bridge->config.batch_size; b++) {

        uint32_t boxes_for_this_image = box_counts_per_img[b];
        // Inner Loop: Format and pad target allocations
        for (uint32_t t = 0; t < bridge->config.max_targets_per_img; t++) {
            if (t >= boxes_for_this_image) {
                break; 
            }

            uint32_t target_base = (b * bridge->config.max_targets_per_img * 6) + (t * 6);

            // Assuming image dimensions are derived via doubling the optical centers (2 * cx, 2 * cy)

            float norm_x_min = ground_truth_boxes[global_box_idx].x_min / img_w;
            float norm_y_min = ground_truth_boxes[global_box_idx].y_min / img_h;
            float norm_x_max = ground_truth_boxes[global_box_idx].x_max / img_w;
            float norm_y_max = ground_truth_boxes[global_box_idx].y_max / img_h;

            bridge->batch_targets->data[target_base + 0] = (float)ground_truth_boxes[global_box_idx].class_id;
            bridge->batch_targets->data[target_base + 1] = norm_x_min;
            bridge->batch_targets->data[target_base + 2] = norm_y_min;
            bridge->batch_targets->data[target_base + 3] = norm_x_max;
            bridge->batch_targets->data[target_base + 4] = norm_y_max;
            bridge->batch_targets->data[target_base + 5] = 1.0f; // Set presence flag active

            global_box_idx++;
        }
    }

    float loss = 0.0f;
    //I think I want to call it outside this function in the main but I am not sure yet
    // loss = NeuralNetwork_forward_backward_update(neural_network_instance, bridge->batch_images, bridge->batch_targets);
    return loss;
}





float DetectorBridge_train_step_pseudo(DetectorBridge* bridge, AutoLabeler* labeler, CameraIntrinsics intrinsics, void* neural_network_instance) {
    // Null safety guard condition: if no labeler is provided, bypass target compilation
    if (!bridge || !labeler || !neural_network_instance) return 0.0f;

    uint32_t batch_cap = bridge->config.batch_size;
    uint32_t max_box_alloc = batch_cap * bridge->config.max_targets_per_img;

    Tensor** sampled_images = (Tensor**)malloc(batch_cap * sizeof(Tensor*));
    uint32_t* box_counts = (uint32_t*)malloc(batch_cap * sizeof(uint32_t));
    BoundingBox2D* harvested_boxes = (BoundingBox2D*)malloc(max_box_alloc * sizeof(BoundingBox2D));

    if (!sampled_images || !harvested_boxes) {
        if (sampled_images) free(sampled_images);
        if (harvested_boxes) free(harvested_boxes);
        if (box_counts) free(box_counts);
        return 0.0f;
    }

    // Call your auto_labeler to pull targets out of the circular buffer
    uint32_t absolute_box_count = AutoLabeler_generate_autoregressive_targets(labeler, sampled_images, harvested_boxes, box_counts, batch_cap);    
   
    float loss = 0.0f;

    // If targets are found, forward them to the standard normalization function
    if (absolute_box_count > 0) {
        loss = DetectorBridge_train_step_standard(bridge, sampled_images, harvested_boxes, box_counts, intrinsics, neural_network_instance);
    }

    free(sampled_images);
    free(harvested_boxes);
    free(box_counts);
    return loss;
}


