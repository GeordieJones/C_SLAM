#ifndef DETECTOR_BRIDGE_H
#define DETECTOR_BRIDGE_H

#include "../../computer_vision/libraries/auto_labeler.h"
#include "../../neural_network/libraries/tensor.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Configuration parameters defining the network input boundaries
 * and training constraint rules.
 */
typedef struct {
    uint32_t batch_size;          // Number of training frames compiled per optimization pass
    uint32_t max_targets_per_img; // Max padded rows allocated for 2D bounding boxes per frame
    
    // Extents of the camera frame the neural network expects
    uint32_t net_input_width;     
    uint32_t net_input_height;    
    uint32_t net_input_channels;  

    // Hyperparameters for network tuning optimization
    float class_loss_weight;      // Loss coefficient multiplier for classification errors
    float box_loss_weight;        // Loss coefficient multiplier for localization errors (IoU / Smooth L1)
    bool apply_data_augmentation; // Toggle switch for injecting stochastic spatial noise/flips
} BridgeConfig;

/**
 * @brief The Translator Core structure managing the input pipeline for inferences
 * and target compilation structures for backpropagation training.
 */
typedef struct {
    BridgeConfig config;
    
    // Pre-allocated tensor blocks ensuring fixed memory footprints during loops
    Tensor* batch_images;   // Structured memory for neural network execution: [Batch, Channels, Height, Width]
    Tensor* batch_targets;  // Uniform target matrix block for the loss function: [Batch, Max_Targets, 6]
                            // Element details inside: [class_id, norm_x_min, norm_y_min, norm_x_max, norm_y_max, valid_flag]
} DetectorBridge;

/* ========================================================================= */
/* LIFECYCLE MANAGEMENT IMPLEMENTATIONS                */
/* ========================================================================= */

/**
 * @brief Instantiates and allocates the memory tensors for a bidirectional network bridge.
 * @param config Structural configurations and dimension shapes defining the system's runtime parameters.
 * @return Pointer to an initialized DetectorBridge instance, or NULL if allocations fail.
 */

DetectorBridge* DetectorBridge_make(BridgeConfig config, uint32_t img_h, uint32_t img_w, uint32_t img_c);
/**
 * @brief Safely frees all pre-allocated internal tensors and destroys the interface bridge instance.
 * @param bridge Pointer to the active interface engine module to release.
 */
void DetectorBridge_free(DetectorBridge* bridge);

/* ========================================================================= */
/* THE INFERENCE ROUTE (DATA IN PIPELINE)             */
/* ========================================================================= */

/**
 * @brief Acts as the raw input sensor filter. Reorders memory layouts, crops/resamples
 * dimensions, and normalizes pixels into standard fractional ranges.
 * @note Pushes the cleaned frame into array slot index 0 of bridge->batch_images,
 * making it fully prepared for immediate neural network forward validation passes.
 * @param bridge Pointer to the active translator interface module.
 * @param raw_frame_tensor The raw unformatted tensor directly out from the hardware camera decoder.
 * Expected shape layout: [Channels, Native_Height, Native_Width]
 */
void DetectorBridge_prepare_input(DetectorBridge* bridge, Tensor* raw_frame_tensor, uint32_t batch_idx);

/* ========================================================================= */
/* THE TRAINING ROUTE (DATA BACK PIPELINE)            */
/* ========================================================================= */


float DetectorBridge_train_step_standard(DetectorBridge* bridge, Tensor** input_images, BoundingBox2D* ground_truth_boxes, uint32_t* box_count, CameraIntrinsics intrinsics, void* neural_network_instance); 

float DetectorBridge_train_step_pseudo(DetectorBridge* bridge, AutoLabeler* labeler, CameraIntrinsics intrinsics, void* neural_network_instance);

#endif // DETECTOR_BRIDGE_H       














