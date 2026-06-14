#ifndef AUTO_LABELER_H
#define AUTO_LABELER_H

#include "slam_math.h"
#include "scene_graph.h"
#include "image_decoders.h"
#include "../../neural_network/libraries/tensor.h"

//Must be a power of two do to buffer optimizations
#define MAX_KEYFRAME_HISTORY 32
#define MAX_LABELS_PER_FRAME 32



typedef struct {
    uint32_t frame_id;
    Tensor* rgb_image;          // Layout: [3, H, W] raw sensor image
    Tensor* depth_map;          // Layout: [1, H, W] matching depth map
    Transform3D camera_pose;    // World space camera position at this timestamp
    BoundingBox2D labels[MAX_LABELS_PER_FRAME]; // 2D detections captured this frame
    uint32_t label_count;
} Keyframe;



typedef struct {
    Keyframe history[MAX_KEYFRAME_HISTORY];
    uint32_t history_count;
    uint32_t history_head;      // Circular buffer pointer for FIFO eviction
    float min_iou_threshold;    // Validation threshold for backpropagation matching
    uint32_t total_frames_pushed; 
} AutoLabeler;



AutoLabeler* AutoLabeler_make(float min_iou_threshold);


void AutoLabeler_free(AutoLabeler* labeler);


//adds on an if it is full replaces the head
void AutoLabeler_push_keyframe(AutoLabeler* labeler, Tensor* rgb_img, Tensor* depth_map, Transform3D pose, BoundingBox2D* current_detections, uint32_t det_count);



void AutoLabeler_export_autoregressive_sequence(AutoLabeler* labeler, SceneGraph* graph, CameraIntrinsics intrinsics, const char* output_csv_path);

void AutoLabeler_backpropagate_labels(AutoLabeler* labeler, SceneGraph* graph, CameraIntrinsics intrinsics);


uint32_t AutoLabeler_generate_pseudo_targets(AutoLabeler* labeler, Tensor** out_training_images, BoundingBox2D* out_target_boxes);


Point3D* AutoLabeler_extract_raw_historical_points(AutoLabeler* labeler, uint32_t target_frame_id, CameraIntrinsics intrinsics, uint32_t* out_point_count);

#endif
