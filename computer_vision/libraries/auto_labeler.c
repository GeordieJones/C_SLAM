#include "auto_labeler.h"
#include "image_decoders.h"
#include "raycasting.h"
#include "scene_graph.h"

#include "../../neural_network/libraries/tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



AutoLabeler* AutoLabeler_make(float min_iou_threshold){
    //defined in the .h file
    AutoLabeler* labeler = (AutoLabeler*)malloc(sizeof(AutoLabeler));
    if (!labeler) return NULL;

    for (int i = 0; i < MAX_KEYFRAME_HISTORY; i++) {
        labeler->history[i].rgb_image = NULL;
        labeler->history[i].depth_map = NULL;
        labeler->history[i].label_count = 0;
    }

    labeler->history_count = 0;
    labeler->history_head = 0;
    labeler->min_iou_threshold = min_iou_threshold;

    return labeler;
}


void AutoLabeler_free(AutoLabeler* labeler){
    if (!labeler) return;

    for (uint32_t i = 0; i < MAX_KEYFRAME_HISTORY; i++) {
        if (labeler->history[i].rgb_image) {
            Tensor_free(labeler->history[i].rgb_image);
        }
        if (labeler->history[i].depth_map) {
            Tensor_free(labeler->history[i].depth_map);
        }
    }

    free(labeler);
}


void AutoLabeler_push_keyframe(AutoLabeler* labeler, Tensor* rgb_img, Tensor* depth_map, Transform3D pose, BoundingBox2D* current_detections, uint32_t det_count){
    if (!labeler) return;

    uint32_t slot = labeler->history_head;

    if (labeler->history[slot].rgb_image) {
        Tensor_free(labeler->history[slot].rgb_image);
        labeler->history[slot].rgb_image = NULL;
    }
    if (labeler->history[slot].depth_map) {
        Tensor_free(labeler->history[slot].depth_map);
        labeler->history[slot].depth_map = NULL;
    }

    labeler->total_frames_pushed++;
    labeler->history[slot].frame_id = labeler->total_frames_pushed;
    labeler->history[slot].camera_pose = pose;

    if (rgb_img)   labeler->history[slot].rgb_image = Tensor_clone(rgb_img);
    if (depth_map) labeler->history[slot].depth_map = Tensor_clone(depth_map);

    uint32_t copy_count = (det_count < MAX_LABELS_PER_FRAME) ? det_count : MAX_LABELS_PER_FRAME;

    for (uint32_t i = 0; i < copy_count; i++) {
        labeler->history[slot].labels[i] = current_detections[i];
    }
    labeler->history[slot].label_count = copy_count;

    // Bitwise optimization to avoid the modulo operator 
    labeler->history_head = (labeler->history_head + 1) & (MAX_KEYFRAME_HISTORY - 1);

    if (labeler->history_count < MAX_KEYFRAME_HISTORY) {
        labeler->history_count++;
    }
}




// Helper function to calculate standard 2D IoU
static float calculate_iou_2d(BoundingBox2D boxA, BoundingBox2D boxB) {
    float x_left = (boxA.x_min > boxB.x_min) ? boxA.x_min : boxB.x_min;
    float y_top = (boxA.y_min > boxB.y_min) ? boxA.y_min : boxB.y_min;
    float x_right = (boxA.x_max < boxB.x_max) ? boxA.x_max : boxB.x_max;
    float y_bottom = (boxA.y_max < boxB.y_max) ? boxA.y_max : boxB.y_max;

    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float areaA = (boxA.x_max - boxA.x_min) * (boxA.y_max - boxA.y_min);
    float areaB = (boxB.x_max - boxB.x_min) * (boxB.y_max - boxB.y_min);
    
    return intersection_area / (areaA + areaB - intersection_area);
}

//helper since Transform3D is not defined as matrix
Transform3D compute_rigid_inverse(const Transform3D* pose) {
    Transform3D inv;
    
    inv.m[0][0] = pose->m[0][0]; inv.m[0][1] = pose->m[1][0]; inv.m[0][2] = pose->m[2][0];
    inv.m[1][0] = pose->m[0][1]; inv.m[1][1] = pose->m[1][1]; inv.m[1][2] = pose->m[2][1];
    inv.m[2][0] = pose->m[0][2]; inv.m[2][1] = pose->m[1][2]; inv.m[2][2] = pose->m[2][2];

    inv.m[0][3] = -(inv.m[0][0] * pose->m[0][3] + inv.m[0][1] * pose->m[1][3] + inv.m[0][2] * pose->m[2][3]);
    inv.m[1][3] = -(inv.m[1][0] * pose->m[0][3] + inv.m[1][1] * pose->m[1][3] + inv.m[1][2] * pose->m[2][3]);
    inv.m[2][3] = -(inv.m[2][0] * pose->m[0][3] + inv.m[2][1] * pose->m[1][3] + inv.m[2][2] * pose->m[2][3]);

    inv.m[3][0] = 0.0f; inv.m[3][1] = 0.0f; inv.m[3][2] = 0.0f; inv.m[3][3] = 1.0f;

    return inv;
}




/* 
   -if we are looping through the transposes anyway why not keep a local pose to this function that contiunally keeps the pose to the original then come back like a FIFO where the first image in the buffer is given up to max keyframe history points assosiated to it. for example the first image has 32 frames after it so make 32 data points for that first image, kinda like how LLM's train on the previous word and everything before it. This might involve making a switch depending on whether you want to generate as much training data as possible or to do active online learning. we could have swith definition and if it is on data generation it doesnt currently update the model but instead just takes the orginal image and then for each buffer frame up to that point label it as what the first one should have predicted and then make a next prediction and so on just like how language models predict the future word we can make traing data of the room at cirtain points to predict the full room

   - this should maybe just be its own seperate function leave AutoLabeler_backpropagate_labels alone and just make a function to call if we want to make more training data
                    
 - need to make a write to csv or a file or something to save the data maybe a helper function

*/


// HELPER: Appends a single projected 2D bounding box data row to a dataset file
static void append_label_to_dataset(const char* filepath, uint32_t frame_id, uint32_t class_id, BoundingBox2D box) {
    FILE* file = fopen(filepath, "a");
    if (!file) {
        printf("ERROR: Failed to open dataset export file: %s\n", filepath);
        return;
    }

    // Format: frame_id, class_id, x_min, y_min, x_max, y_max, confidence
    fprintf(file, "%u,%u,%.2f,%.2f,%.2f,%.2f,%.4f\n",
            frame_id,
            class_id,
            box.x_min,
            box.y_min,
            box.x_max,
            box.y_max,
            box.confidence);

    fclose(file);
}


// Helper: Multiplies two 4x4 matrices to compute relative transformations
// C = A * B
Transform3D multiply_transforms(const Transform3D* A, const Transform3D* B) {
    Transform3D C;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            C.m[r][c] = A->m[r][0] * B->m[0][c] +
                        A->m[r][1] * B->m[1][c] +
                        A->m[r][2] * B->m[2][c] +
                        A->m[r][3] * B->m[3][c];
        }
    }
    return C;
}



void AutoLabeler_export_autoregressive_sequence(AutoLabeler* labeler, SceneGraph* graph, CameraIntrinsics intrinsics, const char* output_csv_path) {
    if (!labeler || !graph || labeler->history_count < 2) return;

    uint32_t current_live_slot = (labeler->history_head == 0) ? (MAX_KEYFRAME_HISTORY - 1) : (labeler->history_head - 1);

    // Initialize running net_transform as identity matrix
    Transform3D net_transform;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            net_transform.m[r][c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    // Trace backwards sequentially from the present live frame into the past.
    // This allows us to accumulate transforms incrementally: O(1) matrix multiply per frame
    uint32_t trace_slot = current_live_slot;
    
    for (uint32_t step = 0; step < labeler->history_count; step++) {
        Keyframe* future_frame = &labeler->history[trace_slot];

        // Apply the current frame's inverse to net_transform BEFORE processing objects
        if (trace_slot != current_live_slot) {
            Transform3D inv_pose = compute_rigid_inverse(&labeler->history[trace_slot].camera_pose);
            net_transform = multiply_transforms(&net_transform, &inv_pose);
        }

        // Run object projections using the cleanly accumulated incremental matrix
        for (uint32_t o = 0; o < graph->object_count; o++) {
            TrackedObject* obj = &graph->objects[o];
            if (!obj->is_active || obj->movability_score > 0.15f || obj->frames_lost > 3) continue;
            
            BoundingBox3D local_bbox = transform_bbox(obj->bbox, net_transform);

            Point3D corners[8] = {
                {local_bbox.min_bounds.x, local_bbox.min_bounds.y, local_bbox.min_bounds.z},
                {local_bbox.max_bounds.x, local_bbox.min_bounds.y, local_bbox.min_bounds.z},
                {local_bbox.min_bounds.x, local_bbox.max_bounds.y, local_bbox.min_bounds.z},
                {local_bbox.max_bounds.x, local_bbox.max_bounds.y, local_bbox.min_bounds.z},
                {local_bbox.min_bounds.x, local_bbox.min_bounds.y, local_bbox.max_bounds.z},
                {local_bbox.max_bounds.x, local_bbox.min_bounds.y, local_bbox.max_bounds.z},
                {local_bbox.min_bounds.x, local_bbox.max_bounds.y, local_bbox.max_bounds.z},
                {local_bbox.max_bounds.x, local_bbox.max_bounds.y, local_bbox.max_bounds.z}
            };

            float min_u = 99999.0f, max_u = -99999.0f;
            float min_v = 99999.0f, max_v = -99999.0f;
            int behind_camera_count = 0;

            for (int c = 0; c < 8; c++) {
                if (corners[c].z <= 0.1f) {
                    behind_camera_count++;
                    continue;
                }
                float u = (intrinsics.fx * corners[c].x / corners[c].z) + intrinsics.cx;
                float v = (intrinsics.fy * corners[c].y / corners[c].z) + intrinsics.cy;

                if (u < min_u) min_u = u;
                if (u > max_u) max_u = u;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }

            if (behind_camera_count == 8) continue;

            if (min_u < 0) min_u = 0; 
            if (max_u >= 640) max_u = 639;
            if (min_v < 0) min_v = 0; 
            if (max_v >= 480) max_v = 479;

            if ((max_u - min_u) > 2.0f && (max_v - min_v) > 2.0f) {
                BoundingBox2D pseudo_label;
                pseudo_label.x_min = min_u;
                pseudo_label.x_max = max_u;
                pseudo_label.y_min = min_v;
                pseudo_label.y_max = max_v;
                pseudo_label.class_id = obj->id;
                pseudo_label.confidence = 0.95f;

                append_label_to_dataset(output_csv_path, future_frame->frame_id, obj->id, pseudo_label);
            }
        }

        // Step backward in time through circular index buffer
        trace_slot = (trace_slot == 0) ? (MAX_KEYFRAME_HISTORY - 1) : (trace_slot - 1);
    }
}






void AutoLabeler_backpropagate_labels(AutoLabeler* labeler, SceneGraph* graph, CameraIntrinsics intrinsics) {
    if (!labeler || !graph) return;

    uint32_t start_idx = (labeler->history_count >= MAX_KEYFRAME_HISTORY) ? labeler->history_head : 0;
    uint32_t current_live_slot = (labeler->history_head == 0) ? (MAX_KEYFRAME_HISTORY - 1) : (labeler->history_head - 1);

    for (uint32_t h = 0; h < labeler->history_count; h++) {
        uint32_t slot = (start_idx + h) % MAX_KEYFRAME_HISTORY;
        Keyframe* frame = &labeler->history[slot];

        if (slot == current_live_slot) continue;

        // OPTIMIZATION: Compute the backprop matrix path ONCE for this historical frame
        Transform3D net_transform;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                net_transform.m[r][c] = (r == c) ? 1.0f : 0.0f;
            }
        }

        uint32_t trace_slot = current_live_slot;
        while (trace_slot != slot) {
            Transform3D inv_pose = compute_rigid_inverse(&labeler->history[trace_slot].camera_pose);
            
            net_transform = multiply_transforms(&net_transform, &inv_pose);
            
            trace_slot = (trace_slot == 0) ? (MAX_KEYFRAME_HISTORY - 1) : (trace_slot - 1);
        }

        // Run object projections using the single precomputed path matrix
        for (uint32_t o = 0; o < graph->object_count; o++) {
            TrackedObject* obj = &graph->objects[o];
            if (!obj->is_active || obj->movability_score > 0.20f || obj->frames_lost > 5) continue;

            BoundingBox3D historical_bbox = transform_bbox(obj->bbox, net_transform);

            Point3D corners[8] = {
                {historical_bbox.min_bounds.x, historical_bbox.min_bounds.y, historical_bbox.min_bounds.z},
                {historical_bbox.max_bounds.x, historical_bbox.min_bounds.y, historical_bbox.min_bounds.z},
                {historical_bbox.min_bounds.x, historical_bbox.max_bounds.y, historical_bbox.min_bounds.z},
                {historical_bbox.max_bounds.x, historical_bbox.max_bounds.y, historical_bbox.min_bounds.z},
                {historical_bbox.min_bounds.x, historical_bbox.min_bounds.y, historical_bbox.max_bounds.z},
                {historical_bbox.max_bounds.x, historical_bbox.min_bounds.y, historical_bbox.max_bounds.z},
                {historical_bbox.min_bounds.x, historical_bbox.max_bounds.y, historical_bbox.max_bounds.z},
                {historical_bbox.max_bounds.x, historical_bbox.max_bounds.y, historical_bbox.max_bounds.z}
            };

            float min_u = 99999.0f, max_u = -99999.0f;
            float min_v = 99999.0f, max_v = -99999.0f;
            int behind_camera_count = 0;

            for (int i = 0; i < 8; i++) {
                if (corners[i].z <= 0.1f) {
                    behind_camera_count++;
                    continue;
                }
                float u = (intrinsics.fx * corners[i].x / corners[i].z) + intrinsics.cx;
                float v = (intrinsics.fy * corners[i].y / corners[i].z) + intrinsics.cy;

                if (u < min_u) min_u = u;
                if (u > max_u) max_u = u;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }

            if (behind_camera_count == 8) continue;

            if (min_u < 0) min_u = 0;
            if (max_u >= 640) max_u = 639;
            if (min_v < 0) min_v = 0;
            if (max_v >= 480) max_v = 479;

            BoundingBox2D projected_box;
            projected_box.x_min = min_u;
            projected_box.x_max = max_u;
            projected_box.y_min = min_v;
            projected_box.y_max = max_v;
            projected_box.class_id = obj->id; 
            projected_box.confidence = 0.85f;

            int matched_idx = -1;
            float best_iou = 0.0f;

            for (uint32_t l = 0; l < frame->label_count; l++) {
                float iou = calculate_iou_2d(projected_box, frame->labels[l]);
                if (iou > best_iou) {
                    best_iou = iou;
                    matched_idx = l;
                }
            }

            if (best_iou > labeler->min_iou_threshold && matched_idx != -1) {
                frame->labels[matched_idx].x_min = (frame->labels[matched_idx].x_min + projected_box.x_min) * 0.5f;
                frame->labels[matched_idx].x_max = (frame->labels[matched_idx].x_max + projected_box.x_max) * 0.5f;
                frame->labels[matched_idx].y_min = (frame->labels[matched_idx].y_min + projected_box.y_min) * 0.5f;
                frame->labels[matched_idx].y_max = (frame->labels[matched_idx].y_max + projected_box.y_max) * 0.5f;
                frame->labels[matched_idx].confidence = 0.98f; 
            } 
            else if (best_iou < 0.10f && frame->label_count < MAX_LABELS_PER_FRAME) {
                frame->labels[frame->label_count] = projected_box;
                frame->label_count++;
            }
        }
    }
}










uint32_t AutoLabeler_generate_pseudo_targets(AutoLabeler* labeler, Tensor** out_training_images, BoundingBox2D* out_target_boxes) {
    
    if (!labeler || !out_training_images || !out_target_boxes || labeler->history_count == 0) {
        return 0; 
    }

    uint32_t total_targets_harvested = 0;

    uint32_t start_idx = 0;
    if (labeler->history_count >= MAX_KEYFRAME_HISTORY) {
        start_idx = labeler->history_head;
    } else {
        start_idx = 0;
    }

    for (uint32_t i = 0; i < labeler->history_count; i++) {
        
        uint32_t current_slot = (start_idx + i) % MAX_KEYFRAME_HISTORY;
        Keyframe* frame = &labeler->history[current_slot];

        if (frame->label_count == 0 || frame->rgb_image == NULL) {
            continue;
        }

        for (uint32_t l = 0; l < frame->label_count; l++) {
            
            out_training_images[total_targets_harvested] = frame->rgb_image;
            
            out_target_boxes[total_targets_harvested] = frame->labels[l];
            
            total_targets_harvested++;
        }
    }

    return total_targets_harvested;
}




// Returns a fresh array of un-downsampled 3D points in the historical camera's local space.
// Output: out_point_count will hold the number of valid non-zero points extracted.
Point3D* AutoLabeler_extract_raw_historical_points(AutoLabeler* labeler, uint32_t target_frame_id, CameraIntrinsics intrinsics, uint32_t* out_point_count) {
    if (!labeler || !out_point_count) return NULL;
    *out_point_count = 0;

    Keyframe* target_frame = NULL;
    for (uint32_t i = 0; i < labeler->history_count; i++) {
        if (labeler->history[i].frame_id == target_frame_id) {
            target_frame = &labeler->history[i];
            break;
        }
    }

    if (!target_frame || !target_frame->depth_map) {
        return NULL;
    }

    uint32_t H = target_frame->depth_map->shape[1];
    uint32_t W = target_frame->depth_map->shape[2];

    // Allocate an array block large enough to hold the absolute worst-case point density pass
    Point3D* raw_points = (Point3D*)malloc(H * W * sizeof(Point3D));
    if (!raw_points) return NULL;

    uint32_t valid_count = 0;

    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            
            uint32_t index = (0 * H * W) + (y * W) + x; 
            float depth_z = target_frame->depth_map->data[index];

            if (depth_z <= 0.1f || depth_z > 10.0f) {
                continue;
            }

            raw_points[valid_count].x = (x - intrinsics.cx) * depth_z / intrinsics.fx;
            raw_points[valid_count].y = (y - intrinsics.cy) * depth_z / intrinsics.fy;
            raw_points[valid_count].z = depth_z;
            
            valid_count++;
        }
    }

    if (valid_count > 0) {
        raw_points = (Point3D*)realloc(raw_points, valid_count * sizeof(Point3D));
    } else {
        free(raw_points);
        raw_points = NULL;
    }

    *out_point_count = valid_count;
    return raw_points; // Caller is responsible for calling free() on this array block later
}



void AutoLabeler_export_causal_scene_dataset(AutoLabeler* labeler, SceneGraph* finalized_graph, CameraIntrinsics intrinsics, const char* output_csv_path) {
    if (!labeler || !finalized_graph || labeler->history_count < 2) return;

    FILE* csv_file = fopen(output_csv_path, "a");
    if (!csv_file) return;

    uint32_t oldest_slot = (labeler->history_count >= MAX_KEYFRAME_HISTORY) ? labeler->history_head : 0;
    uint32_t sequence_batch_id = labeler->total_frames_pushed;

    for (uint32_t t = 0; t < labeler->history_count; t++) {
        uint32_t current_timeline_slot = (oldest_slot + t) % MAX_KEYFRAME_HISTORY;
        Keyframe* historical_frame = &labeler->history[current_timeline_slot];

        Transform3D world_to_camera_transform = compute_rigid_inverse(&historical_frame->camera_pose);

        for (uint32_t o = 0; o < finalized_graph->object_count; o++) {
            TrackedObject* obj = &finalized_graph->objects[o];
            
            if (!obj->is_active || obj->movability_score > 0.15f) continue;

            Point3D world_corners[8] = {
                {obj->bbox.min_bounds.x, obj->bbox.min_bounds.y, obj->bbox.min_bounds.z},
                {obj->bbox.max_bounds.x, obj->bbox.min_bounds.y, obj->bbox.min_bounds.z},
                {obj->bbox.min_bounds.x, obj->bbox.max_bounds.y, obj->bbox.min_bounds.z},
                {obj->bbox.max_bounds.x, obj->bbox.max_bounds.y, obj->bbox.min_bounds.z},
                {obj->bbox.min_bounds.x, obj->bbox.min_bounds.y, obj->bbox.max_bounds.z},
                {obj->bbox.max_bounds.x, obj->bbox.min_bounds.y, obj->bbox.max_bounds.z},
                {obj->bbox.min_bounds.x, obj->bbox.max_bounds.y, obj->bbox.max_bounds.z},
                {obj->bbox.max_bounds.x, obj->bbox.max_bounds.y, obj->bbox.max_bounds.z}
            };

            float min_u = 99999.0f, max_u = -99999.0f;
            float min_v = 99999.0f, max_v = -99999.0f;
            int behind_camera = 0;

            for (int c = 0; c < 8; c++) {
                Point3D local_corner = transform_point(world_corners[c], world_to_camera_transform);

                if (local_corner.z <= 0.1f) {
                    behind_camera++;
                    continue; 
                }

                float u = (intrinsics.fx * local_corner.x / local_corner.z) + intrinsics.cx;
                float v = (intrinsics.fy * local_corner.y / local_corner.z) + intrinsics.cy;

                if (u < min_u) min_u = u;
                if (u > max_u) max_u = u;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }

            if (behind_camera == 8) continue;

            if (min_u < 0) min_u = 0;  
            if (max_u >= intrinsics.width) max_u = (float)(intrinsics.width - 1);
            if (min_v < 0) min_v = 0;  
            if (max_v >= intrinsics.height) max_v = (float)(intrinsics.height - 1);

            if ((max_u - min_u) > 2.0f && (max_v - min_v) > 2.0f) {
                BoundingBox2D generated_target;
                generated_target.x_min = min_u;
                generated_target.x_max = max_u;
                generated_target.y_min = min_v;
                generated_target.y_max = max_v;
                generated_target.class_id = obj->id;
                generated_target.confidence = 1.00f;

                append_label_to_dataset_stream(csv_file, sequence_batch_id + t, obj->id, generated_target);
            }
        }
    }

    fclose(csv_file);
}



//this for sequential autoregressive training data generation
uint32_t AutoLabeler_generate_autoregressive_targets(AutoLabeler* labeler, Tensor** out_training_images, BoundingBox2D* out_target_boxes, uint32_t* out_box_counts, uint32_t max_batch_size) {
    
    if (!labeler || !out_training_images || !out_target_boxes || !out_box_counts || labeler->history_count == 0) {
        return 0; 
    }

    uint32_t images_harvested = 0;
    uint32_t total_boxes_harvested = 0;

    uint32_t start_idx = (labeler->history_count >= MAX_KEYFRAME_HISTORY) ? labeler->history_head : 0;

    for (uint32_t i = 0; i < labeler->history_count; i++) {
        if (images_harvested >= max_batch_size) break; // Don't exceed batch allocations

        uint32_t current_slot = (start_idx + i) % MAX_KEYFRAME_HISTORY;
        Keyframe* frame = &labeler->history[current_slot];

        if (frame->label_count == 0 || frame->rgb_image == NULL) {
            continue;
        }

        // 1. Save the unique image pointer ONCE per frame slot
        out_training_images[images_harvested] = frame->rgb_image;
        
        // 2. Save how many boxes belong to this unique image slot
        out_box_counts[images_harvested] = frame->label_count;

        // 3. Pack the boxes sequentially into the flat target array
        for (uint32_t l = 0; l < frame->label_count; l++) {
            out_target_boxes[total_boxes_harvested] = frame->labels[l];
            total_boxes_harvested++;
        }

        images_harvested++;
    }

    return total_boxes_harvested;

}
