#include "evaluator.h"
#include "profiler.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>




extern uint32_t g_raycast_total_voxels;
extern uint32_t g_raycast_static_skipped;


#if USE_MEMORY_POOL
    extern OctreeNode* node_pool;
    #define EVAL_GET_CHILD(node, i) ((node)->children.indices[i] == -1 ? NULL : &node_pool[(node)->children.indices[i]])
    #define EVAL_CHILD_EXISTS(node, i) ((node)->children.indices[i] != -1)
#else
    #define EVAL_GET_CHILD(node, i) ((node)->children.ptrs[i])
    #define EVAL_CHILD_EXISTS(node, i) ((node)->children.ptrs[i] != NULL)
#endif



// will possibly need to change this based on the compiler
#define MAX_TRACK_ID_HISTORY 1024
static int32_t g_id_map_history[MAX_TRACK_ID_HISTORY] = { [0 ... MAX_TRACK_ID_HISTORY-1] = -1 };


void evaluate_tracking_quality(const TrackedObject* active_objects, uint32_t active_count,const GTObject* gt_objects, uint32_t gt_count) {
   
    uint32_t current_active_tracks = 0;
    uint32_t id_switches = 0;
    uint32_t fragmentations = 0;
    float total_localization_error_cm = 0.0f;
    uint32_t matched_pairs_count = 0;

    for (uint32_t i = 0; i < active_count; i++) {
        if (!active_objects[i].is_active) {
            continue; 
        }
        
        current_active_tracks++;
        
        float min_dist = 1e7f; 
        int best_gt_idx = -1;

        for (uint32_t j = 0; j < gt_count; j++) {
            float dx = active_objects[i].centroid.x - gt_objects[j].gt_centroid[0];
            float dy = active_objects[i].centroid.y - gt_objects[j].gt_centroid[1];
            float dz = active_objects[i].centroid.z - gt_objects[j].gt_centroid[2];
            float distance = sqrtf(dx*dx + dy*dy + dz*dz);

            if (distance < min_dist) {
                min_dist = distance;
                best_gt_idx = j;
            }
        }

        float object_depth = active_objects[i].centroid.z;
        if (object_depth < 0.1f) object_depth = 1.0f; i

        float dynamic_gate_threshold_m = 0.3f + (object_depth * 0.05f);

        if (best_gt_idx != -1 && min_dist < dynamic_gate_threshold_m) {
            total_localization_error_cm += (min_dist * 100.0f);
            matched_pairs_count++;

            uint32_t active_id = active_objects[i].id;
            int32_t current_gt_associated_id = gt_objects[best_gt_idx].gt_id;

            if (active_id < MAX_TRACK_ID_HISTORY) {
                // ID Switch
                if (g_id_map_history[active_id] != -1 && g_id_map_history[active_id] != current_gt_associated_id) {
                    id_switches++;
                }
                g_id_map_history[active_id] = current_gt_associated_id;
            }
        } else {
            // Track exists but falls outside the valid spatial tracking gate
            fragmentations++; 
        }
    }

    float avg_loc_error = (matched_pairs_count > 0) ? (total_localization_error_cm / (float)matched_pairs_count) : 0.0f;

    profiler_update_tracking_metrics(current_active_tracks, id_switches, fragmentations, avg_loc_error);
}




static void compute_tree_differences(const OctreeNode* active_node, const OctreeNode* gt_node,uint32_t* total_leaves, uint32_t* divergent_leaves, float* total_distance_error) {
    if (!active_node || !gt_node) return;

    int active_is_leaf = !EVAL_CHILD_EXISTS(active_node, 0); 
    int gt_is_leaf     = !EVAL_CHILD_EXISTS(gt_node, 0);

    if (active_is_leaf && gt_is_leaf) {
        (*total_leaves)++;
        
        if (active_node->state != gt_node->state) {
            (*divergent_leaves)++;
            
            float dx = active_node->min_bounds.x - gt_node->min_bounds.x;
            float dy = active_node->min_bounds.y - gt_node->min_bounds.y;
            float dz = active_node->min_bounds.z - gt_node->min_bounds.z;
            *total_distance_error += sqrtf(dx*dx + dy*dy + dz*dz);
        }
        return;
    }

    for (int i = 0; i < 8; i++) {
        if (EVAL_CHILD_EXISTS(active_node, i) && EVAL_CHILD_EXISTS(gt_node, i)) {
            const OctreeNode* active_child = EVAL_GET_CHILD(active_node, i);
            const OctreeNode* gt_child     = EVAL_GET_CHILD(gt_node, i);
            
            compute_tree_differences(active_child, gt_child, total_leaves, divergent_leaves, total_distance_error);
        }
    }
}



void evaluate_mapping_quality(const OctreeNode* current_octree_root, const OctreeNode* ground_truth_octree_root) {
    if (!current_octree_root || !ground_truth_octree_root) return;

    uint32_t total_leaves = 0;
    uint32_t divergent_leaves = 0;
    float total_distance_error = 0.0f;

    compute_tree_differences(current_octree_root, ground_truth_octree_root, 
                             &total_leaves, &divergent_leaves, &total_distance_error);

    float convergence_rate = (total_leaves > 0) ? (1.0f - ((float)divergent_leaves / (float)total_leaves)) : 1.0f;
    float avg_divergence_error = (divergent_leaves > 0) ? (total_distance_error / (float)divergent_leaves) : 0.0f;

    // Fixed: Linked variables match exact production runtime definitions
    extern uint32_t g_raycast_total_voxels;
    extern uint32_t g_raycast_static_skipped;

    float skip_ratio = (g_raycast_total_voxels > 0) ? ((float)g_raycast_static_skipped / (float)g_raycast_total_voxels) : 0.0f;

    profiler_update_voxel_metrics(skip_ratio, convergence_rate, avg_divergence_error);
}


































