#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <stdint.h>
#include "scene_graph.h"
#include "raycasting.h"

typedef struct {
    uint32_t gt_id;
    float gt_centroid[3]; // [x, y, z]
} GTObject;


//let gemini write this .h file

/**
 * @brief Computes ID Switches, Fragmentation, and Centroid Error using your active Scene Graph.
 * * @param active_objects Pointer to the array of TrackedObjects inside your SceneGraph struct
 * @param active_count The current object_count from your SceneGraph struct
 * @param gt_objects Array of ground truth objects loaded from your evaluation dataset frame
 * @param gt_count Number of ground truth objects in the current frame
 */
void evaluate_tracking_quality(const TrackedObject* active_objects, uint32_t active_count,
                               const GTObject* gt_objects, uint32_t gt_count);

/**
 * @brief Calculates Map Convergence Rate and Divergence Error against a baseline.
 * * @param current_octree_root Pointer to your active, optimized Octree map root
 * @param ground_truth_octree_root Pointer to an unoptimized, always-updated baseline Octree map root
 */
void evaluate_mapping_quality(const OctreeNode* current_octree_root, 
                              const OctreeNode* ground_truth_octree_root);

#endif
