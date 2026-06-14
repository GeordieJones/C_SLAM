#ifndef RAYCASTING_H
#define RAYCASTING_H

#include <stddef.h>
#include "slam_math.h"
#include <stdint.h>
//#include "scene_graph.h"
// Configuration toggle: 0 for standard Heap (malloc), 1 for Memory Pool
#define USE_MEMORY_POOL 1
// Algorithm toggle: 0 for Parametric Ray-Box Hierarchy, 1 for Amanatides-Woo DDA Step
#define TRAVERSAL_ALGORITHM 0

typedef struct BoundingBox3D BoundingBox3D;

typedef enum {
    VOXEL_FREE,
    VOXEL_OCCUPIED
} VoxelState;

typedef enum {
    NODE_ACTIVE,    // Dynamic every frame
    NODE_STATIC     // static only 30 seconds or if interacted with by active
} NodeStability;

typedef struct OctreeNode {
    VoxelState state;
    NodeStability stability;
    int stability_update; // flag to let stability timer in scene_graph know
    
    union {
        struct OctreeNode* ptrs[8];
        int indices[8];
    } children;

    Point3D min_bounds;
    Point3D max_bounds;
} OctreeNode;




// Initializes resources. If USE_MEMORY_POOL is 1, it allocates the large block.
void Octree_system_init(size_t max_nodes);


// Spawns/Allocates a node (Automatically selects between malloc or pool internally)
OctreeNode* Octree_make_node(Point3D min_b, Point3D max_b);


void Octree_free(OctreeNode* root);


void Octree_traverse_ray(OctreeNode* node, const Point3D* origin, const Point3D* direction, float t0, float t1);


// Computes a normalized unit vector direction pointing from origin to end_point
Point3D Ray_compute_direction(const Point3D* origin, const Point3D* end_point);


/* * APPROACH A: Parametric Ray-Box Intersection (Kay-Kajiya / Smits Method)
 * Checks if a ray intersects an AABB.
 * Populates t_min (entry distance) and t_max (exit distance).
 * Returns 1 if the ray hits the box, 0 otherwise.
 */
int Ray_box_intersect(const Point3D* origin, const Point3D* direction, const Point3D* min_bounds, const Point3D* max_bounds, float* t_min, float* t_max);



/* * APPROACH B: Amanatides-Woo Grid Traversal (3D DDA Step)
 * Stepped sequential traversal. Marches through leaf voxels directly
 * along the ray line tracking boundary plane crossings.
 */
void Ray_dda_traverse(OctreeNode* root, const Point3D* origin, const Point3D* end_point);


void Octree_insert_ray(OctreeNode* root, const Point3D* origin, const Point3D* end_point);


int Octree_get_child_index(OctreeNode* parent, const Point3D* point);




// Flipping nodes from NODE_ACTIVE to NODE_STATIC if they remain unchanged.
// gets called at the end of every frame
void Octree_update_stability_flags(OctreeNode* root);

// Force updates a static section back to ACTIVE (called by your scene_graph when a moving object touches it)
// gets called at begining of frame
void Octree_invalidate_static_region(OctreeNode* root, const Point3D* min_bound, const Point3D* max_bound);



int Octree_get_occupied_leaves(OctreeNode* root, Point3D* output_buffer, int max_capacity);


void Octree_check_predictive_collision(OctreeNode* root, BoundingBox3D pred_box, uint32_t object_id);

#endif
