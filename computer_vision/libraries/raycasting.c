#include "raycasting.h"
#include "slam_math.h"
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "scene_graph.h"
#include <stdio.h>


static OctreeNode* node_pool = NULL;
static size_t pool_size = 0;
static size_t pool_idx = 0;


//telemetry counters 
uint32_t g_raycast_total_voxels    = 0;
uint32_t g_raycast_static_skipped  = 0;


// using this to beable to test between memory storage types

#if USE_MEMORY_POOL
    // Converts an integer index reference into a memory address inside the pool array
    #define GET_CHILD(node, i) ((node)->children.indices[i] == -1 ? NULL : &node_pool[(node)->children.indices[i]])
    #define CHILD_EXISTS(node, i) ((node)->children.indices[i] != -1)
    #define SET_CHILD(node, i, child_ptr) ((node)->children.indices[i] = ((child_ptr) == NULL) ? -1 : (int)((OctreeNode*)(child_ptr) - node_pool))
#else
    // Falls back to direct memory pointers
    #define GET_CHILD(node, i) ((node)->children.ptrs[i])
    #define CHILD_EXISTS(node, i) ((node)->children.ptrs[i] != NULL)
    #define SET_CHILD(node, i, child_ptr) ((node)->children.ptrs[i] = child_ptr)
#endif

void Octree_system_init(size_t max_nodes){
    if(USE_MEMORY_POOL){
        pool_size = max_nodes;
        pool_idx = 0;
        node_pool = (OctreeNode*)malloc(sizeof(OctreeNode) * pool_size);
    }
    
}


OctreeNode* Octree_make_node(Point3D min_b, Point3D max_b){

    OctreeNode* node = NULL;

    if(USE_MEMORY_POOL){
        if(pool_idx < pool_size){
            node = &node_pool[pool_idx];
            for(int i = 0; i < 8; i++) node->children.indices[i] = -1;
            pool_idx++;
        }else{
            return NULL;
        }
    }else{
        node = (OctreeNode*)malloc(sizeof(OctreeNode));

}

    if(node){
        node->state = VOXEL_FREE;
        node->stability = NODE_ACTIVE;
        node->stability_update = 1; // should start needing update
        node->min_bounds = min_b;
        node->max_bounds = max_b;
        for(int i = 0; i < 8; i++)  SET_CHILD(node, i, NULL);
    }
    
    return node;
}




void Octree_free(OctreeNode* root){
    if(USE_MEMORY_POOL){
        pool_idx = 0;
    }else{
        if(!root) return;
        for(int i = 0; i < 8; i++){
            if(GET_CHILD(root, i)) Octree_free(GET_CHILD(root, i));
        }
        free(root);
    }
}




Point3D Ray_compute_direction(const Point3D* origin, const Point3D* end_point){

    Point3D delta;
    delta.x = end_point->x - origin->x;
    delta.y = end_point->y - origin->y;
    delta.z = end_point->z - origin->z;
    
    return vector_normalize(&delta);
}

void Octree_traverse_ray(OctreeNode* node, const Point3D* origin, const Point3D* direction, float t0, float t1) {
    if (!node) return;

    if (t0 > t1) return;

    if (!CHILD_EXISTS(node, 0)) {
        node->state = VOXEL_FREE;
        node->stability_update = 1;
        return;
    }

    for (int i = 0; i < 8; i++) {
        if (!CHILD_EXISTS(node, i)) continue;
        
        OctreeNode* child = GET_CHILD(node, i);

        //telemetry counter
        g_raycast_total_voxels++;

        if (child->stability == NODE_STATIC) {
            g_raycast_static_skipped++;
            continue;
        }

        float child_tmin = 0.0f, child_tmax = 0.0f;
        
        if (Ray_box_intersect(origin, direction, &child->min_bounds, &child->max_bounds, &child_tmin, &child_tmax)) {
            
            float active_tmin = (child_tmin > t0) ? child_tmin : t0;
            float active_tmax = (child_tmax < t1) ? child_tmax : t1;
            
            if (active_tmin <= active_tmax) {
                Octree_traverse_ray(child, origin, direction, active_tmin, active_tmax);
            }
        }
    }
}





//==========================================================================================
//                                   OPTION A - RAY BOX


/*
 *THIS IS THE SAFE ORGININAL DESIGN
 *
 *
int Ray_box_intersect(const Point3D* origin, const Point3D* direction, const Point3D* min_bounds, const Point3D* max_bounds, float* t_min, float* t_max){

    float tmin = -1e7f;
    float tmax =  1e7f;

    if(direction->x == 0.0f){
        if(origin->x < min_bounds->x || origin->x > max_bounds->x){
            return 0;
        }
    }else{
        float tx1 = (min_bounds->x - origin->x) / direction->x; 
        float tx2 = (max_bounds->x - origin->x) / direction->x; 

        float tx_entry = (tx1 > tx2) ? tx2 : tx1;
        float tx_exit = (tx1 > tx2) ? tx1 : tx2;
 
        if(tx_entry > tmin) tmin = tx_entry;
        if(tx_exit < tmax) tmax = tx_exit;
    }



    if(direction->y == 0.0f){
        if(origin->y < min_bounds->y || origin->y > max_bounds->y){
            return 0;
        }
    }else{
        float ty1 = (min_bounds->y - origin->y) / direction->y; 
        float ty2 = (max_bounds->y - origin->y) / direction->y; 

        float ty_entry = (ty1 > ty2) ? ty2 : ty1;
        float ty_exit = (ty1 > ty2) ? ty1 : ty2;

        if(ty_entry > tmin) tmin = ty_entry;
        if(ty_exit < tmax) tmax = ty_exit;

    }



    if(direction->z == 0.0f){
        if(origin->z < min_bounds->z || origin->z > max_bounds->z){
            return 0;
        }
    }else{
        float tz1 = (min_bounds->z - origin->z) / direction->z; 
        float tz2 = (max_bounds->z - origin->z) / direction->z; 

        float tz_entry = (tz1 > tz2) ? tz2 : tz1;
        float tz_exit = (tz1 > tz2) ? tz1 : tz2;

        if(tz_entry > tmin) tmin = tz_entry;
        if(tz_exit < tmax) tmax = tz_exit;

    }

    if(tmin > tmax) return 0;
    if(tmax < 0.0f) return 0;

    *t_min = tmin;
    *t_max = tmax;

    return 1;
}
*/


//This IS THE OPTIMIZE VERSION WITH DIVSION SKIPS AND ASSUMED DIVIDE BY ZERO SAFTY
int Ray_box_intersect(const Point3D* origin, const Point3D* direction, const Point3D* min_bounds, const Point3D* max_bounds, float* t_min, float* t_max) {
    
    // precompute inverse directions to completely bypass divisions in the loop
    // assuming if direction is 0, inv will be +/-Infinity if not need to add 0.0 check
    float invX = 1.0f / (direction->x != 0.0f ? direction->x : 1e-30f);
    float invY = 1.0f / (direction->y != 0.0f ? direction->y : 1e-30f);
    float invZ = 1.0f / (direction->z != 0.0f ? direction->z : 1e-30f);
    

    float tx1 = (min_bounds->x - origin->x) * invX;
    float tx2 = (max_bounds->x - origin->x) * invX;
    
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);

    float ty1 = (min_bounds->y - origin->y) * invY;
    float ty2 = (max_bounds->y - origin->y) * invY;
    
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    
    float tz1 = (min_bounds->z - origin->z) * invZ;
    float tz2 = (max_bounds->z - origin->z) * invZ;
    
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    // overlap check
    if (tmin > tmax || tmax < 0.0f) {
        return 0;
    }

    *t_min = tmin;
    *t_max = tmax;

    return 1;
}



//==========================================================================================
//                                 OPTION B - GRID TRAVERSAL

void Ray_dda_traverse(OctreeNode* root, const Point3D* origin, const Point3D* end_point) {
    if (!root) return;

    float tmin = 0.0f;
    float tmax = 0.0f;

    Point3D direction = Ray_compute_direction(origin, end_point);

    int valid = Ray_box_intersect(origin, &direction, &root->min_bounds, &root->max_bounds, &tmin, &tmax);
    if (!valid) return;

    Point3D P_start;
    P_start.x = origin->x + tmin * direction.x;
    P_start.y = origin->y + tmin * direction.y;
    P_start.z = origin->z + tmin * direction.z;

    float centerX = (root->min_bounds.x + root->max_bounds.x) * 0.5f;
    float centerY = (root->min_bounds.y + root->max_bounds.y) * 0.5f;
    float centerZ = (root->min_bounds.z + root->max_bounds.z) * 0.5f;

    int X_cell = (P_start.x >= centerX) ? 1 : 0;
    int Y_cell = (P_start.y >= centerY) ? 1 : 0;
    int Z_cell = (P_start.z >= centerZ) ? 1 : 0;

    int stepX = (direction.x >= 0.0f) ? 1 : -1;
    int stepY = (direction.y >= 0.0f) ? 1 : -1;
    int stepZ = (direction.z >= 0.0f) ? 1 : -1;

    //How much t it takes to span half the block width
    float half_widthX = (root->max_bounds.x - root->min_bounds.x) * 0.5f;
    float half_widthY = (root->max_bounds.y - root->min_bounds.y) * 0.5f;
    float half_widthZ = (root->max_bounds.z - root->min_bounds.z) * 0.5f;

    float tDeltaX = (direction.x != 0.0f) ? (half_widthX / fabsf(direction.x)) : 1e7f;
    float tDeltaY = (direction.y != 0.0f) ? (half_widthY / fabsf(direction.y)) : 1e7f;
    float tDeltaZ = (direction.z != 0.0f) ? (half_widthZ / fabsf(direction.z)) : 1e7f;

    float tMaxX = (direction.x != 0.0f) ? ((centerX - P_start.x) / direction.x) : 1e7f;
    float tMaxY = (direction.y != 0.0f) ? ((centerY - P_start.y) / direction.y) : 1e7f;
    float tMaxZ = (direction.z != 0.0f) ? ((centerZ - P_start.z) / direction.z) : 1e7f;

    if (tMaxX < 0.0f) tMaxX += tDeltaX;
    if (tMaxY < 0.0f) tMaxY += tDeltaY;
    if (tMaxZ < 0.0f) tMaxZ += tDeltaZ;


    while ((X_cell & ~1) == 0 && (Y_cell & ~1) == 0 && (Z_cell & ~1) == 0) {
        
        int child_idx = (X_cell << 2) | (Y_cell << 1) | Z_cell;
       
        OctreeNode* child = GET_CHILD(root, child_idx);

        //for telemetry delete if you need less cycles
        g_raycast_total_voxels++;

        if (child != NULL) {

            //telemetry counter for option B
            if (child->stability == NODE_STATIC) {
                g_raycast_static_skipped++;
            } else {
                child->state = VOXEL_FREE;
                child->stability_update = 1; 
            } 
        }

        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                X_cell += stepX;
                tMaxX += tDeltaX;
            } else {
                Z_cell += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                Y_cell += stepY;
                tMaxY += tDeltaY;
            } else {
                Z_cell += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }
}


void Octree_insert_ray(OctreeNode* root, const Point3D* origin, const Point3D* end_point) {
    if (!root) return;

#if TRAVERSAL_ALGORITHM == 0
    Point3D direction = Ray_compute_direction(origin, end_point);
    float tmin = 0.0f, tmax = 0.0f;
    
    if (Ray_box_intersect(origin, &direction, &root->min_bounds, &root->max_bounds, &tmin, &tmax)) {
        Octree_traverse_ray(root, origin, &direction, tmin, tmax);
    }
#elif TRAVERSAL_ALGORITHM == 1
    Ray_dda_traverse(root, origin, end_point);
#endif

    OctreeNode* current = root;
    while (current && CHILD_EXISTS(current, 0)) {
        int idx = Octree_get_child_index(current, end_point);
        current = GET_CHILD(current, idx);
    }


    if (current) {
        current->state = VOXEL_OCCUPIED;
        current->stability_update = 1;
    }
    
}





int Octree_get_child_index(OctreeNode* parent, const Point3D* point) {
    if (!parent) return 0;
    float centerX = (parent->min_bounds.x + parent->max_bounds.x) * 0.5f;
    float centerY = (parent->min_bounds.y + parent->max_bounds.y) * 0.5f;
    float centerZ = (parent->min_bounds.z + parent->max_bounds.z) * 0.5f;

    int bit2 = (point->x >= centerX) ? 1 : 0;
    int bit1 = (point->y >= centerY) ? 1 : 0;
    int bit0 = (point->z >= centerZ) ? 1 : 0;

    return (bit2 << 2) | (bit1 << 1) | bit0;
}






void Octree_update_stability_flags(OctreeNode* root){

    if (root == NULL) return;

    if (!CHILD_EXISTS(root, 0)) {
        if (root->stability_update == 1) {
            root->stability = NODE_ACTIVE;
            root->stability_update = 0;   
        }else{
            root->stability = NODE_STATIC;
        }
        return;
    }


    int all_children_static = 1;

    for (int i = 0; i < 8; i++) {
        if (CHILD_EXISTS(root, i)) {
            OctreeNode* child = GET_CHILD(root, i);
            
            Octree_update_stability_flags(child);
            
            if (child->stability == NODE_ACTIVE) {
                all_children_static = 0; // The parent must stay awake
            }
        }
    }

    if (all_children_static) {
        root->stability = NODE_STATIC;
        root->stability_update = 0;
    } else {
        root->stability = NODE_ACTIVE;
    }
}



void Octree_invalidate_static_region(OctreeNode* root, const Point3D* min_bound, const Point3D* max_bound){

    if (!root) return;

    // Check if the invalidation box touches this node
    if (root->max_bounds.x < min_bound->x || root->min_bounds.x > max_bound->x ||
        root->max_bounds.y < min_bound->y || root->min_bounds.y > max_bound->y ||
        root->max_bounds.z < min_bound->z || root->min_bounds.z > max_bound->z) {
        return; // No spatial overlap
    }

    root->stability = NODE_ACTIVE;
    root->stability_update = 1;

    for (int i = 0; i < 8; i++) {
        if (CHILD_EXISTS(root, i)) {
            Octree_invalidate_static_region(GET_CHILD(root, i), min_bound, max_bound);
        }
    }

}



// Internal recursive engine inside raycasting.c
static void gather_leaves_recursive(OctreeNode* node, Point3D* buffers, int* tracking_idx, int max_tracker) {
    if (node == NULL || *tracking_idx >= max_tracker) return;

    int is_leaf = !CHILD_EXISTS(node, 0);

    if (is_leaf) {
        if (node->state == VOXEL_OCCUPIED) {
            buffers[*tracking_idx] = node->min_bounds;
            (*tracking_idx)++;
        }
        return;
    }

    for (int i = 0; i < 8; i++) {
        if (CHILD_EXISTS(node, i)) {
            gather_leaves_recursive(GET_CHILD(node, i), buffers, tracking_idx, max_tracker);
        }
    }
}




int Octree_get_occupied_leaves(OctreeNode* root, Point3D* output_buffer, int max_capacity) {
    int total_found = 0;
    gather_leaves_recursive(root, output_buffer, &total_found, max_capacity);
    return total_found;
}



static int check_aabb_overlap(BoundingBox3D boxA, BoundingBox3D boxB) {
    return (boxA.min_bounds.x <= boxB.max_bounds.x && boxA.max_bounds.x >= boxB.min_bounds.x) &&
           (boxA.min_bounds.y <= boxB.max_bounds.y && boxA.max_bounds.y >= boxB.min_bounds.y) &&
           (boxA.min_bounds.z <= boxB.max_bounds.z && boxA.max_bounds.z >= boxB.min_bounds.z);
}

//Internal recursive engine inside raycasting.c
static void predictive_collision_recursive(OctreeNode* node, BoundingBox3D pred_box, uint32_t object_id) {
    if (node == NULL) return;

    BoundingBox3D node_box;
    node_box.min_bounds = node->min_bounds;
    node_box.max_bounds = node->max_bounds;
    if (!check_aabb_overlap(node_box, pred_box)) return;

    int is_leaf = !CHILD_EXISTS(node, 0);

    if (is_leaf) {
        if (node->state == VOXEL_OCCUPIED) {
            printf("[WARNING] Tracked Object ID %u predicted collision with environment voxel!\n", object_id);
            node->stability_update = 1; // Wake up mapping system to re-verify this threat sector
        }
        return;
    }

    for (int i = 0; i < 8; i++) {
        if (CHILD_EXISTS(node, i)) {
            predictive_collision_recursive(GET_CHILD(node, i), pred_box, object_id);
        }
    }
}

// Public API wrapper inside raycasting.c
void Octree_check_predictive_collision(OctreeNode* root, BoundingBox3D pred_box, uint32_t object_id) {
    predictive_collision_recursive(root, pred_box, object_id);
}


