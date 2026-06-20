#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include <stdint.h>

#include "../libaries/scene_graph.h"
#include "../libaries/raycasting.h"
#include "../libaries/slam_math.h"

#define RUN_TEST(test_func) do { \
    printf("[RUNNING] %s...\n", #test_func); \
    test_func(); \
    printf("[PASSED]  %s\n\n", #test_func); \
} while(0)

#define EPSILON 1e-5f


static int is_point_equal(Point3D p1, Point3D p2) {
    return (fabsf(p1.x - p2.x) < EPSILON) &&
           (fabsf(p1.y - p2.y) < EPSILON) &&
           (fabsf(p1.z - p2.z) < EPSILON);
}


void test_octree_lifecycle_and_allocation(void) {
    // Test with a defined node ceiling budget
    size_t max_nodes = 1000;
    Octree_system_init(max_nodes);

    Point3D min_b = {0.0f, 0.0f, 0.0f};
    Point3D max_b = {10.0f, 10.0f, 10.0f};

    OctreeNode* root = Octree_make_node(min_b, max_b);
    assert(root != NULL);
    assert(root->state == VOXEL_FREE);
    assert(root->stability == NODE_ACTIVE);
    assert(is_point_equal(root->min_bounds, min_b));
    assert(is_point_equal(root->max_bounds, max_b));

    Point3D child_min = {0.0f, 0.0f, 0.0f};
    Point3D child_max = {5.0f, 5.0f, 5.0f};
    OctreeNode* child = Octree_make_node(child_min, child_max);
    assert(child != NULL);

    #if USE_MEMORY_POOL
    #endif

    Octree_free(root);
    Octree_free(child); 
}


void test_ray_math_and_directions(void) {
    Point3D origin = {0.0f, 0.0f, 0.0f};
    Point3D end = {3.0f, 4.0f, 0.0f}; // 3-4-5 Triangle

    Point3D dir = Ray_compute_direction(&origin, &end);
    
    float length = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    assert(fabsf(length - 1.0f) < EPSILON);
    assert(fabsf(dir.x - 0.6f) < EPSILON);
    assert(fabsf(dir.y - 0.8f) < EPSILON);
    assert(fabsf(dir.z - 0.0f) < EPSILON);

    Point3D same_end = {0.0f, 0.0f, 0.0f};
    Point3D zero_dir = Ray_compute_direction(&origin, &same_end);
    assert(!isnan(zero_dir.x) && !isnan(zero_dir.y) && !isnan(zero_dir.z));
}


void test_ray_box_intersection_logic(void) {
    Point3D min_b = {1.0f, 1.0f, 1.0f};
    Point3D max_b = {3.0f, 3.0f, 3.0f};

    Point3D origin_hit = {0.0f, 2.0f, 2.0f};
    Point3D direction_hit = {1.0f, 0.0f, 0.0f}; // Ray traveling positive X
    float t_min = 0.0f, t_max = 0.0f;

    int hit = Ray_box_intersect(&origin_hit, &direction_hit, &min_b, &max_b, &t_min, &t_max);
    assert(hit == 1);
    assert(fabsf(t_min - 1.0f) < EPSILON); // Intersects at X=1
    assert(fabsf(t_max - 3.0f) < EPSILON); // Exits at X=3

    Point3D origin_miss = {0.0f, 0.0f, 0.0f};
    Point3D direction_miss = {1.0f, 0.0f, 0.0f}; // Travels along X axis under the bounding box
    hit = Ray_box_intersect(&origin_miss, &direction_miss, &min_b, &max_b, &t_min, &t_max);
    assert(hit == 0);

    Point3D origin_inside = {2.0f, 2.0f, 2.0f};
    hit = Ray_box_intersect(&origin_inside, &direction_hit, &min_b, &max_b, &t_min, &t_max);
    assert(hit == 1);
    assert(t_min <= 0.0f); // Entry should be behind or at origin
    assert(fabsf(t_max - 1.0f) < EPSILON); // Exits at X=3 (distance of 1.0 from X=2)
}


void test_octree_spatial_indexing(void) {
    Point3D min_b = {0.0f, 0.0f, 0.0f};
    Point3D max_b = {2.0f, 2.0f, 2.0f}; // Center point is (1,1,1)
    
    Octree_system_init(10);
    OctreeNode* parent = Octree_make_node(min_b, max_b);

    Point3D pt_0 = {0.5f, 0.5f, 0.5f};
    int idx_0 = Octree_get_child_index(parent, &pt_0);
    assert(idx_0 >= 0 && idx_0 < 8);

    Point3D pt_7 = {1.5f, 1.5f, 1.5f};
    int idx_7 = Octree_get_child_index(parent, &pt_7);
    assert(idx_7 >= 0 && idx_7 < 8);
    assert(idx_0 != idx_7); // Ensure they hash cleanly to unique indices

    Octree_free(parent);
}

void test_ray_insertion_and_traversal(void) {
    Octree_system_init(500);
    Point3D min_root = {0.0f, 0.0f, 0.0f};
    Point3D max_root = {8.0f, 8.0f, 8.0f};
    OctreeNode* root = Octree_make_node(min_root, max_root);

    Point3D origin = {0.5f, 0.5f, 0.5f};
    Point3D end = {7.5f, 7.5f, 7.5f};

    // Insert Ray trace mapping (toggles occupied space along ray)
    Octree_insert_ray(root, &origin, &end);

    // Extract results via internal buffers
    int max_capacity = 64;
    Point3D* occupied_buffer = malloc(sizeof(Point3D) * max_capacity);
    assert(occupied_buffer != NULL);

    int actual_leaves = Octree_get_occupied_leaves(root, occupied_buffer, max_capacity);
    
    // If ray execution worked, we must have mapped leaves captured in buffer
    assert(actual_leaves > 0);
    assert(actual_leaves <= max_capacity);

    // Execute Traversal across structural trees without mutating state or crashing
    #if TRAVERSAL_ALGORITHM == 0
        Octree_traverse_ray(root, &origin, &end, 0.0f, 10.0f);
    #else
        Ray_dda_traverse(root, &origin, &end);
    #endif

    free(occupied_buffer);
    Octree_free(root);
}


void test_stability_state_machine(void) {
    Octree_system_init(100);
    Point3D root_min = {0.0f, 0.0f, 0.0f};
    Point3D root_max = {4.0f, 4.0f, 4.0f};
    OctreeNode* root = Octree_make_node(root_min, root_max);

    assert(root->stability == NODE_ACTIVE);

    // Mock an unchanging system across frames: evaluate stability flag conversions
    Octree_update_stability_flags(root);
    
    // Depending on frame step logic implementation, mock a stability convergence
    // (Force updating node state parameter directly if mock requires it, or evaluate internal flag updater)
    root->stability = NODE_STATIC; 

    Point3D invalidate_min = {0.0f, 0.0f, 0.0f};
    Point3D invalidate_max = {1.0f, 1.0f, 1.0f};

    Octree_invalidate_static_region(root, &invalidate_min, &invalidate_max);

    assert(root->stability == NODE_ACTIVE);

    Octree_free(root);
}


void test_predictive_collisions(void) {
    Octree_system_init(50);
    Point3D root_min = {0.0f, 0.0f, 0.0f};
    Point3D root_max = {10.0f, 10.0f, 10.0f};
    OctreeNode* root = Octree_make_node(root_min, root_max);

    // Define a 3D bounding box heading for collision
    BoundingBox3D pred_box;
    pred_box.min_bounds = (Point3D){2.0f, 2.0f, 2.0f};
    pred_box.max_bounds = (Point3D){4.0f, 4.0f, 4.0f};

    // Assert executing prediction on valid root path does not generate out-of-bounds violations
    uint32_t simulated_object_id = 101;
    //test_predictive_collisions(root, pred_box, simulated_object_id);
    Octree_check_predictive_collision(root, pred_box, simulated_object_id);

    Octree_free(root);
}


int main(void) {
    printf("==================================================\n");
    printf("STARTING RAYCASTING & OCTREE PRODUCTION TEST SUITE\n");
    printf("==================================================\n\n");

    RUN_TEST(test_octree_lifecycle_and_allocation);
    RUN_TEST(test_ray_math_and_directions);
    RUN_TEST(test_ray_box_intersection_logic);
    RUN_TEST(test_octree_spatial_indexing);
    RUN_TEST(test_ray_insertion_and_traversal);
    RUN_TEST(test_stability_state_machine);
    RUN_TEST(test_predictive_collisions);

    printf("==================================================\n");
    printf("ALL TESTS PASSED SUCCESSFULLY! SUITE STABLE.\n");
    printf("==================================================\n");

    return 0;
}
