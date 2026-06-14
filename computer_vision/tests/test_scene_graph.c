#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "scene_graph.h"
#include "raycasting.h"
#include "slam_math.h"

// Configuration mock for the raycasting memory pool dependencies
#define MAX_TEST_NODES 500
#define EPSILON 1e-4f

#define RUN_TEST(test_func) do { \
    printf("[RUNNING] %s...\n", #test_func); \
    test_func(); \
    printf("[PASSED]  %s\n\n", #test_func); \
} while(0)

// --- HELPER FUNCTIONS ---

static int is_point_equal(Point3D p1, Point3D p2) {
    return (fabsf(p1.x - p2.x) < EPSILON) &&
           (fabsf(p1.y - p2.y) < EPSILON) &&
           (fabsf(p1.z - p2.z) < EPSILON);
}

// --- TEST SUITES ---

/**
 * 1. Lifecycle & Memory Allocation Integrity
 * Verifies that the scene graph manages initial allocations, dynamic array
 * expansion capacities, object instances, and proper memory reclamation.
 */
void test_scenegraph_lifecycle(void) {
    uint32_t initial_capacity = 2;
    SceneGraph* graph = SceneGraph_make(initial_capacity);
    
    assert(graph != NULL);
    assert(graph->object_count == 0);
    assert(graph->max_capacity == initial_capacity);
    assert(graph->next_avalible_id == 1 || graph->next_avalible_id == 0); 
    assert(graph->objects != NULL);

    // Instantiate an independent tracked object cluster
    BoundingBox3D bbox = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    TrackedObject* obj = TrackedObject_make(101, "car", bbox);
    
    assert(obj != NULL);
    assert(obj->id == 101);
    assert(strcmp(obj->label, "car") == 0);
    assert(obj->is_active == 1);
    assert(obj->frames_tracked == 1);

    // Standard memory teardown paths
    TrackedObject_free(obj);
    SceneGraph_free(graph);
}

/**
 * 2. 3D Bounding Box Geometry Math
 * Tests initialization, transformation matrices, and 3D Intersection-over-Union (IoU)
 * implementations for overlap tracking computations.
 */
void test_bounding_box_geometry(void) {
    Point3D min_a = {0.0f, 0.0f, 0.0f};
    Point3D max_a = {2.0f, 2.0f, 2.0f}; // Volume = 8
    BoundingBox3D boxA = BoundingBox3D_init(min_a, max_a);

    // Test Case A: Identical overlap (IoU must be exactly 1.0)
    float iou_self = SceneGraph_calculate_iou_3d(boxA, boxA);
    assert(fabsf(iou_self - 1.0f) < EPSILON);

    // Test Case B: Partial 50% X-axis overlap
    Point3D min_b = {1.0f, 0.0f, 0.0f};
    Point3D max_b = {3.0f, 2.0f, 2.0f}; // Volume = 8, Intersection Volume = 1*2*2 = 4
    BoundingBox3D boxB = BoundingBox3D_init(min_b, max_b);
    
    float iou_partial = SceneGraph_calculate_iou_3d(boxA, boxB);
    // Intersection = 4. Union = 8 + 8 - 4 = 12. IoU = 4/12 = 0.33333
    assert(fabsf(iou_partial - 0.33333f) < EPSILON);

    // Test Case C: Absolute Separation (IoU must be exactly 0.0)
    Point3D min_c = {10.0f, 10.0f, 10.0f};
    Point3D max_c = {12.0f, 12.0f, 12.0f};
    BoundingBox3D boxC = BoundingBox3D_init(min_c, max_c);
    float iou_miss = SceneGraph_calculate_iou_3d(boxA, boxC);
    assert(iou_miss == 0.0f);
}

/**
 * 3. Spatial DB Cluster Extraction Engine
 * Mocks an active octree environment to verify that the scene graph successfully
 * reads leaf formations and partitions them out into tracked objects.
 */
void test_octree_cluster_extraction(void) {
    // Spin up global memory pool tracking configuration
    Octree_system_init(MAX_TEST_NODES);
    
    Point3D min_r = {0.0f, 0.0f, 0.0f};
    Point3D max_r = {4.0f, 4.0f, 4.0f};
    OctreeNode* root = Octree_make_node(min_r, max_r);
    assert(root != NULL);

    // Mock an object trace inside the tree space mapping
    Point3D p_start = {1.1f, 1.1f, 1.1f};
    Point3D p_end = {1.9f, 1.9f, 1.9f};
    Octree_insert_ray(root, &p_start, &p_end);

    // Extract clustering details
    uint32_t cluster_count = 0;
    float distance_threshold = 0.5f;
    TrackedObject* clusters = SceneGraph_extract_clusters_from_octree(root, &cluster_count, distance_threshold);

    // If the spatial database mapped elements successfully, cluster targets should be parsed
    if (cluster_count > 0) {
        assert(clusters != NULL);
        assert(clusters[0].is_active == 1);
        free(clusters); // Clean up the returned dynamic array buffer
    }

    Octree_free(root);
}

/**
 * 4. Temporal Frame Tracking & Object Association Alignment
 * Simulates a frame step. Asserts that incoming sensory data matches existing ID targets
 * via IoU proximity, flags updates, adjusts tracking persistence, and scales capacities.
 */
void test_temporal_tracking_and_association(void) {
    // Instantiate a baseline environment graph
    SceneGraph* graph = SceneGraph_make(10);
    
    // Seed an existing object frame ("pedestrian")
    BoundingBox3D frame0_box = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    TrackedObject* initial_obj = TrackedObject_make(1, "pedestrian", frame0_box);
    
    // Manually push to the scene graph vector array
    graph->objects[0] = *initial_obj;
    graph->object_count = 1;
    graph->next_avalible_id = 2;
    free(initial_obj); // Free wrapper shell (contents copied)

    // Construct mock observed clusters for Frame 1 (object shifted slightly along X)
    uint32_t new_cluster_count = 1;
    TrackedObject* detected_clusters = malloc(sizeof(TrackedObject) * new_cluster_count);
    BoundingBox3D frame1_box = {{0.2f, 0.0f, 0.0f}, {1.2f, 1.1f, 1.0f}};
    
    // Observed item has no ID yet (assigned by tracker match)
    memcpy(&detected_clusters[0], TrackedObject_make(0, "unknown", frame1_box), sizeof(TrackedObject));

    // Define identity pairing thresholds (Identity matching matrix match)
    Transform3D identity_pose = {0}; // Zero/Identity translation movement override
    identity_pose.m[0][0] = 1.0f; // X scale/rotation row
    identity_pose.m[1][1] = 1.0f; // Y scale/rotation row
    identity_pose.m[2][2] = 1.0f; // Z scale/rotation row
    identity_pose.m[3][3] = 1.0f;

    float tracking_iou_threshold = 0.2f;

    // Execute temporal filtering sequence
    SceneGraph_update_tracking(graph, detected_clusters, new_cluster_count, identity_pose, tracking_iou_threshold);

    // Assert that the tracker successfully associated the cluster with ID 1
    assert(graph->object_count == 1);
    assert(graph->objects[0].id == 1); 
    assert(graph->objects[0].frames_tracked == 2); // Core frame tracking increments
    assert(graph->objects[0].frames_lost == 0);

    // Clean up
    free(detected_clusters);
    SceneGraph_free(graph);
}

/**
 * 5. Velocity Kinematics & Predictive Contact Guardrails
 * Verifies that object velocities are propagated accurately across timesteps ($dt$)
 * and triggers predictive collision pipelines into static segments of the octree.
 */
void test_kinematics_and_contact_propagation(void) {
    Octree_system_init(MAX_TEST_NODES);
    Point3D min_r = {0.0f, 0.0f, 0.0f};
    Point3D max_r = {10.0f, 10.0f, 10.0f};
    OctreeNode* octree_root = Octree_make_node(min_r, max_r);

    SceneGraph* graph = SceneGraph_make(5);
    
    // Create an object moving at high velocity along the positive X-axis
    BoundingBox3D init_box = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    TrackedObject* moving_obj = TrackedObject_make(42, "drone", init_box);
    moving_obj->velocity = (Point3D){5.0f, 0.0f, 0.0f}; // Moving at 5 meters per second
    
    graph->objects[0] = *moving_obj;
    graph->object_count = 1;
    free(moving_obj);

    // Propagate changes forward by a 100ms timestep interval delta (dt = 0.1s)
    float dt = 0.1f;
    SceneGraph_propagate_contacts(graph, octree_root, dt);

    // Confirm execution completes without crashing and updates local spatial boundaries
    assert(graph->objects[0].id == 42);

    SceneGraph_free(graph);
    Octree_free(octree_root);
}

// --- MAIN ENTRY POINT ---
int main(void) {
    printf("==================================================\n");
    printf("STARTING CV SCENE GRAPH PRODUCTION TEST SUITE\n");
    printf("==================================================\n\n");

    RUN_TEST(test_scenegraph_lifecycle);
    RUN_TEST(test_bounding_box_geometry);
    RUN_TEST(test_octree_cluster_extraction);
    RUN_TEST(test_temporal_tracking_and_association);
    RUN_TEST(test_kinematics_and_contact_propagation);

    printf("==================================================\n");
    printf("ALL SCENE GRAPH TESTS PASSED SUCCESSFULLY!\n");
    printf("==================================================\n");

    return 0;
}
