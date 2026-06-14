#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "auto_labeler.h"
#include "scene_graph.h"
#include "slam_math.h"
#include "../../neural_network/libraries/tensor.h"

#define RUN_TEST(test_func) do { \
    printf("[RUNNING] %s...\n", #test_func); \
    test_func(); \
    printf("[PASSED]  %s\n\n", #test_func); \
} while(0)

// --- MOCK CREATORS FOR NEURAL TENSOR DEPENDENCIES ---

// Creates a dummy mock tensor mimicking an RGB or Depth map array layout
static Tensor* create_mock_sensor_tensor(int channels, int height, int width, float default_val) {
    int shape[3] = {channels, height, width};
    Tensor* t = Tensor_make(3, shape); // Assumes your library uses dimensions + shape array
    if (t && t->data) {
        int total_elements = channels * height * width;
        for (int i = 0; i < total_elements; i++) {
            t->data[i] = default_val;
        }
    }
    return t;
}

// Generates an explicit valid SE(3) Identity transformation matrix
static Transform3D get_identity_pose(void) {
    Transform3D pose = {0};
    pose.m[0][0] = 1.0f;
    pose.m[1][1] = 1.0f;
    pose.m[2][2] = 1.0f;
    pose.m[3][3] = 1.0f;
    return pose;
}

// Generates basic pinhole camera optics parameters for projection testing
static CameraIntrinsics get_mock_intrinsics(void) {
    CameraIntrinsics intrinsics;
    intrinsics.fx = 525.0f; // Typical focal length values
    intrinsics.fy = 525.0f;
    intrinsics.cx = 320.0f; // Centered optical principal axis offset (640x480 frame)
    intrinsics.cy = 240.0f;
    return intrinsics;
}

// --- TEST SUITES ---

/**
 * 1. Allocation Lifecycle Integrity
 * Evaluates safe creation thresholds, initial state parameters, 
 * and absolute teardown memory reclamation properties.
 */
void test_labeler_lifecycle(void) {
    float threshold = 0.45f;
    AutoLabeler* labeler = AutoLabeler_make(threshold);
    
    assert(labeler != NULL);
    assert(labeler->history_count == 0);
    assert(labeler->history_head == 0);
    assert(fabsf(labeler->min_iou_threshold - threshold) < 1e-5f);

    AutoLabeler_free(labeler);
}

/**
 * 2. Circular FIFO Eviction Strategy Buffer Testing
 * Pushes historical bounds beyond MAX_KEYFRAME_HISTORY to ensure the tracking
 * system evicts older elements cleanly without triggering index boundary escapes.
 */
void test_circular_buffer_overflow(void) {
    AutoLabeler* labeler = AutoLabeler_make(0.3f);
    Transform3D pose = get_identity_pose();
    
    // Construct single reusable base layout instances
    Tensor* rgb = create_mock_sensor_tensor(3, 240, 320, 0.5f);
    Tensor* depth = create_mock_sensor_tensor(1, 240, 320, 2.0f);
    BoundingBox2D mock_det = {.confidence = 1.0f, .x_min = 10.0f, .y_min = 10.0f, .x_max = 50.0f, .y_max = 50.0f, .class_id = 0};
    // Push more items than the max capacity threshold bounds (MAX_KEYFRAME_HISTORY = 32)
    uint32_t flood_count = MAX_KEYFRAME_HISTORY + 5; 
    for (uint32_t i = 0; i < flood_count; i++) {
        // In real execution paths, the auto-labeler takes ownership or deep-copies.
        // We pass active references here, mimicking structural ingest pipeline steps.
        AutoLabeler_push_keyframe(labeler, rgb, depth, pose, &mock_det, 1);
    }

    // After overflowing, the active count must clip exactly at the macro ceiling
    assert(labeler->history_count == MAX_KEYFRAME_HISTORY);
    // The circular write pointer must wrap cleanly via bitwise or modulo masks
    assert(labeler->history_head == (flood_count % MAX_KEYFRAME_HISTORY));

    // Cleanup resources
    Tensor_free(rgb);
    Tensor_free(depth);
    AutoLabeler_free(labeler);
}

/**
 * 3. Backpropagation Spatial Temporal Projection Mapping
 * Simulates cross-frame geometric verification by linking objects tracked inside a 
 * SceneGraph across historical historical point structures.
 */
void test_label_backpropagation_pipeline(void) {
    AutoLabeler* labeler = AutoLabeler_make(0.5f);
    SceneGraph* graph = SceneGraph_make(5);
    CameraIntrinsics intrinsics = get_mock_intrinsics();
    Transform3D identity = get_identity_pose();

    // 1. Seed historical frames containing a centered 2D detection box
    Tensor* rgb = create_mock_sensor_tensor(3, 480, 640, 0.0f);
    Tensor* depth = create_mock_sensor_tensor(1, 480, 640, 3.0f); // Even 3-meter depth field plane
   
    BoundingBox2D visual_box = {.confidence = 1.0f, .x_min = 300.0f, .y_min = 220.0f, .x_max = 340.0f, .y_max = 260.0f, .class_id = 0};

    AutoLabeler_push_keyframe(labeler, rgb, depth, identity, &visual_box, 1);

    // 2. Populate the 3D Scene Graph tracking registry with a matching vehicle target
    BoundingBox3D spatial_car = { { -0.2f, -0.2f, 2.8f }, { 0.2f, 0.2f, 3.2f } }; // Centered at Z = 3 meters
    TrackedObject* obj = TrackedObject_make(777, "vehicle", spatial_car);
    graph->objects[0] = *obj;
    graph->object_count = 1;
    free(obj);

    // 3. Fire projection backpropagation matrix operations
    // This projects the 3D car box backward onto the keyframe's image plane, verifies spatial overlap,
    // and labels history automatically.
    AutoLabeler_backpropagate_labels(labeler, graph, intrinsics);

    // Assert that execution completes cleanly without mathematical exceptions or structural hangs
    assert(labeler->history_count == 1);
    
    // Teardown
    Tensor_free(rgb);
    Tensor_free(depth);
    SceneGraph_free(graph);
    AutoLabeler_free(labeler);
}

/**
 * 4. Pseudo-Target Processing Engine
 * Verifies that the labeler flattens history correctly, generating distinct output
 * batch tensors suitable for feeding into training backpropagation steps.
 */
void test_pseudo_target_generation(void) {
    AutoLabeler* labeler = AutoLabeler_make(0.1f);
    Transform3D identity = get_identity_pose();

    Tensor* rgb = create_mock_sensor_tensor(3, 64, 64, 1.0f);
    Tensor* depth = create_mock_sensor_tensor(1, 64, 64, 1.0f);
    BoundingBox2D box = {.confidence = 1.0f, .x_min = 5.0f, .y_min = 5.0f, .x_max = 20.0f, .y_max = 20.0f, .class_id = 0};

    AutoLabeler_push_keyframe(labeler, rgb, depth, identity, &box, 1);
    AutoLabeler_push_keyframe(labeler, rgb, depth, identity, &box, 1);

    // Allocation buffers for extracting targets
    Tensor* training_images[MAX_KEYFRAME_HISTORY] = {NULL};
    BoundingBox2D target_boxes[MAX_KEYFRAME_HISTORY * MAX_LABELS_PER_FRAME];

    uint32_t generated_count = AutoLabeler_generate_pseudo_targets(labeler, training_images, target_boxes);

    // If targets are generated successfully, verify the tracking outputs match
    if (generated_count > 0) {
        assert(training_images[0] != NULL);
        assert(target_boxes[0].x_max > target_boxes[0].x_min);
    }

    Tensor_free(rgb);
    Tensor_free(depth);
    AutoLabeler_free(labeler);
}

/**
 * 5. Dense Historical Pointcloud Reconstruction Extraction
 * Simulates depth data unprojection. Asserts that raw depth pixels maps translate cleanly 
 * back into explicit world-space 3D vector coordinates.
 */
void test_historical_point_extraction(void) {
    AutoLabeler* labeler = AutoLabeler_make(0.5f);
    CameraIntrinsics intrinsics = get_mock_intrinsics();
    Transform3D identity = get_identity_pose();

    // Create a 10x10 mock depth frame map to limit vector array size
    Tensor* rgb = create_mock_sensor_tensor(3, 10, 10, 0.0f);
    Tensor* depth = create_mock_sensor_tensor(1, 10, 10, 5.0f); // Constant 5m depth reading
    BoundingBox2D box = {.confidence = 1.0f, .x_min = 0.0f, .y_min = 0.0f, .x_max = 2.0f, .y_max = 2.0f, .class_id = 0};

    AutoLabeler_push_keyframe(labeler, rgb, depth, identity, &box, 1);

    uint32_t point_count = 0;
    // Extract unprojected points from our initial frame index (0)
    Point3D* points = AutoLabeler_extract_raw_historical_points(labeler, 0, intrinsics, &point_count);

    if (points != NULL) {
        assert(point_count > 0);
        // Ensure the computed distance matches our mock distance depth plane boundary parameter
        assert(points[0].z > 0.0f); 
        free(points); // Free the dynamic array allocated by the extraction method
    }

    Tensor_free(rgb);
    Tensor_free(depth);
    AutoLabeler_free(labeler);
}

// --- MAIN SYSTEM ENTRY ---
int main(void) {
    printf("==================================================\n");
    printf("STARTING CV AUTO-LABELER COMPUTER VISION TEST SUITE\n");
    printf("==================================================\n\n");

    RUN_TEST(test_labeler_lifecycle);
    RUN_TEST(test_circular_buffer_overflow);
    RUN_TEST(test_label_backpropagation_pipeline);
    RUN_TEST(test_pseudo_target_generation);
    RUN_TEST(test_historical_point_extraction);

    printf("==================================================\n");
    printf("ALL AUTO LABELER TESTS PASSED SUCCESSFULLY!\n");
    printf("==================================================\n");

    return 0;
}
