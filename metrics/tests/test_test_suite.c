
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

#include "../../computer_vision/libraries/slam_math.h"
#include "../../computer_vision/libraries/raycasting.h"
#include "../../computer_vision/libraries/scene_graph.h"
#include "../libraries/evaluator.h"
#include "../libraries/profiler.h"
#include "../libraries/test_suite.h"


int mock_parser_should_succeed = 1;


int32_t g_id_map_history[1024];            
OctreeNode* global_active_octree_root;     
OctreeNode* global_ground_truth_octree_root; 



uint64_t profiler_get_time_us(void) {
    static uint64_t fake_time = 0;
    return fake_time += 1000;
}


int dataset_parser_get_ground_truth_only(uint32_t frame_idx, GTObject* gt_array, uint32_t* count) {
    (void)frame_idx; 
    if (!mock_parser_should_succeed) return 0;
    
    *count = 1;
    gt_array[0].gt_id = 99;
    gt_array[0].gt_centroid[0] = 1.0f;
    gt_array[0].gt_centroid[1] = 2.0f;
    gt_array[0].gt_centroid[2] = 3.0f;
    return 1;
}

extern uint32_t g_raycast_total_voxels;
extern uint32_t g_raycast_static_skipped;

static uint32_t mock_run_frame_calls = 0;
void test_suite_run_frame(uint32_t frame_idx) {
    (void)frame_idx;
    mock_run_frame_calls++;
}

void reset_test_suite_env() {
    mock_run_frame_calls = 0;
    mock_parser_should_succeed = 1;
    g_raycast_total_voxels = 0;
    g_raycast_static_skipped = 0;
    test_suite_shutdown();
}


void test_harness_lifecycle_management() {
    reset_test_suite_env();

    int bad_init = test_suite_init(NULL, 1024);
    assert(bad_init == 0);

    // Mock creation of the loose folder manifest constraint check
    FILE* temp_manifest = fopen("manifest.txt", "w");
    if (temp_manifest) fclose(temp_manifest);

    int valid_init = test_suite_init("./", 512);
    assert(valid_init == 1);

    remove("manifest.txt");

    test_suite_shutdown();
    printf("[PASS] test_harness_lifecycle_management\n");
}

void test_frame_evaluation_pipeline() {
    reset_test_suite_env();
    test_suite_init("./", 128);

    OctreeNode mock_active_root = { .state = VOXEL_FREE, .stability = NODE_ACTIVE };
    OctreeNode mock_gt_root     = { .state = VOXEL_FREE, .stability = NODE_ACTIVE };

    for (int i = 0; i < 8; i++) {
        #if USE_MEMORY_POOL
            mock_active_root.children.indices[i] = -1;
            mock_gt_root.children.indices[i] = -1;
        #else
            mock_active_root.children.ptrs[i] = NULL;
            mock_gt_root.children.ptrs[i] = NULL;
        #endif
    }

    SceneGraph graph;
    graph.object_count = 1;
    graph.max_capacity = 4;
    graph.objects = malloc(sizeof(TrackedObject) * graph.max_capacity);
    
    graph.objects[0].id = 45;
    graph.objects[0].is_active = 1;
    graph.objects[0].centroid.x = 1.0f;
    graph.objects[0].centroid.y = 2.0f;
    graph.objects[0].centroid.z = 3.0f;

    g_raycast_total_voxels = 100;
    g_raycast_static_skipped = 40;

    test_suite_evaluate_frame(1, &mock_active_root, &mock_gt_root, &graph);

    assert(g_raycast_total_voxels == 0);
    assert(g_raycast_static_skipped == 0);

    free(graph.objects);
    test_suite_shutdown();
    printf("[PASS] test_frame_evaluation_pipeline\n");
}

void test_sequence_runner_loops() {
    reset_test_suite_env();
    test_suite_init("./", 64);

    test_suite_run_sequence(50, 10);
    assert(mock_run_frame_calls == 0);

    test_suite_run_sequence(1, 5);
    assert(mock_run_frame_calls == 5);

    test_suite_shutdown();
    printf("[PASS] test_sequence_runner_loops\n");
}

int main() {
    printf("--- RUNNING C_SLAM TEST SUITE ENGINE UNIT TESTS ---\n");
    test_harness_lifecycle_management();
    test_frame_evaluation_pipeline();
    test_sequence_runner_loops();
    printf("--- ALL TESTS PASSED SUCCESSFULLY ---\n");
    return 0;
}
