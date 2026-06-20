#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>

#include "../libraries/evaluator.h"
#include "../../computer_vision/libraries/slam_math.h"
#include "../../computer_vision/libraries/scene_graph.h"

uint32_t g_raycast_total_voxels = 0;
uint32_t g_raycast_static_skipped = 0;

static uint32_t last_active_tracks = 0;
static uint32_t last_id_switches = 0;
static uint32_t last_fragmentations = 0;
static float last_avg_loc_error = -1.0f;

static float last_skip_ratio = -1.0f;
static float last_convergence_rate = -1.0f;
static float last_avg_divergence_error = -1.0f;

void profiler_update_tracking_metrics(uint32_t active, uint32_t switches, uint32_t frag, float error) {
    last_active_tracks = active;
    last_id_switches = switches;
    last_fragmentations = frag;
    last_avg_loc_error = error;
}

void profiler_update_voxel_metrics(float skip, float convergence, float error) {
    last_skip_ratio = skip;
    last_convergence_rate = convergence;
    last_avg_divergence_error = error;
}

void reset_test_env() {

    last_active_tracks = 0; 
    last_id_switches = 0; 
    last_fragmentations = 0; 
    last_avg_loc_error = -1.0f;
    last_skip_ratio = -1.0f; 
    last_convergence_rate = -1.0f; 
    last_avg_divergence_error = -1.0f;
    g_raycast_total_voxels = 0; 
    g_raycast_static_skipped = 0;
}


void test_tracking_perfect_match() {
    reset_test_env();
    
    TrackedObject active[1] = {
        {
            .id = 10, 
            .centroid = {0.0f, 0.0f, 1.0f}, 
            .is_active = 1
        }
    };
    GTObject gt[1] = {
        {
            .gt_id = 100, 
            .gt_centroid = {0.0f, 0.0f, 1.0f}
        }
    };

    evaluate_tracking_quality(active, 1, gt, 1);

    assert(last_active_tracks == 1);
    assert(last_id_switches == 0);
    assert(last_fragmentations == 0);
    assert(fabsf(last_avg_loc_error) < 1e-4f); // 0.0 cm expected
    printf("[PASS] test_tracking_perfect_match\n");
}

void test_tracking_fragmentation_outside_gate() {
    reset_test_env();
    
    TrackedObject active[1] = {
        {
            .id = 20, 
            .centroid = {10.0f, 10.0f, 10.0f}, 
            .is_active = 1
        }
    };
    GTObject gt[1] = {
        {
            .gt_id = 200, 
            .gt_centroid = {0.0f, 0.0f, 0.0f}
        }
    };

    evaluate_tracking_quality(active, 1, gt, 1);

    assert(last_active_tracks == 1);
    assert(last_fragmentations == 1);
    assert(last_id_switches == 0);
    printf("[PASS] test_tracking_fragmentation_outside_gate\n");
}

void test_tracking_id_switch_detection() {
    reset_test_env();
    
    TrackedObject active[1] = {
        {
            .id = 30, 
            .centroid = {0.0f, 0.0f, 1.0f}, 
            .is_active = 1
        }
    };
    GTObject gt_first[1] = {
        {
            .gt_id = 500, 
            .gt_centroid = {0.0f, 0.0f, 1.01f}
        }
    };
    GTObject gt_second[1] = {
        {
            .gt_id = 600, 
            .gt_centroid = {0.0f, 0.0f, 0.99f}
        }
    };

    evaluate_tracking_quality(active, 1, gt_first, 1);
    assert(last_id_switches == 0);

    evaluate_tracking_quality(active, 1, gt_second, 1);
    assert(last_id_switches == 1);
    printf("[PASS] test_tracking_id_switch_detection\n");
}

void test_mapping_perfect_convergence() {
    reset_test_env();
    
    OctreeNode active_root = { .state = 1, .min_bounds = {0.0f,0.0f,0.0f} };
    OctreeNode gt_root = { .state = 1, .min_bounds = {0.0f,0.0f,0.0f} };
    
    for (int i = 0; i < 8; i++) {
        #if USE_MEMORY_POOL
            active_root.children.indices[i] = -1;
            gt_root.children.indices[i] = -1;
        #else
            active_root.children.ptrs[i] = NULL;
            gt_root.children.ptrs[i] = NULL;
        #endif
    }

    g_raycast_total_voxels = 100;
    g_raycast_static_skipped = 30;

    evaluate_mapping_quality(&active_root, &gt_root);

    assert(fabsf(last_convergence_rate - 1.0f) < 1e-4f); // 100% convergence match
    assert(fabsf(last_skip_ratio - 0.30f) < 1e-4f);
    assert(fabsf(last_avg_divergence_error) < 1e-4f);
    printf("[PASS] test_mapping_perfect_convergence\n");
}

int main() {
    printf("--- RUNNING C_SLAM EVALUATOR UNIT TESTS ---\n");
    test_tracking_perfect_match();
    test_tracking_fragmentation_outside_gate();
    test_tracking_id_switch_detection();
    test_mapping_perfect_convergence();
    printf("--- ALL TESTS PASSED SUCCESSFULLY ---\n");
    return 0;
}
