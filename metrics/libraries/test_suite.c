#include "test_suite.h"
#include "profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include "../../computer_vision/libraries/raycasting.h"

static FILE* g_dataset_manifest_file = NULL;
static char g_dataset_root_dir[256]  = {0};
static int g_is_harness_initialized  = 0;


extern int32_t g_id_map_history[1024];



int test_suite_init(const char* dataset_path, int octree_nodes) {
    if (dataset_path == NULL || strlen(dataset_path) == 0) {
        fprintf(stderr, "[ERROR] Invalid dataset path passed to test suite.\n");
        return 0;
    }

    strncpy(g_dataset_root_dir, dataset_path, sizeof(g_dataset_root_dir) - 1);
    
    size_t len = strlen(g_dataset_root_dir);
    if (g_dataset_root_dir[len - 1] != '/' && len < sizeof(g_dataset_root_dir) - 1) {
        g_dataset_root_dir[len] = '/';
        g_dataset_root_dir[len + 1] = '\0';
    }

    char manifest_full_path[512];
    snprintf(manifest_full_path, sizeof(manifest_full_path), "%smanifest.txt", g_dataset_root_dir);
    
    g_dataset_manifest_file = fopen(manifest_full_path, "r");
    if (g_dataset_manifest_file == NULL) {
        fprintf(stderr, "WARNING: No manifest.txt found at %s. Proceeding with loose frame index scanning.\n", manifest_full_path);
    }

    for (int i = 0; i < 1024; i++) {
        g_id_map_history[i] = -1;
    }

    Octree_system_init(octree_nodes); 

    g_is_harness_initialized = 1;
    printf("[INFO] Test harness successfully initialized for dataset sequence: %s\n", g_dataset_root_dir);
    
    return 1;
}




void test_suite_shutdown(void) {
    if (!g_is_harness_initialized) {
        return; 
    }

    if (g_dataset_manifest_file != NULL) {
        fclose(g_dataset_manifest_file);
        g_dataset_manifest_file = NULL;
    }

    extern OctreeNode* global_active_octree_root;
    extern OctreeNode* global_ground_truth_octree_root;

    if (global_active_octree_root) {
        Octree_free(global_active_octree_root);
        global_active_octree_root = NULL;
    }

    if (global_ground_truth_octree_root) {
        Octree_free(global_ground_truth_octree_root);
        global_ground_truth_octree_root = NULL;
    }

    memset(g_dataset_root_dir, 0, sizeof(g_dataset_root_dir));
    g_is_harness_initialized = 0;

    printf("Test harness environmental allocations safely torn down. Verification run complete.\n");
}







void test_suite_evaluate_frame(uint32_t frame_idx, const OctreeNode* active_root, const OctreeNode* gt_root, const SceneGraph* scene_graph){
    extern int g_is_harness_initialized;
    if (!g_is_harness_initialized) return;
    extern uint32_t g_raycast_total_voxels;
    extern uint32_t g_raycast_static_skipped;

    //forward decleration for the compiler
    int dataset_parser_get_ground_truth_only(uint32_t frame_idx, GTObject* gt_array, uint32_t* count);


    GTObject frame_ground_truth[128];
    uint32_t gt_count = 0;

    //need to make funciton that will get just ground truth might just be get frame in dataset
    int load_success = dataset_parser_get_ground_truth_only(frame_idx, frame_ground_truth, &gt_count);
    if (!load_success) {
        fprintf(stderr, "[ERROR] Test suite failed to load ground truth bounds for frame: %u\n", frame_idx);
        return;
    }


    if (scene_graph != NULL) {
        evaluate_tracking_quality(scene_graph->objects, scene_graph->object_count, frame_ground_truth, gt_count);
    }

    if (active_root != NULL && gt_root != NULL) {
        evaluate_mapping_quality(active_root, gt_root);
    }

    profiler_write_frame_csv();

    g_raycast_total_voxels   = 0;
    g_raycast_static_skipped = 0;
    profiler_clear_frame();
}




void test_suite_run_sequence(uint32_t start_frame, uint32_t end_frame) {
    if (start_frame > end_frame) {
        fprintf(stderr, "[ERROR] Invalid sequence bounds: start_frame (%u) is greater than end_frame (%u).\n", 
                start_frame, end_frame);
        return;
    }

    uint32_t total_frames_to_process = (end_frame - start_frame) + 1;
    printf("[INFO] Starting automated evaluation benchmark sequence...\n");
    printf("[INFO] Processing Window: Frames %u to %u (Total: %u frames)\n", 
           start_frame, end_frame, total_frames_to_process);
    printf("------------------------------------------------------------------\n");

    clock_t sequence_start_time = clock();

    uint32_t processed_count = 0;

    for (uint32_t current_frame = start_frame; current_frame <= end_frame; current_frame++) {
        test_suite_run_frame(current_frame);
        
        processed_count++;

        if (processed_count % 10 == 0 || current_frame == end_frame) {
            float progress_percentage = ((float)processed_count / (float)total_frames_to_process) * 100.0f;
            printf("[BENCHMARK PROGRESS] Passed frame %u / %u (%.1f%% complete)\n", 
                   current_frame, end_frame, progress_percentage);
            
            fflush(stdout); 
        }
    }

    clock_t sequence_end_time = clock();
    double total_sequence_duration_sec = (double)(sequence_end_time - sequence_start_time) / CLOCKS_PER_SEC;
    double average_frame_rate = (double)processed_count / total_sequence_duration_sec;

    printf("------------------------------------------------------------------\n");
    printf("[SUMMARY] Benchmark sequence processing complete!\n");
    printf("[SUMMARY] Total Frames Processed  : %u\n", processed_count);
    printf("[SUMMARY] Total Wall Clock Time   : %.2f seconds\n", total_sequence_duration_sec);
    printf("[SUMMARY] Net Execution Processing Speed : %.2f FPS\n", average_frame_rate);
    printf("------------------------------------------------------------------\n");
}















