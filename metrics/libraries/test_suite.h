#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include <stdint.h>
#include "../../computer_vision/libraries/scene_graph.h"
#include "../../computer_vision/libraries/raycasting.h"
#include "evaluator.h"

//Initializes the test harness environment.
//return int Returns 1 if initialized successfully, 0 if sequence loading fails.
int test_suite_init(const char* dataset_path, int octree_nodes);

void test_suite_shutdown(void);

void test_suite_evaluate_frame(uint32_t frame_idx, const OctreeNode* active_root, const OctreeNode* gt_root, const SceneGraph* scene_graph);

//Executes a single deterministic test cycle for the current frame.
void test_suite_run_frame(uint32_t frame_idx);

//brief Runs a complete automated benchmark over a specified frame window.
void test_suite_run_sequence(uint32_t start_frame, uint32_t end_frame);

#endif
