#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include <stdint.h>
#include "scene_graph.h"
#include "raycasting.h"
#include "evaluator.h"

/**
 * @brief Initializes the test harness environment.
 * * Allocates data structures, clears historical trackers, and prepares 
 * file streams for loading target sequences (e.g., KITTI/Waymo).
 * * @param dataset_path System path to the directory containing ground-truth data logs.
 * @return int Returns 1 if initialized successfully, 0 if sequence loading fails.
 */
int test_suite_init(const char* dataset_path, int octree_nodes);

/**
 * @brief Shuts down the test harness environment.
 * * Flushes pending files, clears evaluation buffers, and deallocates 
 * memory blocks used by the dataset parser.
 */
void test_suite_shutdown(void);

/**
 * @brief Executes a single deterministic test cycle for the current frame.
 * * This orchestrates the sequence:
 * 1. Resets background telemetry counters (`g_raycast_total_voxels`, etc.).
 * 2. Clears the profile node tree for the frame context.
 * 3. Pumps the next frame of raw sensor image data into the production pipeline.
 * 4. Triggers `evaluate_tracking_quality` and `evaluate_mapping_quality` to run metrics.
 * 5. Serializes the final state into a new row entry via `profiler_write_frame_csv`.
 * * @param frame_idx The sequential sequence index of the frame being processed.
 */
void test_suite_run_frame(uint32_t frame_idx);

/**
 * @brief Runs a complete automated benchmark over a specified frame window.
 * * Loops through frames sequentially, executing `test_suite_run_frame` for each 
 * step, and handles early exits or dropouts gracefully.
 * * @param start_frame The initial frame number in the sequence.
 * @param end_frame The final frame number in the sequence.
 */
void test_suite_run_sequence(uint32_t start_frame, uint32_t end_frame);

#endif // TEST_SUITE_H
