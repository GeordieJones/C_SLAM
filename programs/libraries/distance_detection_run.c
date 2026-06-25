#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

// Include your pipeline and geometry headers
#include "pipelines.h"
#include "../../computer_vision/libraries/slam_math.h"

// Global pointer to the configuration struct so the signal handler can access it
static volatile Slam_pipeline* global_config_ptr = NULL;

// Signal handler to catch Ctrl+C (SIGINT) and shut down the streaming loop gracefully
void handle_signal_interrupt(int sig) {
    if (global_config_ptr) {
        printf("\n[SIGNAL] Interrupt caught (Signal %d). Shutting down pipeline cleanly...\n", sig);
        global_config_ptr->running = 0; // Toggles the while loop off
    } else {
        exit(sig);
    }
}

int main(int argc, char* argv[]) {
    printf("========================================================\n");
    printf("          INITIALIZING SLAM VISION EXECUTION            \n");
    printf("========================================================\n");

    // 1. Initialize Pipeline Configuration with robust defaults
    Slam_pipeline config;
    
    config.model_path = "models/nyu_distance_model.nn";
    config.point_cloud_path = "output/environment_map.ply";
    config.generated_dataset_path = "output/causal_training_set.csv";

    config.operational_mode = 1;          // 1 = Inference, 2 = Data Generation, 3 = Online Learning
    config.max_octree_nodes = 500000;     // Pool size allocation limit
    config.initial_objects = 100;         // Starting tracking capacity array limit
    
    // Set standard sensor resolution bounds and optical configurations
    config.intrinsics.fx = 518.0f;
    config.intrinsics.fy = 519.0f;
    config.intrinsics.cx = 320.0f;
    config.intrinsics.cy = 240.0f;
    config.intrinsics.width = 120;
    config.intrinsics.height = 160;

    config.dt = 0.033f;                     // ~30 FPS tracking updates delta gap
    config.min_iou_thresh = 0.35f;         // Overlap target baseline parameter for label matching
    config.frame_count_initialization = 10; // Frames to skip / baseline check frames
    config.dist_thresh = 0.25f;            // Clustering distance threshold (meters)

    config.running = 1;                     // Streaming state switch variable
    config.sg_tracking_threshold = 0.05f; 
    //matches our model for load
    Hyperparameters params = {
            .learning_rate = 0.001f,
            .batch_size = 32,
            .max_epochs = 1,
            .early_stop = 5,
            .early_stop_delta = 0.005f,
            .losstype = LOSS_MSE, // Distance/regression mapping uses Mean Squared Error
            .classification_axis = 1,
            .training_ratio = 0.80f,

            .save_location = "models/nyu_distance_model.nn",
            
            // Spatial dimensions matching your compiled NYU binary frames
            .input_height = 120, 
            .input_width = 160,
            .stride = 1,
            .padding = 1,
            .pool_size = 2
        };
    config.params = params;
         
    // 2. Parse Terminal Operational Mode Arguments
    if (argc > 1) {
        int mode = atoi(argv[1]);
        if (mode >= 1 && mode <= 3) {
            config.operational_mode = mode;
        } else {
            fprintf(stderr, "[WARNING] Invalid mode chosen. Defaulting to Mode 1 (Pure Run).\n");
        }
    }

    printf("[CONFIG] Operational Mode: %d\n", config.operational_mode);
    printf("[CONFIG] Input Resolution: %dx%d\n", config.intrinsics.width, config.intrinsics.height);
    printf("[CONFIG] Voxel Cluster Dist Threshold: %.2f meters\n", config.dist_thresh);
    if (config.operational_mode > 1) {
        printf("[CONFIG] Exporting targets to: %s\n", config.generated_dataset_path);
    }

    // 3. Register the Signal Intercept Hook
    global_config_ptr = &config;
    signal(SIGINT, handle_signal_interrupt);

    // 4. Fire Up the Pipeline Processing Engine Loop
    // This blocks stdin waiting for incoming FFmpeg pipe bytes until finished or interrupted
    int status = run_slam_pipeline(&config);

    // 5. Final Diagnostic Status Print Out
    if (status) {
        printf("\n========================================================\n");
        printf("       SLAM VISION PIPELINE TERMINATED SUCCESSFULY       \n");
        printf("========================================================\n");
    } else {
        printf("\n========================================================\n");
        printf("       SLAM VISION PIPELINE ENCOUNTERED CRITICAL ERROR   \n");
        printf("========================================================\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
