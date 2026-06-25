#include "pipelines.h"
#include <stdio.h>




int main(void) {

    printf("------------running distance_detection_train.c ----------------\n");

    
    int hidden_layers = 5;
    LayerType hidden_topology[] = {LAYER_CONVOLUTION, LAYER_POOLING, LAYER_CONVOLUTION, LAYER_POOLING, LAYER_DENSE};
    int hidden_dimensions[]     = {16, 16, 32, 32, 128}; // 16 channels for Conv/Pool, 128 nodes for Dense

    // -------------------------------------------------------------------------
    // 2. RUN #1: NYU Depth Asset Loop
    // -------------------------------------------------------------------------
    Training_config nyu_config = {
        .dataset_path = "../../Python_files/filtered_data/nyu_training_set.bin",
        .model_save_path = "models/nyu_distance_model.nn",
        .checkpoint_save_path = "models/nyu_chk.nn",
        .metrics_csv_path = "models/nyu_performance_profile.csv",
        .metrics_json_path = "models/nyu_performance_profile.json",

        .params = {
            .learning_rate = 0.001f,
            .batch_size = 32,
            .max_epochs = 1,
            .early_stop = 5,
            .early_stop_delta = 0.005f,
            .losstype = LOSS_MSE, // Distance/regression mapping uses Mean Squared Error
            .classification_axis = 1,
            .training_ratio = 0.80f,
            
            // Spatial dimensions matching your compiled NYU binary frames
            .input_height = 120, 
            .input_width = 160,
            .stride = 1,
            .padding = 1,
            .pool_size = 2
        },

        .input_dim = 1,              // 3 Channels (RGB input stream) [temp changed to 1 for NYU]
        .hidden_layer_count = hidden_layers,
        .hidden_types = hidden_topology,
        .hidden_dims = hidden_dimensions,
        .hidden_act = ACTIVATION_RELU,

        .output_dim = 1,             // Regressing a single depth/distance value per spatial point
        .output_type = LAYER_DENSE,
        .output_act = ACTIVATION_NONE, // No bounding clamp for true regression distance calculations

        .kernel_size = 3,
        .output_channels = 16
    };

    printf("\n[EXECUTION] Launching Pipeline Iteration on NYU Depth Dataset...\n");
    if (!make_and_run_training(&nyu_config)) {
        fprintf(stderr, "[ERROR] Pipeline sequence failed on NYU Depth training block.\n");
    }

    printf("process ran without runtime bugs\n");
    
    return 0;
}












