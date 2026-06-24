#include "pipelines.h"
#include "../../metrics/libraries/profiler.h"
#include "../../metrics/libraries/timer.h"
#include "../../neural_network/libraries/dataset.h"
#include <stdio.h>
#include <stdlib.h>




int make_and_run_training(const Pipeline_config* config) {
    printf("ORCHESTRATOR: Initializing Pipeline Configuration Execution Graph...\n");

    /*if (config->metrics_csv_path) {
        if (!profiler_init(config->metrics_csv_path)) {
            fprintf(stderr, "ERROR: Failed to initialize profiler CSV output file at: %s\n", config->metrics_csv_path);
            return 0;
        }
    }

    profiler_clear_frame();*/

    if (config->metrics_json_path) { 
        if (!timer_init(config->metrics_json_path)) {
            fprintf(stderr, "ERROR: Failed to initialize timer JSON output file at: %s\n", config->metrics_json_path);
            return 0;
        }
    }

    timer_clear_frame();

    

    FILE* header_file = fopen(config->dataset_path, "rb");
    if (!header_file) {
        fprintf(stderr, "ERROR: Failed to access dataset at location: %s\n", config->dataset_path);
        //profiler_shutdown();
        timer_shutdown();
        return 0;
    }


    uint32_t magic, num_samples, channels, height, width;
    (void)fread(&magic, sizeof(uint32_t), 1, header_file);
    (void)fread(&num_samples, sizeof(uint32_t), 1, header_file);
    (void)fread(&channels, sizeof(uint32_t), 1, header_file);
    (void)fread(&height, sizeof(uint32_t), 1, header_file);
    (void)fread(&width, sizeof(uint32_t), 1, header_file);
    (void)fclose(header_file);

    num_samples = flip_endian(num_samples); 

    
    Network* net = Network_make();
    if (!net) {
        fprintf(stderr, "[FATAL ERROR] Graph network footprint allocator failure.\n");
        //profiler_shutdown();
        timer_shutdown();
        return 0;
    }

    int current_height = config->params.input_height;
    int current_width = config->params.input_width;
    int current_input_size = config->input_dim;

    int registration_failed = 0;
    
    for (int i = 0; i < config->hidden_layer_count; i++) {
        Layer* new_layer = NULL;
        LayerType current_type = config->hidden_types[i];

        int initial_layers_count = net->num_layers;

        printf("[DEBUG] Layer %d: Type=%d | Input Channels/Nodes=%d | Image Spatial Size=%dx%d\n", i, current_type, current_input_size, current_height, current_width);


        switch (current_type) {
            case LAYER_DENSE: {
                /*
                new_layer = Layer_make_dense(current_input_size, config->hidden_dims[i], config->hidden_act);
                Network_add_dense(net, new_layer);
                current_input_size = config->hidden_dims[i]; 
                break;*/
                if (current_height > 1 && current_width > 1) {
                    current_input_size = current_input_size * current_height * current_width;
                    current_height = 1; // Reset spatial footprints since it's now flat
                    current_width = 1;
                }

                new_layer = Layer_make_dense(current_input_size, config->hidden_dims[i], config->hidden_act);
                Network_add_dense(net, new_layer);
                current_input_size = config->hidden_dims[i];
                break;
            }
            case LAYER_CONVOLUTION: {
                // For conv layers, current_input_size acts as in_channels, hidden_dims[i] acts as 'out_channels'
                new_layer = Layer_make_convolution(
                    current_input_size, 
                    config->hidden_dims[i], 
                    current_height, 
                    current_width, 
                    config->kernel_size, 
                    config->params.stride, 
                    config->params.padding, 
                    config->hidden_act
                );

                int next_height = (current_height - config->kernel_size + 2 * config->params.padding) / config->params.stride + 1;
                int next_width  = (current_width  - config->kernel_size + 2 * config->params.padding) / config->params.stride + 1;

                Network_add_convolution(net, new_layer, next_height, next_width);

                current_height = next_height;
                current_width  = next_width;
                current_input_size = config->hidden_dims[i]; 
                break;
            }
            case LAYER_POOLING: {
                new_layer = Layer_make_pooling(
                    current_input_size, 
                    current_height, 
                    current_width, 
                    config->params.pool_size, 
                    config->params.stride
                );
                Network_add_pooling(net, new_layer);
                current_height = (current_height - config->params.pool_size) / config->params.stride + 1;
                current_width  = (current_width  - config->params.pool_size) / config->params.stride + 1;
                break;
            }
            
            case LAYER_ADD:
                Network_add_addition(net, 0); // Defaults to a skip branch layout
                break;
            case LAYER_UPSAMPLE:
                Network_add_upsample(net, 2); // Standard scaling factor
                break;
            default:
                fprintf(stderr, "[WARNING] Unknown hidden layer type specified at index %d\n", i);
                break;
        }

        if (current_type != LAYER_ADD && current_type != LAYER_UPSAMPLE) {
            if (new_layer == NULL || net->num_layers == initial_layers_count) {
                registration_failed = 1;
                break;
            }
        }
    }

    if (registration_failed) {
        fprintf(stderr, "[FATAL ERROR] Architecture validation failed during graph deployment. Aborting execution.\n");
        Network_free(net);
        //profiler_shutdown();
        timer_shutdown();
        return 0;
    }

    Layer* output_layer = NULL;
    switch (config->output_type) {
        case LAYER_DENSE:
            output_layer = Layer_make_dense(current_input_size, config->output_dim, config->output_act);
            Network_add_dense(net, output_layer);
            break;
        case LAYER_CONVOLUTION:
            output_layer = Layer_make_convolution(
                current_input_size, 
                config->output_dim, 
                current_height, 
                current_width, 
                config->kernel_size, 
                config->params.stride, 
                config->params.padding, 
                config->output_act
            );
            
            int final_height = (current_height - config->kernel_size + 2 * config->params.padding) / config->params.stride + 1;
            int final_width  = (current_width  - config->kernel_size + 2 * config->params.padding) / config->params.stride + 1;
            
            Network_add_convolution(net, output_layer, final_height, final_width);
            break;
        default:
            fprintf(stderr, "ERROR: Invalid or unsupported output layer type specified.\n");
            Network_free(net);
            //profiler_shutdown();
            timer_shutdown();
            return 0;
    }
    

    
    printf("NETWORK: Architecture compiled successfully. Node Layers: %d\n", net->num_layers);

    Hyperparameters runtime_params = config->params;
    runtime_params.save_location = config->model_save_path;

    printf("TRAIN: Launching localized streaming data optimization loop...\n");
    
    Progress operational_state = {
        .current_epoch = 0,
        .best_validation_loss = 1e9f,
        .unchanged_counter = 0
    };

    Trainer_train(net, config->dataset_path, &runtime_params, &operational_state, (int)num_samples);

    /*
    profiler_write_frame_csv();
    profiler_shutdown();*/

    timer_write_frame_json(0);
    timer_shutdown();


    Network_free(net);
    printf("COMPLETED Pipeline execution context terminated cleanly.\n");

    return 1;

}



















