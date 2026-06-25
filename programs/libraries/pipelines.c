#include "pipelines.h"

#include "../../metrics/libraries/profiler.h"
#include "../../metrics/libraries/timer.h"
#include "../../neural_network/libraries/dataset.h"

#include "../../computer_vision/libraries/image_decoders.h"
#include "../../computer_vision/libraries/raycasting.h"
#include "../../computer_vision/libraries/scene_graph.h"
#include "../../computer_vision/libraries/auto_labeler.h"
#include "../../computer_vision/libraries/point_cloud.h"

#include <stdio.h>
#include <stdlib.h>











static Tensor* convert_raw_rgb_to_grayscale_tensor(unsigned char* raw_rgb, int width, int height) {
    // Both datasets preprocess down to a single channel (grayscale)
    int shape[4] = {1, 1, height, width}; 
    Tensor* t = Tensor_make(4, shape);
    if (!t) return NULL;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Indexing into linear interleaved RGB input byte array (R, G, B, R, G, B...)
            int raw_idx = (y * width + x) * 3;
            
            // Destination offset into flat tensor data array
            int out_idx = y * width + x;

            float r = (float)raw_rgb[raw_idx + 0];
            float g = (float)raw_rgb[raw_idx + 1];
            float b = (float)raw_rgb[raw_idx + 2]; // Assumes raw_rgb is in RGB order

            // Apply exact 0.299R + 0.589G + 0.114B formula normalized to [0.0, 1.0]
            t->data[out_idx] = (0.299f * r + 0.589f * g + 0.114f * b) / 255.0f;
        }
    }
    return t;
}












int make_and_run_training(const Training_config* config) {
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
    if (fread(&magic, sizeof(uint32_t), 1, header_file)) {}
    if (fread(&num_samples, sizeof(uint32_t), 1, header_file)) {}
    if (fread(&channels, sizeof(uint32_t), 1, header_file)) {}
    if (fread(&height, sizeof(uint32_t), 1, header_file)) {}
    if (fread(&width, sizeof(uint32_t), 1, header_file)) {}
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





/*

int run_slam_pipeline(const Slam_pipeline* config){

    if(config->operational_mode == 2 && !config->generated_dataset_path){
        fprintf(stderr, "ERROR: generated dataset path not found");
        return 0;
    }

    //initialize cameraIntrinsics, currentpose, octree_system, scenegraph, and autolabeler
    Transform3D current_pose = make_transform_identity();//initilize later after stable set up
    Octree_system_init(config->max_octree_nodes);
    

    //TEMP NEED TO FIND WHAT PROPER INITIAL VALUES ARE--------------------------
    Point3D scene_min = { -10.0f, -10.0f, -10.0f };
    Point3D scene_max = {  10.0f,  10.0f,  10.0f };

    OctreeNode* root_node = Octree_make_node(scene_min, scene_max);


    SceneGraph* tracker = SceneGraph_make(config->initial_objects);
    AutoLabeler* labeler = (config->operational_mode > 1) ? AutoLabeler_make(config->min_iou_thresh) : NULL;

    //initialize varible frame_count
    long int frame_count = 0; 
    float dt = config->dt;

    //load the model
    Network* model = Network_make();
    Trainer_load_model(model, (Hyperparameters*)&(config->params), NULL);//its in trainer 
    if (!model) {
        fprintf(stderr, "ERROR: Neural network model failed to load from %s\n", config->model_path);
        return 0;
    }

    int frame_byte_size = config->intrinsics.width * config->intrinsics.height * 3;
    unsigned char* raw_frame_buffer = (unsigned char*)malloc(frame_byte_size);

    while(config->running){ //the idea of this is to have a way to toggel running on and off with a button or the terminal

        size_t bytes_read = fread(raw_frame_buffer, 1, frame_byte_size, stdin);
        if (bytes_read < (size_t)frame_byte_size) {
            // End of stream pipeline reached or interrupted
            break;
        }

        frame_count++;

        // if frame_count < frame_count_initialization run network on the full image to get baseline this should help to get a steady initialization
        if(frame_count < config->frame_count_initialization){
            continue;
        }

        Tensor* input_frame = convert_raw_rgb_to_grayscale_tensor(raw_frame_buffer, config->intrinsics.width, config->intrinsics.height);
        if (!input_frame) continue;


        Tensor* depth_map = Network_forward(model, input_frame);

        //slam math
        int valid_points = 0;
        Point3D* local_points = project_depth_to_3d(depth_map, &(config->intrinsics), &valid_points);

        Point3D camera_origin = { current_pose.m[0][3], current_pose.m[1][3], current_pose.m[2][3] };
        
        / *
         * need to use these two functions to update position
         *Triangulates a single 3D point from corresponding pixel points 
Point3D triangulate_point(float u1, float v1, float u2, float v2, const Transform3D* T1_to_2, const CameraIntrinsics* K);
        *Point3D transform_identity_3D(const Transform3D* T, const Point3D* p);
        * /
        
        //initilize octree system for this room or area
        for(int i = 0; i < valid_points; i++){
            Point3D world_pt = transform_identity_3D(&current_pose, &local_points[i]);
            Octree_insert_ray(root_node, &camera_origin, &world_pt);
        }

        uint32_t cluster_count = 0;
        TrackedObject* detected_shapes = SceneGraph_extract_clusters_from_octree(root_node, &cluster_count, config->dist_thresh);

        //if objects moved update there positions and movablitiy scores as well as looking at objects in there path
        
        //scene_graph update_tracking
        SceneGraph_update_tracking(tracker, detected_shapes, (int)cluster_count, current_pose, config->sg_tracking_threshold);

        SceneGraph_propagate_contacts(tracker, root_node, dt);

        //switch statement to see if you want to continue, train, or save
        switch(config->operational_mode){
            case 1:
                break;
            case 2://just saving the data
                if (labeler) {
                    // Extract local 2D bounds from image decoders step
                    int box_count = 0;
                    int classes = 10;

                    BoundingBox2D* current_boxes = Image_tensor_to_bounding_boxes(input_frame, 0.5f, &box_count, classes); 
                    
                    int out_filter_count = 0;
                    // Filter overlaps using non-max suppression
                    Image_nonmax_suppression(current_boxes, box_count, config->min_iou_thresh, &out_filter_count);

                    // Push present image parameters down the circular buffer timeline sequence
                    AutoLabeler_push_keyframe(labeler, input_frame, depth_map, current_pose, current_boxes, box_count);
                    
                    // Export causal autoregressive sequences to disk
                    AutoLabeler_export_causal_scene_dataset(labeler, tracker, config->intrinsics, config->generated_dataset_path);
                    
                    free(current_boxes);
                }
                break;
            case 3://same as above but for online learning
                if (labeler) {
                    //THESE PROBABLY NEED ACTUAL VALUES
                    int box_count = 0;
                    int classes = 10;
                    BoundingBox2D* current_boxes = Image_tensor_to_bounding_boxes(depth_map, 0.5f, &box_count, classes);

                    AutoLabeler_push_keyframe(labeler, input_frame, depth_map, current_pose, current_boxes, box_count);
                    
                    // Backpropagate stable multi-object matrices down the timeline
                    AutoLabeler_backpropagate_labels(labeler, tracker, config->intrinsics);
                    
                    // Later need to call training updates inline using generated pseudo targets here
                    free(current_boxes);
                }
                break;

            default:
                break;

        }
        


        Tensor_free(input_frame);
        Tensor_free(depth_map);
        free(local_points);
        if (detected_shapes) free(detected_shapes);

    }

    // shutdown export and free everything;
    PointCloud* final_cloud = PointCloud_make(1000);
    if (final_cloud) {
        int max_capacity = 1000;
        Octree_get_occupied_leaves(root_node, final_cloud->points, max_capacity);
        PointCloud_export_ply(final_cloud, config->point_cloud_path);
        PointCloud_free(final_cloud);
    }

    if (labeler) AutoLabeler_free(labeler);
    SceneGraph_free(tracker);
    free(raw_frame_buffer);
    Octree_free(root_node);
    return 1;

}

*/


int run_slam_pipeline(const Slam_pipeline* config){

    printf("[PIPELINE] Entering run_slam_pipeline...\n");

    if(config->operational_mode == 2 && !config->generated_dataset_path){
        fprintf(stderr, "ERROR: generated dataset path not found\n");
        return 0;
    }

    //initialize cameraIntrinsics, currentpose, octree_system, scenegraph, and autolabeler
    printf("[PIPELINE] Initializing core SLAM geometries and subsystems...\n");
    Transform3D current_pose = make_transform_identity();
    Octree_system_init(config->max_octree_nodes);
    
    Point3D scene_min = { -10.0f, -10.0f, -10.0f };
    Point3D scene_max = {  10.0f,  10.0f,  10.0f };

    OctreeNode* root_node = Octree_make_node(scene_min, scene_max);
    printf("[PIPELINE] Voxel Octree system root node created successfully.\n");

    SceneGraph* tracker = SceneGraph_make(config->initial_objects);
    AutoLabeler* labeler = (config->operational_mode > 1) ? AutoLabeler_make(config->min_iou_thresh) : NULL;
    printf("[PIPELINE] Subsystems initialized. Tracker: %p, Labeler: %p\n", (void*)tracker, (void*)labeler);

    long int frame_count = 0; 
    float dt = config->dt;

    //load the model
    printf("[PIPELINE] Allocating Network workspace and calling Trainer_load_model...\n");
    Network* model = Network_make();
    Trainer_load_model(model, (Hyperparameters*)&(config->params), NULL); 
    if (!model) {
        fprintf(stderr, "ERROR: Neural network model failed to load from %s\n", config->model_path);
        return 0;
    }
    printf("[PIPELINE] Neural network memory configurations synced cleanly.\n");

    int frame_byte_size = config->intrinsics.width * config->intrinsics.height * 3;
    unsigned char* raw_frame_buffer = (unsigned char*)malloc(frame_byte_size);
    printf("[PIPELINE] Frame buffer allocated (%d bytes). Entering streaming loop...\n", frame_byte_size);

    while(config->running){ 
        
        // This print will help us see if the program is hanging right here waiting for stdin bytes!
        printf("[LOOP] Frame %ld: Waiting to read %d bytes from stdin stream...\n", frame_count + 1, frame_byte_size);
        size_t bytes_read = fread(raw_frame_buffer, 1, frame_byte_size, stdin);
        
        if (bytes_read < (size_t)frame_byte_size) {
            printf("[LOOP] Stream boundary reached or pipeline closed. Bytes read: %zu/%d. Exiting loop.\n", 
                   bytes_read, frame_byte_size);
            break;
        }

        frame_count++;
        printf("[LOOP] Frame %ld: Bytes read successfully. Frame processing started.\n", frame_count);

        if(frame_count < config->frame_count_initialization){
            printf("[LOOP] Frame %ld: Base-skipping initialization sync step.\n", frame_count);
            continue;
        }

        printf("[LOOP] Frame %ld: Converting raw RGB to Grayscale tensor...\n", frame_count);
        Tensor* input_frame = convert_raw_rgb_to_grayscale_tensor(raw_frame_buffer, config->intrinsics.width, config->intrinsics.height);
        if (!input_frame) {
            fprintf(stderr, "[WARNING] Frame %ld conversion failed. Skipping frame.\n", frame_count);
            continue;
        }


        printf("[LOOP] Frame %ld: Running Network forward inference pass...\n", frame_count);

        
        Tensor* depth_map = Network_forward(model, input_frame);


        // === ADD THIS DIAGNOSTIC PRINT ===
        printf("[DEBUG] depth_map pointer: %p\n", (void*)depth_map);
        if (depth_map) {
            printf("[DEBUG] depth_map->ndim: %d\n", depth_map->ndim);
            // Only try to print shapes if ndim looks somewhat sane
            if (depth_map->ndim > 0 && depth_map->ndim <= 4) {
                printf("[DEBUG] depth_map shape: [%d, %d, %d, %d]\n", 
                       depth_map->shape[0], depth_map->shape[1], 
                       depth_map->shape[2], depth_map->shape[3]);
            }
        }
        // =================================




        printf("[LOOP] Frame %ld: Projecting depth array maps to 3D space...\n", frame_count);
        int valid_points = 0;
        Point3D* local_points = project_depth_to_3d(depth_map, &(config->intrinsics), &valid_points);

        if (!local_points || valid_points == 0) {
            printf("[LOOP] Frame %ld: No valid 3D points projected. Skipping spatial tracking update.\n", frame_count);
            Tensor_free(input_frame);
            //Tensor_free(depth_map);
            if (local_points) free(local_points);
            continue; // Safely skip to the next camera frame!
        }


        printf("[LOOP] Frame %ld: Extracted %d valid depth coordinate projections.\n", frame_count, valid_points);

        Point3D camera_origin = { current_pose.m[0][3], current_pose.m[1][3], current_pose.m[2][3] };

        // === ADD THIS TEMPORARY COORD DIAGNOSTIC ===
        if (valid_points > 0) {
            printf("\n[DIAGNOSTIC] Sample Points going into Octree:\n");
            for(int p = 0; p < 5 && p < valid_points; p++) {
                Point3D world_pt = transform_identity_3D(&current_pose, &local_points[p]);
                printf("  Pt %d -> Local: [%.2f, %.2f, %.2f] | World: [%.2f, %.2f, %.2f]\n",
                       p, local_points[p].x, local_points[p].y, local_points[p].z,
                       world_pt.x, world_pt.y, world_pt.z);
            }
        }
        // ==========================================
        
        printf("[LOOP] Frame %ld: Raycasting points into Voxel Octree map hierarchy...\n", frame_count);
        for(int i = 0; i < valid_points; i++){
            Point3D world_pt = transform_identity_3D(&current_pose, &local_points[i]);
            Octree_insert_ray(root_node, &camera_origin, &world_pt);
        }

        printf("[LOOP] Frame %ld: Extracting structural tracking clusters from octree...\n", frame_count);
        uint32_t cluster_count = 0;
        TrackedObject* detected_shapes = SceneGraph_extract_clusters_from_octree(root_node, &cluster_count, config->dist_thresh);
        printf("[LOOP] Frame %ld: Clusters extracted -> Total Found: %u\n", frame_count, cluster_count);
        
        printf("[LOOP] Frame %ld: Updating SceneGraph tracking matrices...\n", frame_count);
        SceneGraph_update_tracking(tracker, detected_shapes, (int)cluster_count, current_pose, config->sg_tracking_threshold);

        printf("[LOOP] Frame %ld: Propagating kinematic object collision contacts...\n", frame_count);
        SceneGraph_propagate_contacts(tracker, root_node, dt);

        printf("[LOOP] Frame %ld: Routing operational mode switch conditions (Mode %d)...\n", frame_count, config->operational_mode);
        switch(config->operational_mode){
            case 1:
                break;
            case 2:
                if (labeler) {
                    int box_count = 0;
                    int classes = 10;
                    BoundingBox2D* current_boxes = Image_tensor_to_bounding_boxes(input_frame, 0.5f, &box_count, classes); 
                    
                    int out_filter_count = 0;
                    Image_nonmax_suppression(current_boxes, box_count, config->min_iou_thresh, &out_filter_count);

                    AutoLabeler_push_keyframe(labeler, input_frame, depth_map, current_pose, current_boxes, box_count);
                    AutoLabeler_export_causal_scene_dataset(labeler, tracker, config->intrinsics, config->generated_dataset_path);
                    
                    free(current_boxes);
                }
                break;
            case 3:
                if (labeler) {
                    int box_count = 0;
                    int classes = 10;
                    BoundingBox2D* current_boxes = Image_tensor_to_bounding_boxes(depth_map, 0.5f, &box_count, classes);

                    AutoLabeler_push_keyframe(labeler, input_frame, depth_map, current_pose, current_boxes, box_count);
                    AutoLabeler_backpropagate_labels(labeler, tracker, config->intrinsics);
                    
                    free(current_boxes);
                }
                break;
            default:
                break;
        }
        
        printf("[LOOP] Frame %ld: Releasing processing workspace variables...\n", frame_count);
        Tensor_free(input_frame);
        //Tensor_free(depth_map);
        free(local_points);
        if (detected_shapes) free(detected_shapes);

        printf("[LOOP] Frame %ld: Completed cleanly.\n", frame_count);
    }

    printf("[PIPELINE] Execution loop broken out. Starting clean shutdown sequence...\n");

    PointCloud* final_cloud = PointCloud_make(500000);
    if (final_cloud) {
        printf("[PIPELINE] Exporting final volumetric PLY environment map to disk...\n");
        int max_capacity = 500000;
        
        // This function must return an integer tracking exactly how many points it found!
        int actual_points = Octree_get_occupied_leaves(root_node, final_cloud->points, max_capacity);
        printf("[DEBUG] Octree leaf extraction matched: %d nodes.\n", actual_points);
        
        // Make sure your point cloud structure tracks the exact vertex count found
        final_cloud->count = actual_points; 
        
        PointCloud_export_ply(final_cloud, config->point_cloud_path);
        PointCloud_free(final_cloud);
    }

    printf("[PIPELINE] Deallocating subsystem arrays and data structural layers...\n");
    if (labeler) AutoLabeler_free(labeler);
    SceneGraph_free(tracker);
    free(raw_frame_buffer);
    Octree_free(root_node);

    printf("[PIPELINE] All resources freed successfully. Pipeline exiting.\n");
    return 1;
}


