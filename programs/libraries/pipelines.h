#ifndef PIPELINES_H
#define PIPELINES_H

#include "../../neural_network/libraries/trainer.h"
#include "../../neural_network/libraries/layer.h"

#include "../../computer_vision/libraries/slam_math.h"

//essentially trainer.c / .h handels the training we just need to set up the network and layers for it to run on

typedef struct{
    
    //paths to files needed
    const char* dataset_path;
    const char* model_save_path;
    const char* checkpoint_save_path;
    const char* metrics_csv_path;
    const char* metrics_json_path;

    //struct in trainer.h with all params including train_test_ratio and checkpoint frequency
    Hyperparameters params;

    //parameters for making network
    int input_dim;
    int hidden_layer_count;
    LayerType* hidden_types;
    int* hidden_dims; //array of dimesions for each layer
    ActivationType hidden_act; //Activation type is in layer.c
    int output_dim;
    ActivationType output_act;
    LayerType output_type;


    //varibles for convolution and pooling 
    //padding, pool size, and stride are in Hyperparameters
    int kernel_size;
    int output_channels;

}Training_config;


typedef struct{

    const char* model_path;
    const char* point_cloud_path;
    const char* generated_dataset_path;

    int operational_mode; //1 = only run, 2 = just save data, 3 = online training
    int max_octree_nodes;
    int initial_objects;
    CameraIntrinsics intrinsics;

    Hyperparameters params;

    float dt;
    float min_iou_thresh;
    int frame_count_initialization;
    float dist_thresh;
    float sg_tracking_threshold;

    int running;    

}Slam_pipeline;


int make_and_run_training(const Training_config* config);

int run_slam_pipeline(const Slam_pipeline* config);



#endif
