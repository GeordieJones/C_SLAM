#ifndef IMAGE_DECODERS_H
#define IMAGE_DECODERS_H

//need the -I flag on compile
#include "../../neural_network/libraries/tensor.h"

typedef enum{

    COLOR_BGR,
    COLOR_RGB,
    COLOR_GRAYSCALE

}ColorChannelOrder;

typedef struct{

    float confidence;
    float x_min;
    float y_min;
    float x_max;
    float y_max;   
    int class_id;

}BoundingBox2D;


//loads from disk to tensor mapping val either 1.0 or 255.0 out -> [channels, height, width]
Tensor* Image_load_to_tensor(const char* filepath, ColorChannelOrder channel_order, float mapping_val);


int Image_save_tensor_to_disk(const char* filepath, const Tensor* img_tensor, ColorChannelOrder channel_order, float mapping_val);




BoundingBox2D* Image_tensor_to_bounding_boxes(const Tensor* output, float confidense_threshold, int* out_box_count, int classes);

BoundingBox2D* Image_nonmax_suppression(BoundingBox2D* boxes, int box_count, float iou_threshold, int* out_filtered_count);



void Decode_disparity_to_meters(Tensor* disparity_pred, Tensor* dest_meters, CameraIntrinsics intrinsics, float alpha, float beta);

#endif
