#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "image_decoders.h"
#include "../../neural_network/libraries/tensor.h"

#include <stdio.h>
#include <float.h>
#include <stdlib.h>



Tensor* Image_load_to_tensor(const char* filepath, ColorChannelOrder channel_order, float mapping_val){

     if(mapping_val != 255.0 && mapping_val != 1.0){
        printf("ERROR: mapping val must be either 1.0 or 255.0\n");
        return NULL;
    }


    int width, height, channels;

    unsigned char* data = stbi_load(filepath, &width, &height, &channels, 0);
    if(!data){ 
        printf("could not get data with stb_image_write\n");
        return NULL;
    }


    int tensor_shape[3] = {channels, height, width};
    Tensor* output = Tensor_make(3, tensor_shape);
    if(!output){
        printf("ERROR could not make tensor");
        stbi_image_free(data);
        return NULL;
    }

    float divisor = (mapping_val == 1.0f) ? 255.0f : 1.0f;

    int image_plane_size = height * width;

    for(int h = 0; h < height; h++){
        int stb_base = (h * width * channels);
        for(int w = 0; w < width; w++){
            for(int c = 0; c < channels; c++){
                
                int stb_index = stb_base + (w * channels) + c;
                int tensor_index = (c * image_plane_size) + (h * width) + w;
                output->data[tensor_index] = (float)data[stb_index] / divisor;
            }
        }
    }

    stbi_image_free(data);
    return output;
}





int Image_save_tensor_to_disk(const char* filepath, const Tensor* img_tensor, ColorChannelOrder channel_order, float mapping_val){

    int channels = img_tensor->shape[0];
    int height = img_tensor->shape[1];
    int width = img_tensor->shape[2];
    int total_elements = channels * height * width;
    int image_plane_size = height * width;

    
    unsigned char* temp_arr = malloc(sizeof(unsigned char) * total_elements);
    
    if(!temp_arr){
        printf("ERROR: out of memory for image export\n");
        return 0;
    }


    //inverse of img to tensor
    float multiplier = (mapping_val == 1.0f) ? 255.0f : 1.0f;
    
    for(int h = 0; h < height; h++){
        int stb_base = (h * width * channels);
        for(int w = 0; w < width; w++){
            for(int c = 0; c < channels; c++){
                int stb_index = stb_base + (w * channels) + c;
                int tensor_index = (c * image_plane_size) + (h * width) + w;

                float val = img_tensor->data[tensor_index] * multiplier;

                if(val < 0.0f) val = 0.0f;
                if(val > 255.0f) val = 255.0f;

                int target = stb_index;
                if(channels ==3 && channel_order == COLOR_BGR){
                    if(c==0) target = stb_index + 2;
                    if(c==2) target = stb_index - 2;
                }
                temp_arr[target] = (unsigned char)(val);
            }
        }    
    }


    
    int status = stbi_write_png(filepath, width, height, channels, temp_arr, width * channels);

    free(temp_arr);
    return status;
}








BoundingBox2D* Image_tensor_to_bounding_boxes(const Tensor* output, float confidense_threshold, int* out_box_count, int classes){

    *out_box_count = 0;
    BoundingBox2D* boxes = (BoundingBox2D*)malloc(10 * sizeof(BoundingBox2D));

    int step = 1;
    for(int i = 1; i < output->ndim; i++){
        step *= output->shape[i];
    }
    
    for(int i = 0; i < output->total_elements; i += step){
        float confidence = output->data[i];

        if(!(confidence >= confidense_threshold)){
            continue;
        }

        boxes[*out_box_count].confidence = confidence;
        boxes[*out_box_count].x_min = output->data[i+1];
        boxes[*out_box_count].y_min = output->data[i+2];
        boxes[*out_box_count].x_max = output->data[i+3];
        boxes[*out_box_count].y_max = output->data[i+4];

        if(classes > 0){
            int class_idx = 0;
            float max_prob = -FLT_MAX;
            for(int c = 0; c < classes; c++){
                int data_idx = i + 5 + c;
                if(max_prob < output->data[data_idx]){
                     max_prob = output->data[data_idx];
                     class_idx = c;
                } 
            }
            boxes[*out_box_count].class_id = class_idx;
        }

        (*out_box_count)++;

        if(*out_box_count > 10){
            int new_cap = *out_box_count * 2;
            BoundingBox2D* temp = (BoundingBox2D*)realloc(boxes, new_cap * sizeof(BoundingBox2D));
            if(!temp){
                printf("ERROR: realloc failed in bounding box making\n");
                free(boxes);
                return NULL;
            }
            boxes = temp;
        }

    }

    if(*out_box_count == 0){
        free(boxes);
        return NULL;
    }

    return boxes;
}


static int compare_boxes(const void* a, const void* b) {
    BoundingBox2D* boxA = (BoundingBox2D*)a;
    BoundingBox2D* boxB = (BoundingBox2D*)b;

    if (boxA->confidence > boxB->confidence) return -1;
    if (boxA->confidence < boxB->confidence) return 1;
    return 0;
}


BoundingBox2D* Image_nonmax_suppression(BoundingBox2D* boxes, int box_count, float iou_threshold, int* out_filtered_count){

    int filtered_count = box_count;

    *out_filtered_count = 0;
    BoundingBox2D* output = (BoundingBox2D*)malloc(10 * sizeof(BoundingBox2D));
    int* suppressed = calloc(box_count, sizeof(int));

    if (!suppressed || !output) {
        free(suppressed); free(output);
        return NULL;
    }

    qsort(boxes, box_count, sizeof(BoundingBox2D), compare_boxes);

    for(int i = 0; i < box_count; i++){

        if(suppressed[i]) continue;

        output[*out_filtered_count] = boxes[i];
        (*out_filtered_count)++;

        for(int j = i+1; j < box_count; j++){
            if(suppressed[j]) continue;
            
            //this feels really inefficent
            if(boxes[j].class_id == boxes[i].class_id){
                BoundingBox2D A = boxes[j];
                BoundingBox2D B = boxes[i];

                float x_min_inter = (A.x_min > B.x_min) ? A.x_min : B.x_min;
                float y_min_inter = (A.y_min > B.y_min) ? A.y_min : B.y_min;
                float x_max_inter = (A.x_max < B.x_max) ? A.x_max : B.x_max;
                float y_max_inter = (A.y_max < B.y_max) ? A.y_max : B.y_max;

                float overlap_width  = x_max_inter - x_min_inter;
                float overlap_height = y_max_inter - y_min_inter;

                if (overlap_width > 0.0f && overlap_height > 0.0f) {
                    float overlap_area = overlap_width * overlap_height;
                    float area_a = (A.x_max - A.x_min) * (A.y_max - A.y_min);
                    float area_b = (B.x_max - B.x_min) * (B.y_max - B.y_min);
                     
                    float union_area = area_a + area_b - overlap_area;
                    float IoU = overlap_area / union_area;

                    if (IoU >= iou_threshold) {
                        suppressed[j] = 1;
                    }
                }
            }
        }
    }

    free(suppressed);
    
    if(*out_filtered_count == 0){
        free(output);
    }


    BoundingBox2D* cleaned_output = realloc(output, (*out_filtered_count) * sizeof(BoundingBox2D));
    return (cleaned_output) ? cleaned_output : output;

}



