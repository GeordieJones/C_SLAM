#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>
#include "tensor.h"

typedef struct {
    Tensor* x;
    Tensor* y;
    int num_samples;
    int* row_indexs;
}Dataset;


Dataset* Dataset_make(Tensor* x, Tensor* y);

void Dataset_free(Dataset* self);

Dataset* Dataset_read_csv(const char* filename, int* label_cols, int num_labels, int has_header);

Dataset* Dataset_read_mnist(const char* image_filename, const char* label_filename);

void Dataset_read_video(Tensor* dest, unsigned char* raw_frame_bytes, int src_width, int src_height);
void img_bgr_to_greyscale(unsigned char* rgb_data, Tensor* dest, int width, int height);
void img_rgb_to_greyscale(unsigned char* rgb_data, Tensor* dest, int width, int height);
void img_resize_nearest(Tensor* src, Tensor* dest);
void img_invert(Tensor* self);
void img_threshold(Tensor* self, float threshold);
void Dataset_min_max_scale(Dataset* self, float new_min, float new_max, int is_same);

void Dataset_normalize(Dataset* self, int is_same); // z score normalization

void Dataset_shuffle(Dataset* self); // use an array of indexs to shuffle instead of shuffling all, needs to shuffle X and Y together to keep mapping

void Dataset_set_batch(Dataset*self, Dataset* batch, int batch_idx, int batch_size);


int Dataset_read_batch(const char* filename, int batch_idx, int batch_size, Dataset* dest);

//must free the out_train.row_indexs and out_val.row_indexs before the master Dataset
void Dataset_split(Dataset* self, float train_ratio, Dataset* out_train, Dataset* out_val);



void Tensor_align_scale_and_shift(Tensor* prediction, Tensor* target, Tensor* dest_aligned);



uint32_t flip_endian(uint32_t num);


#endif












































