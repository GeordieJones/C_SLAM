#include "dataset.h"
#include "tensor.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <float.h>


Dataset* Dataset_make(Tensor* x, Tensor* y){

    if(x == NULL || y == NULL) return NULL;

    if(x->shape[0] != y->shape[0]){
        fprintf(stderr, "Error: features and label rows are not the same");
        return NULL;
    }

    Dataset* self = malloc(sizeof(Dataset));
    if(self ==NULL){
        return NULL;
    }
    self->x = x;
    self->y = y;
    self->num_samples = x->shape[0];
    self->row_indexs = malloc(self->num_samples * sizeof(int));
    if(self->row_indexs == NULL){
        free(self);
        return NULL;
    }
    for(int i = 0; i < self->num_samples; i++){
        self->row_indexs[i] = i;
    }


    return self;
}

void Dataset_free(Dataset* self){
    if(self != NULL){
        Tensor_free(self->x);
        Tensor_free(self->y);
        free(self->row_indexs);
        free(self);
    }
}




Dataset* Dataset_read_csv(const char* filename, int* label_cols, int num_labels,  int has_header){
    FILE* file = fopen(filename, "r");
    if(!file){
        fprintf(stderr, "Error: could not open file %s\n", filename);
        return NULL;
    }
    
    char line[4096];
    int total_cols = 0;
    int total_rows = 0;

    if (fgets(line, sizeof(line), file)){
        if(!has_header){
            total_rows++;
        }
        char line_copy[4096];
        strcpy(line_copy, line);

        char* token = strtok(line_copy, ",\n\r");
        while(token){
            total_cols++;
            token = strtok(NULL, ",\n\r");
        }

    }


    while(fgets(line, sizeof(line), file)){
        if(line[0] == '\n' || line[0]=='\r') continue;
        total_rows++;

    }

    if(total_rows == 0 || total_cols == 0){
        fprintf(stderr, "Error: csv is empty or invalid");
        fclose(file);
        return NULL;
    }

    int num_features = total_cols - num_labels;
    int xdims[2] = {total_rows, num_features};
    int ydims[2] = {total_rows, num_labels};


    Tensor* x = Tensor_make(2, xdims);
    Tensor* y = Tensor_make(2, ydims);

    if(!x || !y){
        Tensor_free(x);
        Tensor_free(y);
        fclose(file);
        return NULL;    
    }

    rewind(file);
    
    if(has_header){
        if(!fgets(line, sizeof(line), file)){
            fprintf(stderr, "ERROR: failed to read header\n");
            fclose(file);
            return NULL;
        }
    }

    int current_row = 0;
    while(fgets(line, sizeof(line), file) && current_row < total_rows){
        if(line[0] == '\n' || line[0] == '\r') continue;

        int current_col = 0;
        int feature_col_idx = 0;
        //int label_col_idx = 0;
        char* token = strtok(line, ",\n\r");

        while(token){
            float value = strtof(token, NULL);
            
            int is_label = 0;
            int target_position = -1;

            for(int i = 0; i < num_labels; i++){
                if(current_col == label_cols[i]){
                    is_label = 1;
                    target_position = i;
                    break;
                }
            }

            if(is_label){
                y->data[current_row * y->shape[1] + target_position] = value;
            }else{
                x->data[current_row * x->shape[1] + feature_col_idx] = value;
                feature_col_idx++;
            }

            current_col++;
            token = strtok(NULL, ",\n\r");
            
        }
        
        current_row++;

    }

    fclose(file);
    Dataset* dataset = Dataset_make(x,y);
    if(!dataset){
        Tensor_free(x);
        Tensor_free(y);
    }

    return dataset;

}




uint32_t flip_endian(uint32_t num ){
    return((num >> 24) & 0xff) |
          ((num << 8) & 0xff0000) |
          ((num >> 8) & 0xff00)|
          ((num << 24) & 0xff000000);
}




Dataset* Dataset_read_mnist(const char* image_filename, const char* label_filename){

    FILE* img_file = fopen(image_filename, "rb");
    FILE* lbl_file = fopen(label_filename, "rb");

    if(!img_file || !lbl_file){
        fprintf(stderr, "ERROR: coulf not open MNIST files\n");
        if(img_file) fclose(img_file);
        if(lbl_file) fclose(lbl_file);
        return NULL;
    }

    uint32_t magic_img, magic_lbl;
    uint32_t num_imgs, num_lbls;
    uint32_t rows, cols;

    if(
    fread(&magic_img, sizeof(uint32_t), 1, lbl_file) != 1||
    fread(&num_imgs, sizeof(uint32_t), 1, lbl_file)  != 1||
    fread(&rows, sizeof(uint32_t), 1, lbl_file)      != 1||
    fread(&cols, sizeof(uint32_t), 1, lbl_file)      != 1||
    fread(&magic_lbl, sizeof(uint32_t), 1, lbl_file) != 1||
    fread(&num_lbls, sizeof(uint32_t), 1, lbl_file)  != 1
    ){
        fprintf(stderr, "ERROR:failed to read mnist header values metadata \n");
        fclose(img_file);
        fclose(lbl_file);
        return NULL;
    }
    
    magic_img = flip_endian(magic_img);
    num_imgs = flip_endian(num_imgs);
    rows = flip_endian(rows);
    cols = flip_endian(cols);
    magic_lbl = flip_endian(magic_lbl);
    num_lbls = flip_endian(num_lbls);


    if(magic_img != 2051 || magic_lbl != 2049) {
        fprintf(stderr, "ERROR: invalid mnist numbers \n");
        fclose(img_file); fclose(lbl_file);
        return NULL;
    }
    if(num_imgs !=  num_lbls){
        fprintf(stderr, "ERROR: image count %d doesnt match label count %d. \n", num_imgs, num_lbls);
        fclose(img_file); fclose(lbl_file);
        return NULL;
    }

    int num_samples = num_imgs;
    int num_features = rows * cols;
    int num_classes = 10;

    int xdims[2] = {num_samples, num_features};
    int ydims[2] = {num_samples, num_classes};


    Tensor* x = Tensor_make(2, xdims);
    Tensor* y = Tensor_make(2, ydims);

    if(!x || !y){
        Tensor_free(x); Tensor_free(y);
        fclose(img_file);fclose(lbl_file);
        return NULL;
    }

    Tensor_zero(y);

    unsigned char* img_buffer = malloc(num_features);
    unsigned char lbl_byte;

    for(int i = 0; i < num_samples; i++){
        if(fread(img_buffer, sizeof(unsigned char), num_features, img_file) != (size_t)num_features || fread(&lbl_byte, sizeof(unsigned char), 1, lbl_file) != 1){
            fprintf(stderr, " ERROR: unexpected end of file when fread MNIST sample %d\n", i);
            free(img_buffer);
            Tensor_free(x);
            Tensor_free(y);
            fclose(img_file);
            fclose(lbl_file);
            return NULL;
        }



        for(int j = 0; j<num_features; j++){
            x->data[i * num_features + j ] = (float)img_buffer[j] / 255.0f;
        }

        if(lbl_byte < num_classes){
            y->data[i * num_classes + lbl_byte] = 1.0f;
        }

    }

    free(img_buffer);
    fclose(img_file);
    fclose(lbl_file);

    Dataset* dataset = Dataset_make(x, y);
    if(!dataset){
        Tensor_free(x);
        Tensor_free(y);
    }

    return dataset;
}






/*
int Dataset_read_batch(const char* filename, int batch_idx, int batch_size, Dataset* dest) {
    
    if (!dest || !dest->x || !dest->y) {
        fprintf(stderr, "ERROR: Destination dataset or internal tensors are null.\n");
        return 0; 
    }

    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: Could not open dataset file %s\n", filename);
        return 0;
    }

    uint32_t magic, num_samples, channels, height, width;
    size_t header_size = 5 * sizeof(uint32_t); // 20 bytes
    
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_samples, sizeof(uint32_t), 1, file) != 1 ||
        fread(&channels, sizeof(uint32_t), 1, file) != 1 ||
        fread(&height, sizeof(uint32_t), 1, file) != 1 ||
        fread(&width, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "ERROR: Failed to read binary data headers.\n");
        fclose(file);
        return 0;
    }

    magic = flip_endian(magic);
    num_samples = flip_endian(num_samples);
    channels = flip_endian(channels);
    height = flip_endian(height);
    width = flip_endian(width);

    if (magic != 2060) {
        fprintf(stderr, "ERROR: Invalid dataset magic signature mismatch.\n");
        fclose(file);
        return 0;
    }

    int start_sample = batch_idx * batch_size;
    if (start_sample >= (int)num_samples) {
        fprintf(stderr, "ERROR: Requested batch_idx %d out of bounds.\n", batch_idx);
        fclose(file);
        return 0;
    }
    
    int current_batch_size = batch_size;
    if (start_sample + current_batch_size > (int)num_samples) {
        current_batch_size = (int)num_samples - start_sample; // Truncate last batch
    }

    int total_features_per_sample = channels * height * width;
    int total_labels_per_sample = dest->y->shape[1];

    size_t element_size = sizeof(float);
    size_t sample_bytes = total_features_per_sample * element_size;
    size_t label_bytes  = total_labels_per_sample * element_size;

    long offset_x = (long)(header_size + ((size_t)start_sample * sample_bytes));
    if (fseek(file, offset_x, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: Failed to seek to Feature batch position.\n");
        goto cleanup_error;
    }
    size_t x_elements_read = fread(dest->x->data, element_size, current_batch_size * total_features_per_sample, file);

    size_t total_features_block_bytes = (size_t)num_samples * sample_bytes;
    long offset_y = (long)(header_size + total_features_block_bytes + ((size_t)start_sample * label_bytes));
    if (fseek(file, offset_y, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: Failed to seek to Label batch position.\n");
        goto cleanup_error;
    }
    size_t y_elements_read = fread(dest->y->data, element_size, current_batch_size * total_labels_per_sample, file);

    if (x_elements_read != (size_t)(current_batch_size * total_features_per_sample) || 
        y_elements_read != (size_t)(current_batch_size * total_labels_per_sample)) {
        fprintf(stderr, "ERROR: Structural streaming truncation or broken batch records encountered.\n");
        goto cleanup_error;
    }

    fclose(file);

    dest->x->shape[0] = current_batch_size;
    dest->y->shape[0] = current_batch_size;

    return current_batch_size;

    cleanup_error:
        fclose(file);
        return 0;
}*/



int Dataset_read_batch(const char* filename, int batch_idx, int batch_size, Dataset* dest) {
    //printf("  [DATA DEBUG] Entering Dataset_read_batch for batch_idx: %d\n", batch_idx);
    
    if (!dest || !dest->x || !dest->y) {
        fprintf(stderr, "  [DATA ERROR] Destination dataset or internal tensors are null.\n");
        return 0; 
    }

    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "  [DATA ERROR] Could not open dataset file %s\n", filename);
        return 0;
    }

    uint32_t magic, num_samples, channels, height, width;
    size_t header_size = 5 * sizeof(uint32_t); // 20 bytes
    
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_samples, sizeof(uint32_t), 1, file) != 1 ||
        fread(&channels, sizeof(uint32_t), 1, file) != 1 ||
        fread(&height, sizeof(uint32_t), 1, file) != 1 ||
        fread(&width, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "  [DATA ERROR] Failed to read binary data headers.\n");
        fclose(file);
        return 0;
    }

    magic = flip_endian(magic);
    num_samples = flip_endian(num_samples);
    channels = flip_endian(channels);
    height = flip_endian(height);
    width = flip_endian(width);

    printf("  [DATA DEBUG] Parsed Headers inside loader -> Samples: %u, Channels: %u, HxW: %ux%u\n", 
           num_samples, channels, height, width);

    if (magic != 2060) {
        fprintf(stderr, "  [DATA ERROR] Invalid dataset magic signature mismatch: %u (Expected 2060)\n", magic);
        fclose(file);
        return 0;
    }

    int start_sample = batch_idx * batch_size;
    printf("  [DATA DEBUG] start_sample: %d | total num_samples: %u\n", start_sample, num_samples);
    if (start_sample >= (int)num_samples) {
        fprintf(stderr, "  [DATA ERROR] Requested batch_idx %d out of bounds.\n", batch_idx);
        fclose(file);
        return 0;
    }
    
    int current_batch_size = batch_size;
    if (start_sample + current_batch_size > (int)num_samples) {
        current_batch_size = (int)num_samples - start_sample; 
        printf("  [DATA DEBUG] Truncating trailing edge batch size down to: %d\n", current_batch_size);
    }

    int total_features_per_sample = channels * height * width;
    int total_labels_per_sample = dest->y->shape[1]; 

    size_t element_size = sizeof(float);
    size_t sample_bytes = total_features_per_sample * element_size;
    size_t label_bytes  = total_labels_per_sample * element_size;

    printf("  [DATA DEBUG] Features/sample: %d (%zu bytes) | Labels/sample: %d (%zu bytes)\n", 
           total_features_per_sample, sample_bytes, total_labels_per_sample, label_bytes);

    // 1. Feature reading diagnostics
    long offset_x = (long)(header_size + ((size_t)start_sample * sample_bytes));
    //printf("  [DATA DEBUG] Seeking to Feature position. Offset_X: %ld\n", offset_x);
    
    if (fseek(file, offset_x, SEEK_SET) != 0) {
        fprintf(stderr, "  [DATA ERROR] Failed to seek to Feature batch position.\n");
        goto cleanup_error;
    }
    
    size_t x_elements_to_read = (size_t)current_batch_size * total_features_per_sample;
    printf("  [DATA DEBUG] Reading features. Destination buffer pointer: %p, Elements to read: %zu\n", 
           (void*)dest->x->data, x_elements_to_read);
    
    size_t x_elements_read = fread(dest->x->data, element_size, x_elements_to_read, file);
    printf("  [DATA DEBUG] Features read successfully: %zu elements\n", x_elements_read);

    // 2. Label reading diagnostics
    size_t total_features_block_bytes = (size_t)num_samples * sample_bytes;
    long offset_y = (long)(header_size + total_features_block_bytes + ((size_t)start_sample * label_bytes));
    printf("  [DATA DEBUG] Seeking to Label position. Total feature block size: %zu, Offset_Y: %ld\n", 
           total_features_block_bytes, offset_y);
    
    if (fseek(file, offset_y, SEEK_SET) != 0) {
        fprintf(stderr, "  [DATA ERROR] Failed to seek to Label batch position.\n");
        goto cleanup_error;
    }
    
    size_t y_elements_to_read = (size_t)current_batch_size * total_labels_per_sample;
    printf("  [DATA DEBUG] Reading labels. Destination buffer pointer: %p, Elements to read: %zu\n", 
           (void*)dest->y->data, y_elements_to_read);
    
    size_t y_elements_read = fread(dest->y->data, element_size, y_elements_to_read, file);
    printf("  [DATA DEBUG] Labels read successfully: %zu elements\n", y_elements_read);

    if (x_elements_read != x_elements_to_read || y_elements_read != y_elements_to_read) {
        fprintf(stderr, "  [DATA ERROR] Structural streaming truncation or broken batch records encountered.\n");
        fprintf(stderr, "  [DATA ERROR] Expected X: %zu (Got: %zu) | Expected Y: %zu (Got: %zu)\n", 
                x_elements_to_read, x_elements_read, y_elements_to_read, y_elements_read);
        goto cleanup_error;
    }

    fclose(file);

    dest->x->shape[0] = current_batch_size;
    dest->y->shape[0] = current_batch_size;

    //printf("  [DATA DEBUG] Exiting Dataset_read_batch successfully. Returning size: %d\n", current_batch_size);
    return current_batch_size;

    cleanup_error:
        printf("  [DATA DEBUG] Branching into error cleanup routine.\n");
        fclose(file);
        return 0;
}







void Dataset_read_video(Tensor* dest, unsigned char* raw_frame_bytes, int src_width, int src_height){
    int dims[2] = {src_height, src_width};
    Tensor* gray = Tensor_make(2, dims);
    img_rgb_to_greyscale(raw_frame_bytes, gray, src_width, src_height);
    Tensor_copy(dest, gray);
    Tensor_free(gray);
}




void img_bgr_to_greyscale(unsigned char* rgb_data, Tensor* dest, int width, int height){
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            
            int byte_idx = (r * width + c) * 3;

            float blue = (float)rgb_data[byte_idx];
            float green = (float)rgb_data[byte_idx + 1];
            float red = (float)rgb_data[byte_idx + 2];

            float grey_val = (0.299f * red) + (0.589 * green) + (0.114 * blue);

            float normed = grey_val / 255.0;

            int Tensor_idx = r * dest->shape[1] + c;
            dest->data[Tensor_idx] = normed;
        }
    }
}




void img_rgb_to_greyscale(unsigned char* rgb_data, Tensor* dest, int width, int height){
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            
            int byte_idx = (r * width + c) * 3;

            float red = (float)rgb_data[byte_idx];
            float green = (float)rgb_data[byte_idx + 1];
            float blue = (float)rgb_data[byte_idx + 2];

            float grey_val = (0.299f * red) + (0.589 * green) + (0.114 * blue);

            float normed = grey_val / 255.0;

            int Tensor_idx = r * dest->shape[1] + c;
            dest->data[Tensor_idx] = normed;
        }
    }
}



void img_resize_nearest(Tensor* src, Tensor* dest){

    int src_hight = src->shape[0];
    int src_width = src->shape[1];
    int dest_hight = dest->shape[0];
    int dest_width = dest->shape[1];

    float ratio_x = ((float)(src_width))/((float)(dest_width));   
    float ratio_y = ((float)(src_hight))/((float)(dest_hight));
    
    int c_val;
    int r_val;

    for(int i = 0; i < dest_hight; i++){
        r_val = (int)(ratio_y * i);
        for (int j = 0; j < dest_width; j++){
            c_val = (int)(ratio_x * j);
            dest->data[i*dest_width + j] = src->data[r_val*src_width + c_val];    
        }
    }

}





static float invert(float input){
    return 1.0 - input;
}

void img_invert(Tensor* self){
    Tensor_map(self, invert);
}


void img_threshold(Tensor* self, float threshold){
    int total_elements = self->total_elements; 

    for(int i = 0; i< total_elements; i++){
        if(self->data[i] > threshold){
            self->data[i] = 1.0f;
        }else{
            self->data[i] = 0.0f;
        }
    }

}


void Dataset_min_max_scale(Dataset* self, float new_min, float new_max, int is_same){
    int total_elements = self->x->total_elements; 
    
    float rng;

    if(is_same){ 
        
        MinMax results = Tensor_min_max(self->x); 
        float max = results.max;
        float min = results.min;

        rng = max - min;
        if(rng == 0.0f) rng = 1.0f;

        
        float target_range = new_max - new_min;
        
        for(int i = 0; i < total_elements; i++){
            self->x->data[i] = (((self->x->data[i] - min) / rng) * target_range) + new_min;
        }
    } else {
        
        int num_features = self->x->shape[1];
        int num_samples = self->x->shape[0];

        
        int stat_dims[1] = {num_features};
        Tensor* col_mins = Tensor_make(1, stat_dims);
        Tensor* col_maxes = Tensor_make(1, stat_dims);

        if (!col_mins || !col_maxes) {
            if (col_mins) Tensor_free(col_mins);
            if (col_maxes) Tensor_free(col_maxes);
            return; 
        }

        Tensor_argsmin(self->x, col_mins, 0);
        Tensor_argsmax(self->x, col_maxes, 0);

        for(int c = 0; c < num_features; c++){
            float col_min = col_mins->data[c];
            float col_max = col_maxes->data[c];

            rng = col_max - col_min;
            if(rng == 0.0f) rng = 1.0f;

            float target_range = new_max - new_min;

            
            for(int r = 0; r < num_samples; r++){
                int idx = r * num_features + c;
                float raw_val = self->x->data[idx];
                self->x->data[idx] = (((raw_val - col_min) / rng) * target_range) + new_min;
            }
        }

        
        Tensor_free(col_mins);
        Tensor_free(col_maxes);
    }    
}



void Dataset_normalize(Dataset* self, int is_same){ 
    if(is_same){
        int total = self->x->total_elements;
        float sum = 0.0f;
        for(int c = 0; c < self->x->shape[1]; c++){    
            for(int r = 0; r < self->x->shape[0]; r++){
                sum += self->x->data[r*(self->x->shape[1]) + c];
            }
        }
        float mean = sum / ((float)(total));

        float sq_diff_sum = 0.0f;
        float val;

        for(int c = 0; c<self->x->shape[0]; c++){
            for(int r = 0; r<self->x->shape[0]; r++){
                val = self->x->data[r*(self->x->shape[1]) + c];
                sq_diff_sum += (val - mean) * (val-mean);
            }
        }

        float std_dev = sqrtf(sq_diff_sum / ((float)(total)));
        if(std_dev == 0.0f) std_dev = 1.0f;


        for(int c = 0; c < self->x->shape[1]; c++){
            for(int r = 0; r < self->x->shape[0]; r++){
                int idx = r*self->x->shape[1] + c;
                float raw_val = self->x->data[idx];
                self->x->data[idx] = ((raw_val - mean) / std_dev);
            }
        }

    
    }else{
        for(int c = 0; c < self->x->shape[1]; c++){
            float sum = 0.0f;
            for(int r = 0; r < self->x->shape[0]; r++){
                sum += self->x->data[r*(self->x->shape[1]) + c];
            }
            float mean = sum / self->x->shape[0];

            float sq_diff_sum = 0.0f;
            float val;

            for(int r = 0; r<self->x->shape[0]; r++){
                val = self->x->data[r*(self->x->shape[1]) + c];
                sq_diff_sum += (val - mean) * (val-mean);
            }

            float std_dev = sqrtf(sq_diff_sum / self->x->shape[0]);

            if(std_dev == 0.0f) std_dev = 1.0f;
            

            for(int r = 0; r < self->x->shape[0]; r++){
                int idx = r*self->x->shape[1] + c;

                float raw_val = self->x->data[idx];
                self->x->data[idx] = ((raw_val - mean) / std_dev);
            }
        }
    }        
    
}





void Dataset_shuffle(Dataset* self){
    for(int i = self->num_samples - 1; i > 0; i--){
        int j = rand() % (i+1);

        int temp = self->row_indexs[i];
        self->row_indexs[i] = self->row_indexs[j];
        self->row_indexs[j] = temp;
    }
}


void Dataset_set_batch(Dataset* self, Dataset* batch, int batch_idx, int batch_size){
    
    int x_samp = 1;
    for(int d = 1; d < self->x->ndim; d++) {
        if(self->x->shape[d] > 0) x_samp *= self->x->shape[d];
    }

    int y_samp = 1;
    for(int d = 1; d < self->y->ndim; d++) {
        if(self->y->shape[d] > 0) y_samp *= self->y->shape[d];
    }

    int start_idx = batch_idx * batch_size;

    for(int i = 0; i < batch_size; i++){
        int global_sample_idx = start_idx + i;
        if(global_sample_idx >= self->num_samples) {
            batch->num_samples = i;
            //batch->x->shape[0] = i;
            //batch->y->shape[0] = i;
            return;
        }

        int actual_row = self->row_indexs[global_sample_idx];

        memcpy(&batch->x->data[i * x_samp], &self->x->data[actual_row * x_samp], x_samp * sizeof(float));
        memcpy(&batch->y->data[i * y_samp], &self->y->data[actual_row * y_samp], y_samp * sizeof(float));
    }

    batch->num_samples = batch_size;
    batch->x->shape[0] = batch_size;
    batch->y->shape[0] = batch_size;


}




void Dataset_split(Dataset* self, float train_ratio, Dataset* out_train, Dataset* out_val){
    int train_count = (int)(self->num_samples * train_ratio);
    int val_count = self->num_samples - train_count;

    out_train->x = self->x;
    out_train->y = self->y;
    out_train->num_samples = train_count;

    out_val->x = self->x;
    out_val->y = self->y;
    out_val->num_samples = val_count;

    out_train->row_indexs = malloc(train_count * sizeof(int));
    out_val->row_indexs = malloc(val_count * sizeof(int));

    memcpy(out_train->row_indexs, self->row_indexs, train_count * sizeof(int));
    memcpy(out_val->row_indexs, &self->row_indexs[train_count], val_count * sizeof(int));


}





//MiDaS-style least-squares alignment
void Tensor_align_scale_and_shift(Tensor* prediction, Tensor* target, Tensor* dest_aligned) {
    int N = prediction->total_elements;
    
    float sum_pred = 0.0f, sum_target = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < N; i++) {
        if (target->data[i] > 0.0f) {
            sum_pred += prediction->data[i];
            sum_target += target->data[i];
            valid_count++;
        }
    }

    if (valid_count == 0) return;

    float mean_pred = sum_pred / valid_count;
    float mean_target = sum_target / valid_count;

    float num = 0.0f; 
    float den = 0.0f;

    for (int i = 0; i < N; i++) {
        if (target->data[i] > 0.0f) {
            float diff_pred = prediction->data[i] - mean_pred;
            float diff_target = target->data[i] - mean_target;
            num += diff_pred * diff_target;
            den += diff_pred * diff_pred;
        }
    }

    float s = (den != 0.0f) ? (num / den) : 1.0f;
    float t = mean_target - s * mean_pred;

    for (int i = 0; i < N; i++) {
        dest_aligned->data[i] = s * prediction->data[i] + t;
    }
}




