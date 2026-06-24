#include "image_handler.h"
#include <stdio.h>
#include <stdlib.h>

void Diagnostic_dump_raw_rgb(const char* prefix, unsigned char* raw_rgb_frame, int width, int height) {
    if (!prefix || !raw_rgb_frame) return;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s_1_original.ppm", prefix);
    
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(raw_rgb_frame, 1, width * height * 3, f);
    fclose(f);
}

void Diagnostic_dump_depth(const char* prefix, Tensor* preprocessed_tensor, Tensor* depth_prediction) {
    if (!prefix || !preprocessed_tensor || !depth_prediction) return;

    char filename[512];

    snprintf(filename, sizeof(filename), "%s_2_preprocessed.pgm", prefix);
    FILE* f_gray = fopen(filename, "wb");
    if (f_gray) {
        int p_h = preprocessed_tensor->shape[0];
        int p_w = preprocessed_tensor->shape[1];
        
        fprintf(f_gray, "P5\n%d %d\n255\n", p_w, p_h);
        for (int i = 0; i < p_h * p_w; i++) {
            unsigned char pixel = (unsigned char)(preprocessed_tensor->data[i] * 255.0f);
            fputc(pixel, f_gray);
        }
        fclose(f_gray);
    }

    snprintf(filename, sizeof(filename), "%s_3_depth_map.txt", prefix);
    FILE* f_depth = fopen(filename, "w");
    if (f_depth) {
        int d_h = depth_prediction->shape[0];
        int d_w = depth_prediction->shape[1];
        
        fprintf(f_depth, "%d %d\n", d_h, d_w);
        for (int i = 0; i < d_h * d_w; i++) {
            fprintf(f_depth, "%f ", depth_prediction->data[i]);
        }
        fclose(f_depth);
    }
}

void Diagnostic_dump_tracking(const char* prefix, TrackedObject* objects, uint32_t num_objects) {
    if (!prefix || !objects || num_objects == 0) return;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s_4_tracking.csv", prefix);
    
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "id,label,is_active,movability_score,frames_tracked,frames_lost,vx,vy,vz,x_min,y_min,z_min,x_max,y_max,z_max\n");
    for (uint32_t v = 0; v < num_objects; v++) {
        fprintf(f, "%u,%s,%d,%f,%u,%u,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                objects[v].id,
                objects[v].label,
                objects[v].is_active,
                objects[v].movability_score,
                objects[v].frames_tracked,
                objects[v].frames_lost,
                objects[v].velocity.x, objects[v].velocity.y, objects[v].velocity.z,
                objects[v].bbox.min_bounds.x, objects[v].bbox.min_bounds.y, objects[v].bbox.min_bounds.z,
                objects[v].bbox.max_bounds.x, objects[v].bbox.max_bounds.y, objects[v].bbox.max_bounds.z);
    }
    fclose(f);
}
