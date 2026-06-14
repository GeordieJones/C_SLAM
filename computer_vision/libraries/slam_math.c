#include "slam_math.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../../neural_network/libraries/tensor.h"


Transform3D make_transform_identitiy(void){
    Transform3D T = {0};
    T.m[0][0] = 1.0f; T.m[1][1] = 1.0f;T.m[2][2] = 1.0f;T.m[3][3] = 1.0f;

    return T;
}


Point3D transform_identity_3D(const Transform3D* T, const Point3D* p){
Point3D out;
    // matrix-vector multiplication with homogomus last row
    out.x = T->m[0][0] * p->x + T->m[0][1] * p->y + T->m[0][2] * p->z + T->m[0][3];
    out.y = T->m[1][0] * p->x + T->m[1][1] * p->y + T->m[1][2] * p->z + T->m[1][3];
    out.z = T->m[2][0] * p->x + T->m[2][1] * p->y + T->m[2][2] * p->z + T->m[2][3];
    return out;
}



Point3D* project_depth_to_3d(const Tensor* depth_map, const CameraIntrinsics* K, int* valid_points_out_count){


    if(!depth_map || !K || !valid_points_out_count) return NULL;

    int height = depth_map->shape[2];
    int width = depth_map->shape[3];
    int total_pixels = height * width;

    Point3D* points = (Point3D*)malloc(total_pixels * sizeof(Point3D));
    if(!points){
        printf("ERROR: out of memory allocating 3D array\n");
        *valid_points_out_count = 0;
        return NULL;
    }

    int valid_points = 0;

    for(int v = 0; v < height; v++){
        for(int u = 0; u < width; u++){
            
            int pixel_idx = v * width + u;
            float d = depth_map->data[pixel_idx];

            if(d <= 0.0001f){
                continue;
            }

            points[valid_points].z = d;
            points[valid_points].x = ((float)u - K->cx) * d / K->fx;
            points[valid_points].y = ((float)v - K->cy) * d / K->fy;

            valid_points++;
        }
    }

    if(valid_points < total_pixels && valid_points > 0){
        Point3D* optimized_points = (Point3D*)realloc(points, valid_points * sizeof(Point3D));
        if(optimized_points){
            points = optimized_points;
        }
    }else if(valid_points == 0){
        free(points);
        points = NULL;
    }

    *valid_points_out_count = valid_points;
    return points;
}



Point3D triangulate_point(float u1, float v1, float u2, float v2, const Transform3D* T1_to_2, const CameraIntrinsics* K){

    float ray1_x = (u1 - K->cx) / K->fx;
    float ray1_y = (v1 - K->cy) / K->fy;

    float ray2_x = (u2 - K->cx) / K->fx;
    float ray2_y = (v2 - K->cy) / K->fy;

    float tx = T1_to_2->m[0][3];
    float ty = T1_to_2->m[1][3];

    float num = tx * (T1_to_2->m[0][0] * ray2_x + T1_to_2->m[1][0] * ray2_y - ray2_x) + ty * (T1_to_2->m[0][1] * ray2_x + T1_to_2->m[1][1] * ray2_y - ray2_y);
    float den = (ray1_x * (T1_to_2->m[1][0] - ray2_y * T1_to_2->m[2][0])) - (ray1_y * (T1_to_2->m[0][0] - ray2_x * T1_to_2->m[2][0]));

    float depth1 = 1.0f;
    if(fabsf(den) > 1e-5f){
        depth1 = fabsf(num / den);
    }

    Point3D estimated_point;
    estimated_point.z = depth1;
    estimated_point.x = ray1_x * depth1;
    estimated_point.y = ray1_y * depth1;

    return estimated_point;
}






float vector_magnitude(const Point3D* v) {
    return sqrtf((v->x * v->x) + (v->y * v->y) + (v->z * v->z));
}



Point3D vector_normalize(const Point3D* v) {
    float mag = vector_magnitude(v);
    Point3D norm = {0.0f, 0.0f, 0.0f};
    if (mag > 0.00001f) {
        norm.x = v->x / mag;
        norm.y = v->y / mag;
        norm.z = v->z / mag;
    }
    return norm;
}






Point3D vector_diff(const Point3D* a, const Point3D* b){
    Point3D diff;
    diff.x = a->x - b->x;
    diff.y = a->y - b->y;
    diff.z = a->z - b->z;
    return diff;
}
