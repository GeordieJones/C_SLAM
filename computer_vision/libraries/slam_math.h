#ifndef SLAM_MATH_H
#define SLAM_MATH_H

#include "../../neural_network/libraries/tensor.h"


typedef struct {
    float x;
    float y;
    float z;
} Point3D;

// Camera Intrinsics Configuration
typedef struct {
    float fx; // Focal length along the horizontal axis
    float fy; // Focal length along the vertical axis
    float cx; // Optical center x-coordinate (principal point)
    float cy; // Optical center y-coordinate (principal point)
} CameraIntrinsics;

// 4x4 Transformation Matrix (Coordinates for Camera Pose Tracking)
typedef struct {
    float m[4][4];
} Transform3D;


Transform3D make_transform_identity(void);

Point3D transform_identity_3D(const Transform3D* T, const Point3D* p);

// depth_map shape [1, 1, height, width] for pixel depths
Point3D* project_depth_to_3d(const Tensor* depth_map, const CameraIntrinsics* K, int* vaild_points_out_count);


//Triangulates a single 3D point from corresponding pixel points 
Point3D triangulate_point(float u1, float v1, float u2, float v2, const Transform3D* T1_to_2, const CameraIntrinsics* K);


float vector_magnitude(const Point3D* v);

Point3D vector_diff(const Point3D* a, const Point3D* b);

Point3D vector_normalize(const Point3D* v);

#endif 
