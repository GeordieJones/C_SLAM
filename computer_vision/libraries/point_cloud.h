#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

#include "slam_math.h"

typedef struct{
    Point3D* points;
    int count;         //current num of points
    int capacity;      //max points that the current cloud can hold
} PointCloud;

PointCloud* PointCloud_make(int initial_capacity);

void PointCloud_add_point(PointCloud* cloud, Point3D p);

void PointCloud_free(PointCloud* cloud);

//1 for good 0 for error
int PointCloud_export_ply(const PointCloud* cloud, const char* filename);

//voxels_size is the side length of each cube grid in meters 0.05f = 5cm
void PointCloud_voxel_downsample(PointCloud* cloud, float voxel_size);


#endif
