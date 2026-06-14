#include "point_cloud.h"
#include "slam_math.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>






PointCloud* PointCloud_make(int initial_capactiy){
    PointCloud* cloud = (PointCloud*)malloc(sizeof(PointCloud));

    if(!cloud){
        printf("ERROR: no point cloud made failed to make\n");
        return NULL;
    }
    
    cloud->points = (Point3D*)malloc(initial_capacity * sizeof(Point3D));
    if (!cloud->points) {
        printf("ERROR: Failed to allocate point cloud memory array.\n");
        free(cloud);
        return NULL;
    }


    cloud->capacity = initial_capacity;
    cloud->count = 0;

    return cloud;
}



void PointCloud_add_point(PointCloud* cloud, Point3D p){

    if(cloud->count >= cloud->capacity){
        int new_capacity = cloud->capacity * 2;
        Point3D* new_points = (Point3D*)realloc(cloud->points, new_capacity * sizeof(Point3D));

        if (!new_points) {
            printf("ERROR: Out of memory while expanding point cloud.\n");
            return;
        }
        cloud->points = new_points;
        cloud->capacity = new_capacity;
    }

    cloud->points[cloud->count] = p;
    cloud->count++;
}




void PointCloud_free(PointCloud* cloud){
    if (cloud) {
        if (cloud->points) free(cloud->points);
        free(cloud);
    }
}




int PointCloud_export_ply(PointCloud* cloud, const char* filename){
    
    FILE* file = fopen(filename, "w");
    if(!file){
        fprintf(stderr, "ERROR: could not open file %s\n", filename);
        return NULL;
    }

    fprintf(file, "ply\n");
    fprintf(file, "format ascii 1.0\n");
    fprintf(file, "element vertex %d\n", cloud->count);
    fprintf(file, "property float x\n");
    fprintf(file, "property float y\n");
    fprintf(file, "property float z\n");
    fprintf(file, "end_header\n");
    
    for(int i = 0; i < cloud->count; i++) {
        fprintf(file, "%f %f %f\n", cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
    }

    if(ferror(file)){
        printf("ERROR: OS error occured in writing to .ply file\n");
        fclose(file);
        return 0;
    }else{
        printf("ply succesfully saved\n");
        fclose(file);
        return 1;
    }
    //catch all
    fclose(file);
}


typedef struct VoxelBucket {
    uint64_t hash_key;     // Unique spatial identifier
    Point3D sum_points;    // Accumulated coordinate totals 
    int point_count;       // Total individual points 
    struct VoxelBucket* next; // Linked list pointe
} VoxelBucket;

typedef struct {
    VoxelBucket** buckets;
    int table_size;
} SpatialHashTable;



void PointCloud_voxel_downsample(PointCloud* cloud, float voxel_size){

    if (!cloud || cloud->count == 0 || voxel_size <= 0.0001f) return;
    

    int table_size = cloud->count * 2 + 3; 
    SpatialHashTable table;
    table.table_size = table_size;
    table.buckets = (VoxelBucket**)calloc(table_size, sizeof(VoxelBucket*));
    if (!table.buckets) return; 

    
    for (int i = 0; i < cloud->count; i++) {
        Point3D p = cloud->points[i];
        
        int gx = (int)floorf(p.x / voxel_size);
        int gy = (int)floorf(p.y / voxel_size);
        int gz = (int)floorf(p.z / voxel_size);

        //randomly genrated primes
        uint64_t hash_key = ((uint64_t)gx * 73856093) ^ ((uint64_t)gy * 19349663) ^ ((uint64_t)gz * 83492791);
        
        int table_idx = (int)(hash_key % table_size);

        VoxelBucket* curr = table.buckets[table_idx];
        VoxelBucket* target_bucket = NULL;

        while (curr != NULL) {
            if (curr->hash_key == hash_key) {
                target_bucket = curr;
                break;
            }
            curr = curr->next;
        }

        if (target_bucket == NULL) {
            target_bucket = (VoxelBucket*)malloc(sizeof(VoxelBucket));
            target_bucket->hash_key = hash_key;
            target_bucket->sum_points.x = 0.0f;
            target_bucket->sum_points.y = 0.0f;
            target_bucket->sum_points.z = 0.0f;
            target_bucket->point_count = 0;

            // Push to front of linked list chain (O(1))
            target_bucket->next = table.buckets[table_idx];
            table.buckets[table_idx] = target_bucket;
        }

        target_bucket->sum_points.x += p.x;
        target_bucket->sum_points.y += p.y;
        target_bucket->sum_points.z += p.z;
        target_bucket->point_count++;

    }


    int write_idx = 0;
    for (int b = 0; b < table_size; b++) {
        VoxelBucket* curr = table.buckets[b];

        while (curr != NULL) {
            cloud->points[write_idx].x = curr->sum_points.x / (float)curr->point_count;
            cloud->points[write_idx].y = curr->sum_points.y / (float)curr->point_count;
            cloud->points[write_idx].z = curr->sum_points.z / (float)curr->point_count;
            write_idx++;


            VoxelBucket* temp = curr;
            curr = curr->next;
            free(temp);
        }
    }

    cloud->count = write_idx;
    free(table.buckets);
}

















