#ifndef SCENE_GRAPH_H
#define SCENE_GRAPH_H

#include "slam_math.h"

#include <stdint.h>

typedef struct OctreeNode OctreeNode;


typedef struct BoundingBox3D{
    Point3D min_bounds;
    Point3D max_bounds;
}BoundingBox3D;

typedef struct{
    uint32_t id;
    char label[32];
    BoundingBox3D bbox; //current 3D boundries
    Point3D centroid; // center of object
    Point3D velocity; // delta movement between frames

    float movability_score; // 0.0 to 1.0-constantly moving
    uint32_t frames_tracked; // total tracked frames to train from
    uint32_t frames_lost; //counter for missing frames before deleting from memory
    int is_active; //int 0 - flag for cleanup ;;;  1- curently being tracked 
}TrackedObject;

typedef struct{
    TrackedObject* objects;
    uint32_t object_count;
    uint32_t max_capacity;
    uint32_t next_avalible_id;
}SceneGraph;


SceneGraph* SceneGraph_make(uint32_t initial_capacity);

TrackedObject* TrackedObject_make(int in_id, const char* in_label, BoundingBox3D in_bbox);

BoundingBox3D BoundingBox3D_init(Point3D min_b, Point3D max_b);


void TrackedObject_free(TrackedObject* obj);

void SceneGraph_free(SceneGraph* graph);

//creates objects from the graph based on where everything is
TrackedObject* SceneGraph_extract_clusters_from_octree(OctreeNode* root, uint32_t* out_cluster_count, float dist_thresh);


float SceneGraph_calculate_iou_3d(BoundingBox3D boxA, BoundingBox3D boxB);



BoundingBox3D transform_bbox(BoundingBox3D box, Transform3D pose);


// takes the clusterd objects and maps them to existing ID's
void SceneGraph_update_tracking(SceneGraph* graph, TrackedObject* new_clusters, uint32_t cluster_count, Transform3D camera_pose, float tracking_threshold);

//checks if moving object is going to interact with static object 
//shoul use velocity to calculate how far it needs to check ahead to start looking ahead at what could be hit before they are hit
void SceneGraph_propagate_contacts(SceneGraph* graph, OctreeNode* octree_root, float dt);

#endif
