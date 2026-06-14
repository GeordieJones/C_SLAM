#include "slam_math.h"
#include "raycasting.h"
#include "scene_graph.h"

#include "../../neural_network/libraries/tensor.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


// does this need to be like raycasting.c??
SceneGraph* SceneGraph_make(uint32_t initial_capacity){

    SceneGraph* graph = (SceneGraph*)malloc(sizeof(SceneGraph));
    if (!graph) return NULL;

    graph->objects = (TrackedObject*)malloc(initial_capacity * sizeof(TrackedObject));
    if (!graph->objects) {
        free(graph);
        return NULL;
    }

    graph->object_count = 0;
    graph->max_capacity = initial_capacity;
    graph->next_avalible_id = 1;

    return graph;

}


TrackedObject* TrackedObject_make(int in_id, const char* in_label, BoundingBox3D in_bbox) {
    
    TrackedObject* obj = (TrackedObject*)malloc(sizeof(TrackedObject));
    if (!obj) return NULL;
    
    obj->id = in_id;
    
    strncpy(obj->label, in_label, sizeof(obj->label) - 1);
    obj->label[sizeof(obj->label) - 1] = '\0';
    
    obj->bbox = in_bbox;

    obj->centroid.x = (obj->bbox.max_bounds.x + obj->bbox.min_bounds.x) / 2.0f;
    obj->centroid.y = (obj->bbox.max_bounds.y + obj->bbox.min_bounds.y) / 2.0f;
    obj->centroid.z = (obj->bbox.max_bounds.z + obj->bbox.min_bounds.z) / 2.0f;

    obj->velocity.x = 0.0f;
    obj->velocity.y = 0.0f;
    obj->velocity.z = 0.0f;

    obj->movability_score = 0.75f;
    obj->frames_tracked = 1;
    obj->frames_lost = 0;
    obj->is_active = 1;

    return obj;
}



BoundingBox3D BoundingBox3D_init(Point3D min_b, Point3D max_b) {
    BoundingBox3D box;
    box.min_bounds = min_b;
    box.max_bounds = max_b;
    return box;
}

void TrackedObject_free(TrackedObject* obj) {
    if (obj) {
        free(obj);
    }
}

void SceneGraph_free(SceneGraph* graph) {
    if (!graph) return;
   
    if (graph->objects) {
        free(graph->objects);
    }
    
    free(graph);
}





TrackedObject* SceneGraph_extract_clusters_from_octree(OctreeNode* root, uint32_t* out_cluster_count, float dist_thresh){

    uint32_t max_leaves = 20000;
    Point3D* leaf_points = (Point3D*)malloc(max_leaves * sizeof(Point3D));

    int total_points = Octree_get_occupied_leaves(root, leaf_points, max_leaves);

    if (total_points == 0) {
        free(leaf_points);
        *out_cluster_count = 0;
        return NULL;
    }

    int* visited = (int*)calloc(total_points, sizeof(int));
    
   
    uint32_t cluster_capacity = 128;
    TrackedObject* found_clusters = (TrackedObject*)malloc(cluster_capacity * sizeof(TrackedObject));
    uint32_t cluster_idx = 0;

    int* queue = (int*)malloc(total_points * sizeof(int));
    
    for (int i = 0; i < total_points; i++) {
        if (visited[i] == 1) continue; 

        int head = 0;
        int tail = 0;

        visited[i] = 1;
        queue[tail] = i;
        tail++;

        float min_x = 10000.0f, min_y = 10000.0f, min_z = 10000.0f;
        float max_x = -10000.0f, max_y = -10000.0f, max_z = -10000.0f;


        while (head < tail) {
            int curr_idx = queue[head];
            head++;

            Point3D curr_pt = leaf_points[curr_idx];

            if (curr_pt.x < min_x) min_x = curr_pt.x;
            if (curr_pt.y < min_y) min_y = curr_pt.y;
            if (curr_pt.z < min_z) min_z = curr_pt.z;
            
            if (curr_pt.x > max_x) max_x = curr_pt.x;
            if (curr_pt.y > max_y) max_y = curr_pt.y;
            if (curr_pt.z > max_z) max_z = curr_pt.z;

            for (int j = 0; j < total_points; j++) {
                if (visited[j] == 1) continue;

                //two functions expect const pointers to Point3D's
                Point3D delta = vector_diff(&curr_pt, &leaf_points[j]);
                float dist = vector_magnitude(&delta);

                if (dist < dist_thresh) {
                    visited[j] = 1;
                    queue[tail] = j;
                    tail++;
                }
            }
        }

        BoundingBox3D cluster_box;
        cluster_box.min_bounds.x = min_x;
        cluster_box.min_bounds.y = min_y;
        cluster_box.min_bounds.z = min_z;
        cluster_box.max_bounds.x = max_x;
        cluster_box.max_bounds.y = max_y;
        cluster_box.max_bounds.z = max_z;


        if (cluster_idx >= cluster_capacity) {
            cluster_capacity *= 2;
            found_clusters = (TrackedObject*)realloc(found_clusters, cluster_capacity * sizeof(TrackedObject));
        }


        TrackedObject* temp_obj = TrackedObject_make(0, "unknown", cluster_box);
        found_clusters[cluster_idx] = *temp_obj;
        free(temp_obj); 
        cluster_idx++;
    }


    free(queue);
    free(visited);
    free(leaf_points);

    *out_cluster_count = cluster_idx;
    return found_clusters;

}






float SceneGraph_calculate_iou_3d(BoundingBox3D boxA, BoundingBox3D boxB) {
    float inner_min_x = fmaxf(boxA.min_bounds.x, boxB.min_bounds.x);
    float inner_min_y = fmaxf(boxA.min_bounds.y, boxB.min_bounds.y);
    float inner_min_z = fmaxf(boxA.min_bounds.z, boxB.min_bounds.z);

    float inner_max_x = fminf(boxA.max_bounds.x, boxB.max_bounds.x);
    float inner_max_y = fminf(boxA.max_bounds.y, boxB.max_bounds.y);
    float inner_max_z = fminf(boxA.max_bounds.z, boxB.max_bounds.z);

    float inner_width  = inner_max_x - inner_min_x;
    float inner_height = inner_max_y - inner_min_y;
    float inner_depth  = inner_max_z - inner_min_z;

    if (inner_width <= 0.0f || inner_height <= 0.0f || inner_depth <= 0.0f) {
        return 0.0f;
    }

    float inner_volume = inner_width * inner_height * inner_depth;
    float volume_A = (boxA.max_bounds.x - boxA.min_bounds.x) * (boxA.max_bounds.y - boxA.min_bounds.y) * (boxA.max_bounds.z - boxA.min_bounds.z);           
    float volume_B = (boxB.max_bounds.x - boxB.min_bounds.x) * (boxB.max_bounds.y - boxB.min_bounds.y) * (boxB.max_bounds.z - boxB.min_bounds.z);

    float union_volume = volume_A + volume_B - inner_volume;

    if (union_volume <= 0.0f) {
        return 0.0f;
    }

    return inner_volume / union_volume;
}





//protection from  0 matrix core dumps might need to remove if it is taking to long
static int is_matrix_zero(Transform3D pose) {
    return (pose.m[0][0] == 0.0f && pose.m[0][1] == 0.0f && pose.m[0][2] == 0.0f &&
            pose.m[1][0] == 0.0f && pose.m[1][1] == 0.0f && pose.m[1][2] == 0.0f &&
            pose.m[2][0] == 0.0f && pose.m[2][1] == 0.0f && pose.m[2][2] == 0.0f);
}


// Helper function to transform an AABB by a 4x4 camera pose matrix
BoundingBox3D transform_bbox(BoundingBox3D box, Transform3D pose) {
    Point3D corners[8];
    float x_vals[2] = {box.min_bounds.x, box.max_bounds.x};
    float y_vals[2] = {box.min_bounds.y, box.max_bounds.y};
    float z_vals[2] = {box.min_bounds.z, box.max_bounds.z};

    int idx = 0;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                corners[idx].x = x_vals[x];
                corners[idx].y = y_vals[y];
                corners[idx].z = z_vals[z];
                idx++;
            }
        }
    }

    float min_x = 10000.0f, min_y = 10000.0f, min_z = 10000.0f;
    float max_x = -10000.0f, max_y = -10000.0f, max_z = -10000.0f;

    for (int i = 0; i < 8; i++) {
        float tx = pose.m[0][0]*corners[i].x + pose.m[0][1]*corners[i].y + pose.m[0][2]*corners[i].z + pose.m[0][3];
        float ty = pose.m[1][0]*corners[i].x + pose.m[1][1]*corners[i].y + pose.m[1][2]*corners[i].z + pose.m[1][3];
        float tz = pose.m[2][0]*corners[i].x + pose.m[2][1]*corners[i].y + pose.m[2][2]*corners[i].z + pose.m[2][3];

        if (tx < min_x) min_x = tx;
        if (ty < min_y) min_y = ty;
        if (tz < min_z) min_z = tz;

        if (tx > max_x) max_x = tx;
        if (ty > max_y) max_y = ty;
        if (tz > max_z) max_z = tz;
    }

    BoundingBox3D transformed_box;
    transformed_box.min_bounds.x = min_x;
    transformed_box.min_bounds.y = min_y;
    transformed_box.min_bounds.z = min_z;
    transformed_box.max_bounds.x = max_x;
    transformed_box.max_bounds.y = max_y;
    transformed_box.max_bounds.z = max_z;

    return transformed_box;
}





void SceneGraph_update_tracking(SceneGraph* graph, TrackedObject* new_clusters, uint32_t cluster_count, Transform3D camera_pose, float tracking_threshold) {

    int valid_pose = (!is_matrix_zero(camera_pose)) ? 1 : 0;
    if (!valid_pose) {
        printf("[WARN] SceneGraph_update_tracking: Invalid camera pose. Skipping SLAM spatial alignment for this frame.\n");
    }

    
    // SLAM ALIGNMENT Shift previous object coordinates to account for camera movement
    for (uint32_t i = 0; i < graph->object_count; i++) {
        if (!graph->objects[i].is_active) continue;

        Point3D old_cent = graph->objects[i].centroid;
        graph->objects[i].centroid.x = camera_pose.m[0][0]*old_cent.x + camera_pose.m[0][1]*old_cent.y + camera_pose.m[0][2]*old_cent.z + camera_pose.m[0][3];
        graph->objects[i].centroid.y = camera_pose.m[1][0]*old_cent.x + camera_pose.m[1][1]*old_cent.y + camera_pose.m[1][2]*old_cent.z + camera_pose.m[1][3];
        graph->objects[i].centroid.z = camera_pose.m[2][0]*old_cent.x + camera_pose.m[2][1]*old_cent.y + camera_pose.m[2][2]*old_cent.z + camera_pose.m[2][3];
        
        graph->objects[i].bbox = transform_bbox(graph->objects[i].bbox, camera_pose);
    }


    for (uint32_t i = 0; i < graph->object_count; i++) {
        if (!graph->objects[i].is_active) continue;

        int best_idx = -1;
        float best_iou = -1.0f;

        for (uint32_t j = 0; j < cluster_count; j++) {
            if (new_clusters[j].id != 0) continue; 

            float iou = SceneGraph_calculate_iou_3d(graph->objects[i].bbox, new_clusters[j].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = j;
            }
        }


        if (best_idx != -1 && best_iou >= tracking_threshold) {
            Point3D new_cent = new_clusters[best_idx].centroid;

            Point3D diff = vector_diff(&new_cent, &(graph->objects[i].centroid));
            float vel_mag = vector_magnitude(&diff);
            

            graph->objects[i].bbox = new_clusters[best_idx].bbox;
            graph->objects[i].centroid = new_cent;
            graph->objects[i].frames_lost = 0;
            graph->objects[i].frames_tracked++;

            float noise_floor = 0.05f;      // 5 cm/s: sensor jitters
            float maximum_impact = 1.5f;    // 1.5 m/s: trigger instant high movability

            float velocity_factor = 0.0f;
            if (vel_mag > noise_floor) {
                velocity_factor = (vel_mag - noise_floor) / (maximum_impact - noise_floor);
                if (velocity_factor > 1.0f) velocity_factor = 1.0f; 
            }
            
            
            if (velocity_factor > 0.5f) {
                graph->objects[i].movability_score = (graph->objects[i].movability_score * 0.4f) + (velocity_factor * 0.6f);
            } else {
                graph->objects[i].movability_score = (graph->objects[i].movability_score * 0.8f) + (velocity_factor * 0.2f);
            }



            new_clusters[best_idx].id = graph->objects[i].id; 
        } else {
            graph->objects[i].frames_lost++;
            if (graph->objects[i].frames_lost >= 10) {
                graph->objects[i].is_active = 0;
            }
        }
    }

    for (uint32_t j = 0; j < cluster_count; j++) {
        if (new_clusters[j].id == 0) {
            if (graph->object_count < graph->max_capacity) {
                uint32_t new_id = graph->next_avalible_id;
                graph->next_avalible_id++;

                graph->objects[graph->object_count] = new_clusters[j];
                graph->objects[graph->object_count].id = new_id;
                graph->objects[graph->object_count].is_active = 1;
                graph->objects[graph->object_count].frames_tracked = 1;
                graph->objects[graph->object_count].frames_lost = 0;
                graph->objects[graph->object_count].movability_score = 0.5f;

                graph->object_count++;
            }
        }
    }
}


void SceneGraph_propagate_contacts(SceneGraph* graph, OctreeNode* octree_root, float dt) {
    
    for (uint32_t i = 0; i < graph->object_count; i++) {
        if (!graph->objects[i].is_active) continue;

        if (graph->objects[i].movability_score <= 0.35f) continue;

        TrackedObject* obj = &graph->objects[i];
        BoundingBox3D box = obj->bbox;

        Point3D delta;
        delta.x = obj->velocity.x * dt;
        delta.y = obj->velocity.y * dt;
        delta.z = obj->velocity.z * dt;

        BoundingBox3D pred_box;
        pred_box.min_bounds.x = (delta.x < 0.0f) ? box.min_bounds.x + delta.x : box.min_bounds.x;
        pred_box.max_bounds.x = (delta.x > 0.0f) ? box.max_bounds.x + delta.x : box.max_bounds.x;

        pred_box.min_bounds.y = (delta.y < 0.0f) ? box.min_bounds.y + delta.y : box.min_bounds.y;
        pred_box.max_bounds.y = (delta.y > 0.0f) ? box.max_bounds.y + delta.y : box.max_bounds.y;

        pred_box.min_bounds.z = (delta.z < 0.0f) ? box.min_bounds.z + delta.z : box.min_bounds.z;
        pred_box.max_bounds.z = (delta.z > 0.0f) ? box.max_bounds.z + delta.z : box.max_bounds.z;

        // Call to the recursive tree search in raycasting.c
        Octree_check_predictive_collision(octree_root, pred_box, obj->id);
    }
}



