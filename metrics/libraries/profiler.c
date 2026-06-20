#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "profiler.h"

#define MAX_PROFILE_NODES 2048


FILE* g_csv_file = NULL;
static SystemMetrics g_metrics = {0};

static ProfileNode g_node_pool[MAX_PROFILE_NODES];
static uint32_t    g_node_pool_index = 0;

static ProfileNode* g_root_node = NULL;
static ProfileNode* g_current_node = NULL;



bool profiler_init(const char* csv_output_path) {
    g_csv_file = fopen(csv_output_path, "w");
    if (!g_csv_file) {
        return false;
    }
    
    //original header can change based on metrics used
    fprintf(g_csv_file, "frame_time_us,nn_inference_us,id_switches,gpu_wattage,function_tree\n");
    return true;
}




void __attribute__((__no_instrument_function__)) profiler_clear_frame(void) {
    g_node_pool_index = 0;
    
    g_root_node = &g_node_pool[g_node_pool_index++];
    g_root_node->function_name = "FRAME_ROOT";
    g_root_node->start_time_us = profiler_get_time_us();
    g_root_node->total_self_time_us = 0;
    g_root_node->total_child_time_us = 0;
    g_root_node->call_count = 1;
    g_root_node->parent = NULL;
    g_root_node->first_child = NULL;
    g_root_node->next_sibling = NULL;
    
    g_current_node = g_root_node;

    g_metrics.nn_inference_us = 0;
    g_metrics.octree_raycast_us = 0;
    g_metrics.scene_graph_us = 0;
    g_metrics.total_frame_us = 0;
    
    // peak_ram_mb and cumulative metrics need to persist across frames
}



static void __attribute__((__no_instrument_function__)) serialize_tree_string(ProfileNode* node, char* buffer, size_t max_len) {
    if (!node) return;
    
    char node_info[256];


    snprintf(node_info, sizeof(node_info), "%s: (%lu/%lu)| ", 
             node->function_name, 
             node->total_self_time_us, 
             (node->total_self_time_us + node->total_child_time_us));
             
    if (strlen(buffer) + strlen(node_info) < max_len) {
        strcat(buffer, node_info);
    }
    
    serialize_tree_string(node->first_child, buffer, max_len);
    
    serialize_tree_string(node->next_sibling, buffer, max_len);
}



void __attribute__((__no_instrument_function__)) profiler_write_frame_csv(void) {
    if (!g_csv_file || !g_root_node) return;

    uint64_t frame_end = profiler_get_time_us();
    uint64_t total_duration = frame_end - g_root_node->start_time_us;
    g_root_node->total_self_time_us = total_duration - g_root_node->total_child_time_us;
    g_metrics.total_frame_us = total_duration;

    char tree_string_buffer[4096] = {0};
    serialize_tree_string(g_root_node, tree_string_buffer, sizeof(tree_string_buffer));

    fprintf(g_csv_file, 
        // Latencies, Throughput, Memory
        "%lu,%lu,%lu,%lu,%lu,%.2f,%zu,%zu,"
        // Tracking & Active Perception
        "%u,%u,%u,%.3f,%.3f,%.3f,%.3f,"
        // Power/Thermals and the call tree string
        "%.2f,%.1f,%.1f,\"%s\"\n",
        
        g_metrics.nn_inference_us, g_metrics.octree_raycast_us, g_metrics.scene_graph_us, 
        g_metrics.total_frame_us, g_metrics.jitter_us, g_metrics.fps, 
        g_metrics.ram_allocated_mb, g_metrics.peak_ram_mb,
        
        g_metrics.active_tracks_count, g_metrics.id_switches, g_metrics.fragmentations, 
        g_metrics.avg_localization_error_cm, g_metrics.voxel_update_skip_ratio, 
        g_metrics.map_convergence_rate, g_metrics.voxel_divergence_error,
        
        g_metrics.gpu_wattage, g_metrics.cpu_temperature_c, g_metrics.gpu_temperature_c,
        tree_string_buffer
    );

    fflush(g_csv_file);
}



void __attribute__((__no_instrument_function__)) profiler_shutdown(void) {
    if (g_csv_file != NULL) {
        fflush(g_csv_file);
        
        fclose(g_csv_file);
        g_csv_file = NULL;
    }

    g_root_node = NULL;
    g_current_node = NULL;
    g_node_pool_index = 0;

    // zero out the global metrics memory structure
    memset(&g_metrics, 0, sizeof(SystemMetrics));
}




void __attribute__((__no_instrument_function__)) profiler_set_metric(const SystemMetrics* source_metrics) {
    if (!source_metrics) return;

    g_metrics.nn_inference_us   = source_metrics->nn_inference_us;
    g_metrics.octree_raycast_us = source_metrics->octree_raycast_us;
    g_metrics.scene_graph_us    = source_metrics->scene_graph_us;
    g_metrics.total_frame_us    = source_metrics->total_frame_us;
    g_metrics.jitter_us         = source_metrics->jitter_us;
    g_metrics.fps               = source_metrics->fps;

    g_metrics.ram_allocated_mb  = source_metrics->ram_allocated_mb;
    
    if (source_metrics->ram_allocated_mb > g_metrics.peak_ram_mb) {
        g_metrics.peak_ram_mb = source_metrics->ram_allocated_mb;
    }
    
    if (source_metrics->peak_ram_mb > g_metrics.peak_ram_mb) {
        g_metrics.peak_ram_mb = source_metrics->peak_ram_mb;
    }

    g_metrics.active_tracks_count        = source_metrics->active_tracks_count;
    g_metrics.id_switches                = source_metrics->id_switches;
    g_metrics.fragmentations             = source_metrics->fragmentations;
    g_metrics.avg_localization_error_cm  = source_metrics->avg_localization_error_cm;

    g_metrics.voxel_update_skip_ratio   = source_metrics->voxel_update_skip_ratio;
    g_metrics.map_convergence_rate      = source_metrics->map_convergence_rate;
    g_metrics.voxel_divergence_error    = source_metrics->voxel_divergence_error;

    g_metrics.gpu_wattage        = source_metrics->gpu_wattage;
    g_metrics.cpu_temperature_c  = source_metrics->cpu_temperature_c;
    g_metrics.gpu_temperature_c  = source_metrics->gpu_temperature_c;
}





void __attribute__((__no_instrument_function__)) profiler_update_tracking_metrics(uint32_t active_tracks, uint32_t id_switches, uint32_t frags, float loc_err) {
    g_metrics.active_tracks_count       = active_tracks;
    g_metrics.id_switches               = id_switches;
    g_metrics.fragmentations            = frags;
    g_metrics.avg_localization_error_cm = loc_err;
}



void __attribute__((__no_instrument_function__)) profiler_update_voxel_metrics(float skip_ratio, float convergence, float divergence) {
    g_metrics.voxel_update_skip_ratio = skip_ratio;
    g_metrics.map_convergence_rate    = convergence;
    g_metrics.voxel_divergence_error   = divergence;
}



void __attribute__((__no_instrument_function__)) profiler_track_malloc(size_t bytes) {
    double mb = (double)bytes / (1024.0 * 1024.0);

    g_metrics.ram_allocated_mb += mb;

    if (g_metrics.ram_allocated_mb > g_metrics.peak_ram_mb) {
        g_metrics.peak_ram_mb = g_metrics.ram_allocated_mb;
    }
}



void __attribute__((__no_instrument_function__)) profiler_track_free(size_t bytes) {
    double mb = (double)bytes / (1024.0 * 1024.0);

    if (mb >= g_metrics.ram_allocated_mb) {
        g_metrics.ram_allocated_mb = 0;
    } else {
        g_metrics.ram_allocated_mb -= mb;
    }
    
    // Peak memory only goes up do not subtract from it
}





