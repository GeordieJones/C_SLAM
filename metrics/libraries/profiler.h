//* * Compiling with '-finstrument-functions' will automatically route all function


#ifndef PROFILER_H
#define PROFILER_H


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>


typedef struct {
    // Latency Metrics (Microseconds)
    uint64_t nn_inference_us;
    uint64_t octree_raycast_us;
    uint64_t scene_graph_us;
    uint64_t total_frame_us;
    uint64_t jitter_us;

    // Throughput & Allocation Metrics
    float    fps;
    size_t   ram_allocated_mb;
    size_t   peak_ram_mb;

    // Tracking Quality Metrics (Paper #2)
    uint32_t active_tracks_count;
    uint32_t id_switches;
    uint32_t fragmentations;
    float    avg_localization_error_cm;

    // Active Perception Metrics (Paper #2)
    float    voxel_update_skip_ratio;   // 0.0 to 1.0
    float    map_convergence_rate;     // Delta percentage
    float    voxel_divergence_error;

    // Hardware/Power Metrics
    float    gpu_wattage;
    float    cpu_temperature_c;
    float    gpu_temperature_c;
} SystemMetrics;




typedef struct ProfileNode {
    const char* function_name;
    uint64_t start_time_us;
    uint64_t total_self_time_us;  // Time minus inner/nested calls
    uint64_t total_child_time_us; // Time spent inside nested functions
    uint32_t call_count;
    struct ProfileNode* parent;
    struct ProfileNode* first_child;
    struct ProfileNode* next_sibling;
} ProfileNode;

//done
bool profiler_init(const char* csv_output_path);

//done
void profiler_clear_frame(void);

//done
void profiler_write_frame_csv(void);

//done
void profiler_shutdown(void);

//done
void profiler_set_metric(const SystemMetrics* source_metrics);

//done
void profiler_update_tracking_metrics(uint32_t active_tracks, uint32_t id_switches, uint32_t frags, float loc_err);

//done
void profiler_update_voxel_metrics(float skip_ratio, float convergence, float divergence);

//done
void profiler_track_malloc(size_t bytes);
//done
void profiler_track_free(size_t bytes);

/*
 *To use the functions above you must define this in the main and then rap all the mallocs you want to track

    void* my_custom_malloc(size_t size) {
        void* ptr = malloc(size);
        if (ptr) {
            profiler_track_malloc(size);
        }
        return ptr;
    }

    void my_custom_free(void* ptr, size_t size) {
        if (ptr) {
            free(ptr);
            profiler_track_free(size);
        }
    }
 
 * */










//forward decliration
void profiler_log_manual_block(const char* name, uint64_t duration_us);


//for microsecond timers on each function
#if defined(__GNUC__) || defined(__clang__)

void __attribute__((__no_instrument_function__)) __cyg_profile_func_enter(void *this_fn, void *call_site);

void __attribute__((__no_instrument_function__)) __cyg_profile_func_exit(void *this_fn, void *call_site);
#endif




uint64_t profiler_get_time_us(void);


typedef struct {
    const char* name;
    uint64_t start;
} ManualScopeTimer;

static inline ManualScopeTimer profiler_start_scope(const char* name) {
    ManualScopeTimer t;
    t.name = name;
    t.start = profiler_get_time_us();
    return t;
}

static inline void profiler_end_scope(ManualScopeTimer* t) {
    uint64_t duration = profiler_get_time_us() - t->start;
    // Internally routes this log into the profile node tree structure
    void profiler_log_manual_block(const char* name, uint64_t duration_us);
    profiler_log_manual_block(t->name, duration);
}












#endif
