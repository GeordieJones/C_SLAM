#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "timer.h"

#define MAX_CALL_TIMELINE_EVENTS 8192

static FILE* g_json_file = NULL;

// Contiguous block to completely avoid malloc overhead during execution
static CallEvent g_timeline[MAX_CALL_TIMELINE_EVENTS];
static uint32_t g_timeline_index = 0;
static uint32_t g_current_depth = 0;

bool __attribute__((__no_instrument_function__)) timer_init(const char* json_output_path) {
    g_json_file = fopen(json_output_path, "w");
    if (!g_json_file) {
        return false;
    }
    fprintf(g_json_file, "[\n");
    return true;
}

void __attribute__((__no_instrument_function__)) timer_clear_frame(void) {
    g_timeline_index = 0;
    g_current_depth = 0;
}

void __attribute__((__no_instrument_function__)) timer_write_frame_json(uint32_t frame_id) {
    if (!g_json_file) return;

    fprintf(g_json_file, "  {\n");
    fprintf(g_json_file, "    \"frame_id\": %u,\n", frame_id);
    fprintf(g_json_file, "    \"execution_timeline\": [\n");
    
    for (uint32_t i = 0; i < g_timeline_index; i++) {
        fprintf(g_json_file, "      {\n");
        fprintf(g_json_file, "        \"call_order\": %u,\n", i);
        fprintf(g_json_file, "        \"function_address\": \"%p\",\n", g_timeline[i].function_address);
        fprintf(g_json_file, "        \"start_time_us\": %lu,\n", g_timeline[i].start_time_us);
        fprintf(g_json_file, "        \"duration_us\": %lu,\n", g_timeline[i].duration_us);
        fprintf(g_json_file, "        \"nesting_depth\": %u\n", g_timeline[i].call_depth);
        fprintf(g_json_file, "      }%s\n", (i + 1 < g_timeline_index) ? "," : "");
    }
    
    fprintf(g_json_file, "    ]\n");
    
    // Manage trailing comma cleanly
    long current_pos = ftell(g_json_file);
    fflush(g_json_file);
    fprintf(g_json_file, "  },\n"); 
}

void __attribute__((__no_instrument_function__)) timer_shutdown(void) {
    if (g_json_file != NULL) {
        long pos = ftell(g_json_file);
        if (pos > 3) {
            fseek(g_json_file, pos - 2, SEEK_SET);
        }
        fprintf(g_json_file, "\n]\n");
        fclose(g_json_file);
        g_json_file = NULL;
    }
}

uint64_t __attribute__((__no_instrument_function__)) timer_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000ULL) + ((uint64_t)ts.tv_nsec / 1000ULL);
}

#if defined(__GNUC__) || defined(__clang__)

static uint64_t g_start_stack[MAX_CALL_TIMELINE_EVENTS];
static uint32_t g_stack_ptr = 0;

void __attribute__((__no_instrument_function__)) __cyg_profile_func_enter(void *this_fn, void *call_site) {
    (void)call_site;
    if (g_stack_ptr < MAX_CALL_TIMELINE_EVENTS) {
        g_start_stack[g_stack_ptr++] = timer_get_time_us();
    }
    g_current_depth++;
}

void __attribute__((__no_instrument_function__)) __cyg_profile_func_exit(void *this_fn, void *call_site) {
    (void)call_site;
    uint64_t end_time = timer_get_time_us();
    
    if (g_current_depth > 0) g_current_depth--;
    if (g_stack_ptr == 0) return;

    uint64_t start_time = g_start_stack[--g_stack_ptr];

    if (g_timeline_index < MAX_CALL_TIMELINE_EVENTS) {
        g_timeline[g_timeline_index] = (CallEvent){
            .function_address = this_fn,
            .start_time_us = start_time,
            .duration_us = end_time - start_time,
            .call_depth = g_current_depth
        };
        g_timeline_index++;
    }
}
#endif
