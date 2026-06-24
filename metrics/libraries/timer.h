#ifndef TIMER_H
#define TIMER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    void* function_address;
    uint64_t start_time_us;
    uint64_t duration_us;
    uint32_t call_depth;
} CallEvent;

bool timer_init(const char* json_output_path);
void timer_clear_frame(void);
void timer_write_frame_json(uint32_t frame_id);
void timer_shutdown(void);
uint64_t timer_get_time_us(void);

#endif 
