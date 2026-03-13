/**
* app.c
* subkernel-load Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>

#if ENERGY
#include <dpu_probe.h>
#endif

// Define the DPU Binary path as DPU_BINARY here
#define DPU_BINARY "./bin/sk-load-bench"

static uint64_t time_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int perform_benchmark(bool auth_only) {
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

    uint64_t launch_time = time_ms();

    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    printf("INFO: SK load (baseline): %lums.\n", time_ms() - launch_time);

    DPU_ASSERT(dpu_free(dpu_set));
    return EXIT_SUCCESS;

error:
    printf("cannot run benchmark\n");
    return EXIT_FAILURE;
}

int main(void) {
    return perform_benchmark(true);
}
