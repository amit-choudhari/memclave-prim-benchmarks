/**
* app.c
* VA Host Application Source File
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

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"
#include "../support/prim_results.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

#define LOG_WORDS 8
#define LOG_MAGIC 0x534B4C4F475631ULL

// Pointer declaration
static T* A;
static T* B;
static T* C;
static T* C2;

// Create input arrays
static void read_input(T* A, T* B, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand());
        B[i] = (T) (rand());
    }
}

// Compute output in the host
static void vector_addition_host(T* C, T* A, T* B, unsigned int nr_elements) {
    for (unsigned int i = 0; i < nr_elements; i++) {
        C[i] = A[i] + B[i];
    }
}

// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s) %d\n", nr_of_dpus, p.input_size);
    unsigned int i = 0;

    const unsigned int input_size = p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size; // Total input size (weak or strong scaling)
    const unsigned int input_size_8bytes = 
        ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size; // Input size per DPU (max.), 8-byte aligned
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned

    // Input/output allocation
    A = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    B = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C2 = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    T *bufferA = A;
    T *bufferB = B;
    T *bufferC = C2;

    // Create an input file with arbitrary data
    read_input(A, B, input_size);

    // Timer declaration
    Timer timer;

    printf("NR_TASKLETS\t%d\tBL\t%d\n", NR_TASKLETS, BL);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Compute output on CPU (performance comparison and verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        vector_addition_host(C, A, B, input_size);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);
        // Input arguments
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments[NR_DPUS];
        for(i=0; i<nr_of_dpus-1; i++) {
            input_arguments[i].size=input_size_dpu_8bytes * sizeof(T); 
            input_arguments[i].transfer_size=input_size_dpu_8bytes * sizeof(T); 
            input_arguments[i].kernel=kernel;
        }
        input_arguments[nr_of_dpus-1].size=(input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS-1)) * sizeof(T); 
        input_arguments[nr_of_dpus-1].transfer_size=input_size_dpu_8bytes * sizeof(T); 
        input_arguments[nr_of_dpus-1].kernel=kernel;

        // Copy input arrays
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferB + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        if(rep >= p.n_warmup)
            stop(&timer, 1);

        printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup);
            #if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
            #endif
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 2);
            #if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
            #endif
        }
			{
			/* gather 64B log from each DPU by symbol name */
			uint64_t logs[NR_DPUS][LOG_WORDS];

			struct dpu_set_t dpu;
			uint32_t idx = 0;
			DPU_FOREACH(dpu_set, dpu) {
			    dpu_prepare_xfer(dpu, &logs[idx][0]);
			    idx++;
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "sk_log",
			                         0, LOG_WORDS * sizeof(uint64_t),
			                         DPU_XFER_DEFAULT));

			/* reduce max cycles over DPUs (only those that wrote a valid record) */
			uint64_t max_cycles = 0;
			for (uint32_t i = 0; i < idx; i++) {
 			   if (logs[i][0] == LOG_MAGIC) {
			        if (logs[i][1] > max_cycles) max_cycles = logs[i][1];
			    }
			}

			/* print an easy-to-compare line */
			printf("DPU cycles (whole-kernel, max over DPUs): %llu\n",
			       (unsigned long long)max_cycles);

			}

#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

        printf("Retrieve results\n");
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);
        i = 0;
        // PARALLEL RETRIEVE TRANSFER
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferC + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        if(rep >= p.n_warmup)
            stop(&timer, 3);

    }

    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, p.n_reps);
    // update CSV
#define TEST_NAME "VA"
#define RESULTS_FILE "../prim_results.csv"
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "U_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "U_D2C");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "UPMEM");

#if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
#endif	

    // Check output
    bool status = true;
    for (i = 0; i < input_size; i++) {
        if(C[i] != bufferC[i]){ 
            status = false;
#if PRINT
            printf("%d: %u -- %u\n", i, C[i], bufferC[i]);
#endif
        }
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A);
    free(B);
    free(C);
    free(C2);
    DPU_ASSERT(dpu_free(dpu_set));
	
    return status ? 0 : -1;
}
