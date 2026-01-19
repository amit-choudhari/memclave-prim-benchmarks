/*
* Vector addition with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// vector_addition: Computes the vector addition of a cached block 
static void vector_addition(T *bufferB, T *bufferA, unsigned int l_size) {
    for (unsigned int i = 0; i < l_size; i++){
        bufferB[i] += bufferA[i];
    }
}

#define LOG_WORDS 8
#define LOG_MAGIC 0x534B4C4F475631ULL /* "SKLOGV1" */
__mram_noinit uint64_t sk_log[LOG_WORDS];
__dma_aligned static uint32_t tl_cycles[NR_TASKLETS];
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);

int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) { 
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
}

// main_kernel1
int main_kernel1() {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
	if (me() == 0) {
	    perfcounter_config(COUNT_CYCLES, true);
	}
    // Barrier
    barrier_wait(&my_barrier);
    uint32_t t0 = perfcounter_get();

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in bytes

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

    // Initialize a local cache to store the MRAM block
    T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_B = (T *) mem_alloc(BLOCK_SIZE);

    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // Bound checking
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*)(mram_base_addr_A + byte_index), cache_A, l_size_bytes);
        mram_read((__mram_ptr void const*)(mram_base_addr_B + byte_index), cache_B, l_size_bytes);

        // Computer vector addition
        vector_addition(cache_B, cache_A, l_size_bytes >> DIV);

        // Write cache to current MRAM block
        mram_write(cache_B, (__mram_ptr void*)(mram_base_addr_B + byte_index), l_size_bytes);

    }
    barrier_wait(&my_barrier);
	uint32_t t1 = perfcounter_get();

	/* store each tasklet’s cycles */
	tl_cycles[me()] = t1 - t0;
    barrier_wait(&my_barrier);

	/* tasklet 0 reduces to max and writes a 64B record into MRAM */
	if (me() == 0) {
	    uint64_t mx = 0;
	    for (int t = 0; t < NR_TASKLETS; t++)
	        if (tl_cycles[t] > mx) mx = tl_cycles[t];

	    uint64_t rec[LOG_WORDS] = {
	        LOG_MAGIC,        /* [0] magic */
	        mx,               /* [1] max cycles across tasklets */
	        (uint64_t)t0,     /* [2] start snapshot (optional) */
	        (uint64_t)t1,     /* [3] end   snapshot (optional) */
	        (uint64_t)NR_TASKLETS, /* [4] tasklet count (aux) */
	        0,                /* [5] spare */
	        0,                /* [6] spare */
	        1                 /* [7] done flag */
	    };
	    mram_write(rec, (__mram_ptr void *)sk_log, sizeof rec);
	}

    return 0;
}
