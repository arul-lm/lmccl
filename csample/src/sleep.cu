#include <time.h>

// CUDA kernel to pause for at least num_cycle cycles
__global__ void sleep(int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock64();
    while(cycles < num_cycles) {
        cycles = clock64() - start;
    }
}

// Launches a kernel that sleeps for num_cycles
extern "C" void sleep_kernel(int64_t num_cycles, cudaStream_t stream)
{
    // Launch a single GPU thread to sleep
    int blockSize, gridSize;
    blockSize = 1;
    gridSize = 1;
 
    // Execute the kernel
    sleep<<< gridSize, blockSize, 0, stream >>>(num_cycles);
}