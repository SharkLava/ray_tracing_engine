#include "cuda_utils.h"
#include <stdio.h>

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
