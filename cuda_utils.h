#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void checkCudaError(cudaError_t error, const char *file, int line);

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    checkCudaError(error, __FILE__, __LINE__);                                 \
  } while (0)

#endif