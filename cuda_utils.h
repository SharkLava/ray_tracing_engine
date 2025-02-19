#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking function that prints the error location and message
inline void checkCudaError(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

// Macro to wrap CUDA calls with automatic error checking
// Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    checkCudaError(error, __FILE__, __LINE__);                                 \
  } while (0)

// Optional: Helper template for safe CUDA memory management
template <typename T> class CudaMemory {
private:
  T *ptr = nullptr;
  size_t elements = 0;

public:
  CudaMemory(size_t count) : elements(count) {
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  }

  ~CudaMemory() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }

  // Prevent copying
  CudaMemory(const CudaMemory &) = delete;
  CudaMemory &operator=(const CudaMemory &) = delete;

  // Allow moving
  CudaMemory(CudaMemory &&other) noexcept
      : ptr(other.ptr), elements(other.elements) {
    other.ptr = nullptr;
    other.elements = 0;
  }

  // Copy data from host to device
  void copyFromHost(const T *host_data) {
    CUDA_CHECK(cudaMemcpy(ptr, host_data, elements * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  // Copy data from device to host
  void copyToHost(T *host_data) const {
    CUDA_CHECK(cudaMemcpy(host_data, ptr, elements * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  // Get raw pointer (for kernel launches)
  T *get() { return ptr; }
  const T *get() const { return ptr; }

  // Get number of elements
  size_t size() const { return elements; }
};

#endif
