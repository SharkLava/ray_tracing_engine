#include "cuda_utils.h"
#include "intersection.h"
#include "ray_generation.h"
#include "shading.h"
// #include <memory>
#include "image_utils.h"
#include <vector>

// Launch parameters for kernels
struct LaunchParams {
  static constexpr int BLOCK_SIZE_X = 16;
  static constexpr int BLOCK_SIZE_Y = 16;

  static dim3 getBlockSize() { return dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y); }

  static dim3 getGridSize(int width, int height) {
    return dim3((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
  }
};

// RAII wrapper for CUDA memory
template <typename T> class CudaBuffer {
  T *ptr = nullptr;
  size_t count = 0;

public:
  CudaBuffer(size_t count) : count(count) {
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  }

  ~CudaBuffer() {
    if (ptr)
      cudaFree(ptr);
  }

  T *get() { return ptr; }
  const T *get() const { return ptr; }
  size_t size() const { return count; }

  void copyFromHost(const T *host_data) {
    CUDA_CHECK(
        cudaMemcpy(ptr, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copyToHost(T *host_data) const {
    CUDA_CHECK(
        cudaMemcpy(host_data, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
};

int main() {
  const int width = 800;
  const int height = 600;
  constexpr int NUM_SPHERES = 2;

  // Create stream for asynchronous operations
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Scene setup
  float3 camera_pos = make_float3(0.0f, 0.0f, -2.0f);
  float3 camera_dir = make_float3(0.0f, 0.0f, 1.0f);
  float fov = 90.0f;

  float3 plane_normal = make_float3(0.0f, 1.0f, 0.0f);
  float3 plane_point = make_float3(0.0f, -0.5f, 0.0f);

  // Setup scene objects
  std::vector<Sphere> spheres(NUM_SPHERES);
  spheres[0] = {{0.0f, 0.5f, 0.0f}, 0.5f};
  spheres[1] = {{0.75f, 0.0f, 0.5f}, 0.25f};

  std::vector<Material> materials(NUM_SPHERES);
  materials[0] = {
      {0.1f, 0.0f, 0.0f}, {0.8f, 0.0f, 0.0f}, {0.2f, 0.2f, 0.2f}, 32.0f};
  materials[1] = {
      {0.0f, 0.1f, 0.0f}, {0.0f, 0.8f, 0.0f}, {0.2f, 0.2f, 0.2f}, 32.0f};

  Material plane_material = {
      {0.1f, 0.1f, 0.1f}, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f}, 1.0f};

  Light light = {{-1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}};

  // Allocate device memory using RAII wrappers
  CudaBuffer<Ray> d_rays(width * height);
  CudaBuffer<Sphere> d_spheres(NUM_SPHERES);
  CudaBuffer<Material> d_materials(NUM_SPHERES);
  CudaBuffer<Intersection> d_intersections(width * height);
  CudaBuffer<float3> d_output_image(width * height);

  // Copy data to device
  d_spheres.copyFromHost(spheres.data());
  d_materials.copyFromHost(materials.data());

  // Launch kernels
  const dim3 blockSize = LaunchParams::getBlockSize();
  const dim3 gridSize = LaunchParams::getGridSize(width, height);

  generate_rays<<<gridSize, blockSize, 0, stream>>>(
      camera_pos, camera_dir, fov, width, height, d_rays.get());

  intersect_spheres<<<gridSize, blockSize, 0, stream>>>(
      d_rays.get(), d_spheres.get(), NUM_SPHERES, plane_normal, plane_point,
      width, height, d_intersections.get());

  shade_pixels<<<gridSize, blockSize, 0, stream>>>(
      d_rays.get(), d_intersections.get(), d_spheres.get(), d_materials.get(),
      plane_normal, plane_material, light, width, height, d_output_image.get());

  // Allocate pinned memory for output
  float3 *h_output_image;
  CUDA_CHECK(cudaMallocHost(&h_output_image, width * height * sizeof(float3)));

  // Copy result back to host asynchronously
  CUDA_CHECK(cudaMemcpyAsync(h_output_image, d_output_image.get(),
                             width * height * sizeof(float3),
                             cudaMemcpyDeviceToHost, stream));

  // Wait for all operations to complete
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Save the image
  saveImagePPM("output.ppm", h_output_image, width, height);

  // Cleanup
  cudaFreeHost(h_output_image);
  cudaStreamDestroy(stream);

  return 0;
}
