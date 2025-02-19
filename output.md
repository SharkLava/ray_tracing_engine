## cuda_math.cu

#include "cuda_math.h"
#include <vector_types.h>

__device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross(float3 a, float3 b) {
  float3 result;
  result.x = a.y * b.z - a.z * b.y;
  result.y = a.z * b.x - a.x * b.z;
  result.z = a.x * b.y - a.y * b.x;
  return result;
}

__device__ float length(float3 a) { return sqrtf(dot(a, a)); }

__device__ float3 normalize(float3 a) {
  float len = length(a);
  return make_float3(a.x / len, a.y / len, a.z / len);
}

__device__ float3 multiply_float3(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 multiply_float3_scalar(float3 a, float scalar) {
  return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__device__ float3 add_float3(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


## cuda_utils.cu

#include "cuda_utils.h"
#include <stdio.h>

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


## intersection.cu

#include "cuda_math.h"
#include "intersection.h"
#include "ray.h"
#include "sphere.h"
#include <vector_types.h>

#define PLANE_INDEX -1

__device__ float ray_sphere_intersection(Ray ray, Sphere sphere) {
    float3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return -1.0f;
    } else {
        float t1 = (-b - sqrt(discriminant)) / (2 * a);
        float t2 = (-b + sqrt(discriminant)) / (2 * a);

        if (t1 > 0.00001f && t2 > 0.00001f) {
            return fminf(t1, t2);
        } else if (t1 > 0.00001f) {
            return t1;
        } else if (t2 > 0.00001f) {
            return t2;
        } else {
            return -1.0f;
        }
    }
}


__device__ float ray_plane_intersection(Ray ray, float3 plane_normal, float3 plane_point) {
    float denom = dot(plane_normal, ray.direction);
    if (abs(denom) > 0.00001f) {
        float3 p0l0 = plane_point - ray.origin;
        float t = dot(p0l0, plane_normal) / denom;
        return (t > 0.00001f) ? t : -1.0f;
    }
    return -1.0f;
}


__global__ void intersect_spheres(Ray *rays, Sphere *spheres, int num_spheres, float3 plane_normal, float3 plane_point, int width, int height,
                                    Intersection *intersections) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int pixel_index = y * width + x;

  if (x < width && y < height) {
    float closest_t = 1e38f;
    int closest_sphere_index = -1;
    
    float t_plane = ray_plane_intersection(rays[pixel_index], plane_normal, plane_point);
    if (t_plane > 0.0f && t_plane < closest_t) {
        closest_t = t_plane;
    }

    for (int i = 0; i < num_spheres; ++i) {
      float t = ray_sphere_intersection(rays[pixel_index], spheres[i]);

      if (t > 0 && t < closest_t) {
        closest_t = t;
        closest_sphere_index = i;
      }
    }

    intersections[pixel_index].t = closest_t;
    intersections[pixel_index].sphere_index = closest_sphere_index;
  }
}


## main.cu

// #include "cuda_math.h"
#include "cuda_utils.h"
#include "intersection.h"
#include "light.h"
#include "material.h"
#include "ray.h"
#include "ray_generation.h"
#include "shading.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
// #include <stdio.h>
#include <vector_types.h>

void saveImagePPM(const char *filename, float3 *image, int width, int height) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write PPM header
  file << "P6" << std::endl
       << width << " " << height << std::endl
       << "255" << std::endl;

  // Convert floating-point colors to bytes and write to file
  for (int i = 0; i < width * height; i++) {
    // Clamp colors to [0,1] range
    float r = std::min(std::max(image[i].x, 0.0f), 1.0f);
    float g = std::min(std::max(image[i].y, 0.0f), 1.0f);
    float b = std::min(std::max(image[i].z, 0.0f), 1.0f);

    // Convert to bytes
    unsigned char pixel[3] = {(unsigned char)(r * 255.0f),
                              (unsigned char)(g * 255.0f),
                              (unsigned char)(b * 255.0f)};

    file.write((char *)pixel, 3);
  }

  file.close();
  std::cout << "Image saved to " << filename << std::endl;
}

int main(int argc, char **argv) {
  // Set up image dimensions
  const int width = 800;
  const int height = 600;

  // Camera setup - moved closer to see the triangle better
  float3 camera_pos = make_float3(0.0f, 0.0f, -2.0f);
  float3 camera_dir = make_float3(0.0f, 0.0f, 1.0f);
  float fov = 90.0f; // Wider FOV to see more

  // Create a plane
  float3 plane_normal = make_float3(0.0f, 1.0f, 0.0f);
  float3 plane_point = make_float3(0.0f, -0.5f, 0.0f);

  // Create two spheres
  Sphere spheres[2];
  spheres[0].center = make_float3(0.0f, 0.5f, 0.0f);
  spheres[0].radius = 0.5f;
  spheres[1].center = make_float3(0.75f, 0.0f, 0.5f);
  spheres[1].radius = 0.25f;

  Material plane_material;
  plane_material.ambient = make_float3(0.1f, 0.1f, 0.1f);
  plane_material.diffuse = make_float3(0.5f, 0.5f, 0.5f);
  plane_material.specular = make_float3(0.0f, 0.0f, 0.0f);
  plane_material.shininess = 1.0f;

  // Create materials
  Material material1;
  material1.ambient = make_float3(0.1f, 0.0f, 0.0f);
  material1.diffuse = make_float3(0.8f, 0.0f, 0.0f);
  material1.specular = make_float3(0.2f, 0.2f, 0.2f);
  material1.shininess = 32.0f;

  Material material2;
  material2.ambient = make_float3(0.0f, 0.1f, 0.0f);
  material2.diffuse = make_float3(0.0f, 0.8f, 0.0f);
  material2.specular = make_float3(0.2f, 0.2f, 0.2f);
  material2.shininess = 32.0f;

  // Create a light - moved to better illuminate the scene
  Light light;
  light.position = make_float3(-1.0f, 1.0f, -1.0f);
  light.color = make_float3(1.0f, 1.0f, 1.0f);

  // Allocate device memory
  Ray *d_rays;
  Material *d_materials;
  Intersection *d_intersections;
  float3 *d_output_image;
  float3 *d_plane_normal;
  float3 *d_plane_point;
  Material *d_plane_material;
  Sphere *d_spheres;

  CUDA_CHECK(cudaMalloc(&d_rays, width * height * sizeof(Ray)));
  CUDA_CHECK(cudaMalloc(&d_materials, 2 * sizeof(Material)));
  CUDA_CHECK(
      cudaMalloc(&d_intersections, width * height * sizeof(Intersection)));
  CUDA_CHECK(cudaMalloc(&d_output_image, width * height * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_plane_normal, sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_plane_point, sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_plane_material, sizeof(Material)));
  CUDA_CHECK(cudaMalloc(&d_spheres, 2 * sizeof(Sphere)));

  CUDA_CHECK(cudaMemcpy(d_plane_normal, &plane_normal, sizeof(float3),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_plane_point, &plane_point, sizeof(float3),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_plane_material, &plane_material, sizeof(Material),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_spheres, spheres, 2 * sizeof(Sphere),
                        cudaMemcpyHostToDevice));

  // Initialize output image to a debug color to verify memory is working
  float3 debug_color = make_float3(0.0f, 0.0f, 0.0f);
  CUDA_CHECK(cudaMemset(d_output_image, 0, width * height * sizeof(float3)));

  // Set up grid and block dimensions
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);

  // Generate rays
  generate_rays<<<grid_size, block_size>>>(camera_pos, camera_dir, fov, width,
                                           height, d_rays);
  CUDA_CHECK(cudaGetLastError());

  // Debug: Copy back some rays to verify they're generated correctly
  /*Ray *h_debug_rays = new Ray[10];
    CUDA_CHECK(cudaMemcpy(h_debug_rays, d_rays, 10 * sizeof(Ray),
                          cudaMemcpyDeviceToHost));
    std::cout << "
  First few rays:" << std::endl;
    for (int i = 0; i < 3; i++) {
      std::cout << "Ray " << i << " origin: (" << h_debug_rays[i].origin.x <<
  ","
                << h_debug_rays[i].origin.y << "," << h_debug_rays[i].origin.z
                << ")" << std::endl;
      std::cout << "Ray " << i << " direction: (" << h_debug_rays[i].direction.x
                << "," << h_debug_rays[i].direction.y << ","
                << h_debug_rays[i].direction.z << ")" << std::endl;
    }
    delete[] h_debug_rays;*/

  // Perform ray-object intersection
  intersect_spheres<<<grid_size, block_size>>>(d_rays, d_spheres, 2,
                                               plane_normal, plane_point, width,
                                               height, d_intersections);
  CUDA_CHECK(cudaGetLastError());

  // Debug: Check some intersections
  Intersection *h_debug_intersections = new Intersection[10];
  CUDA_CHECK(cudaMemcpy(h_debug_intersections, d_intersections,
                        10 * sizeof(Intersection), cudaMemcpyDeviceToHost));
  std::cout << "First few intersections:" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << "Intersection " << i << ": t=" << h_debug_intersections[i].t
              << ", sphere=" << h_debug_intersections[i].sphere_index
              << std::endl;
  }
  delete[] h_debug_intersections;

  // Shade pixels
  shade_pixels<<<grid_size, block_size>>>(
      d_rays, d_intersections, d_spheres, d_materials, plane_normal,
      plane_material, light, width, height, d_output_image);
  CUDA_CHECK(cudaGetLastError());

  // Allocate host memory for the output image
  float3 *h_output_image = new float3[width * height];

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_output_image, d_output_image,
                        width * height * sizeof(float3),
                        cudaMemcpyDeviceToHost));

  // Print some pixel colors before saving
  std::cout << "First few pixels colors:" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << "Pixel " << i << ": (" << h_output_image[i].x << ", "
              << h_output_image[i].y << ", " << h_output_image[i].z << ")"
              << std::endl;
  }

  // Save the image
  saveImagePPM("output.ppm", h_output_image, width, height);

  // Cleanup
  delete[] h_output_image;
  CUDA_CHECK(cudaFree(d_rays));
  CUDA_CHECK(cudaFree(d_materials));
  CUDA_CHECK(cudaFree(d_intersections));
  CUDA_CHECK(cudaFree(d_output_image));
  CUDA_CHECK(cudaFree(d_spheres));
  CUDA_CHECK(cudaFree(d_plane_normal));
  CUDA_CHECK(cudaFree(d_plane_point));
  CUDA_CHECK(cudaFree(d_plane_material));

  return 0;
}


## ray_generation.cu

#include <vector_types.h>
#include "ray_generation.h"
#include "cuda_math.h"

__global__ void generate_rays(float3 camera_pos, float3 camera_dir, float fov, int width, int height, Ray* rays) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Convert FOV to radians
        float fov_rad = fov * 3.14159f / 180.0f;
        float aspect_ratio = (float)width / height;
        
        // Calculate pixel positions in NDC space (-1 to 1)
        float px = (2.0f * ((x + 0.5f) / width) - 1.0f) * tan(fov_rad/2.0f) * aspect_ratio;
        float py = (1.0f - 2.0f * ((y + 0.5f) / height)) * tan(fov_rad/2.0f);
        
        // Create ray direction
        float3 ray_dir = make_float3(px, py, 1.0f);  // Forward = +z
        ray_dir = normalize(ray_dir);

        int index = y * width + x;
        rays[index].origin = camera_pos;
        rays[index].direction = ray_dir;
    }
}


## shading.cu

#include "cuda_math.h"
#include "light.h"
#include "shading.h"
#include "sphere.h"
#include <vector_types.h>

__device__ float3 calculate_diffuse(Material material, Light light,
                                    float diffuse_intensity) {
  float3 color = multiply_float3(material.diffuse, light.color);
  return multiply_float3_scalar(color, diffuse_intensity);
}

__global__ void shade_pixels(Ray *rays, Intersection *intersections,
                             Sphere *spheres, Material *materials,
                             float3 plane_normal, Material plane_material,
                             Light light, int width, int height,
                             float3 *output_image) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int pixel_index = y * width + x;

  if (x < width && y < height) {
    float3 normal;
    Material material = *materials;
    float3 intersection_point;

    if (intersections[pixel_index].sphere_index != -1) {
      intersection_point = make_float3(
          rays[pixel_index].origin.x +
              rays[pixel_index].direction.x * intersections[pixel_index].t,
          rays[pixel_index].origin.y +
              rays[pixel_index].direction.y * intersections[pixel_index].t,
          rays[pixel_index].origin.z +
              rays[pixel_index].direction.z * intersections[pixel_index].t);

      Sphere sphere = spheres[intersections[pixel_index].sphere_index];
      material = materials[intersections[pixel_index].sphere_index];

      // Calculate surface normal for sphere
      normal = normalize(intersection_point - sphere.center);
    } else {
      intersection_point = make_float3(
          rays[pixel_index].origin.x +
              rays[pixel_index].direction.x * intersections[pixel_index].t,
          rays[pixel_index].origin.y +
              rays[pixel_index].direction.y * intersections[pixel_index].t,
          rays[pixel_index].origin.z +
              rays[pixel_index].direction.z * intersections[pixel_index].t);

      normal = plane_normal;
      material = plane_material;
    }

    // Calculate light direction
    float3 light_direction =
        normalize(make_float3(light.position.x - intersection_point.x,
                             light.position.y - intersection_point.y,
                             light.position.z - intersection_point.z));

    // Calculate diffuse intensity using dot product
    float diffuse_intensity = max(dot(normal, light_direction), 0.0f);
    float3 diffuse = calculate_diffuse(material, light, diffuse_intensity);

    output_image[pixel_index] = add_float3(material.ambient, diffuse);
  }
}


## cuda_math.h

#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <vector_types.h>

__device__ float3 operator-(float3 a, float3 b);
__device__ float dot(float3 a, float3 b);
__device__ float3 cross(float3 a, float3 b);
__device__ float length(float3 a);
__device__ float3 normalize(float3 a);

__device__ float3 multiply_float3(float3 a, float3 b);
__device__ float3 multiply_float3_scalar(float3 a, float scalar);
__device__ float3 add_float3(float3 a, float3 b);

#endif


## cuda_utils.h

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

## intersection.h

#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "ray.h"
#include "sphere.h"
#include <cuda_runtime.h>

typedef struct {
  float t;
  int sphere_index;
} Intersection;


__global__ void intersect_spheres(Ray *rays, Sphere *spheres, int num_spheres, float3 plane_normal, float3 plane_point, int width, int height,
                                    Intersection *intersections);

#endif

## light.h

#ifndef LIGHT_H
#define LIGHT_H

#include <vector_types.h>

typedef struct {
  float3 position;
  float3 color;
} Light;

#endif

## material.h

#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector_types.h>

typedef struct {
  float3 ambient;
  float3 diffuse;
  float3 specular;
  float shininess;
} Material;

#endif

## ray_generation.h

#ifndef RAY_GENERATION_H
#define RAY_GENERATION_H

#include "ray.h"
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void generate_rays(float3 camera_pos, float3 camera_dir, float fov,
                              int width, int height, Ray *rays);

#endif


## ray.h

#ifndef RAY_H
#define RAY_H

#include "cuda_math.h"
#include <vector_types.h>

typedef struct {
  float3 origin;
  float3 direction;
} Ray;

#endif

## shading.h

#ifndef SHADING_H
#define SHADING_H

#include <vector_types.h>

#include "intersection.h"
#include "light.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"

__global__ void shade_pixels(Ray *rays, Intersection *intersections,
                             Sphere *spheres, Material *materials,
                             float3 plane_normal, Material plane_material,
                             Light light, int width, int height,
                             float3 *output_image);

#endif

## sphere.h

#ifndef SPHERE_H
#define SPHERE_H

#include <vector_types.h>

struct Sphere {
    float3 center;
    float radius;
};

#endif

