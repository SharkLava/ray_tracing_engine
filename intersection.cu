#include "cuda_math.h"
#include "intersection.h"

__device__ __forceinline__ float ray_sphere_intersection(const Ray &ray,
                                                         const Sphere &sphere) {
  float3 oc = ray.origin - sphere.center;
  float b = dot(oc, ray.direction);
  float c = dot(oc, oc) - sphere.radius * sphere.radius;
  float h = b * b - c;

  if (h < 0.0f)
    return -1.0f;
  h = sqrtf(h);
  float t = -b - h;
  return (t > 0.00001f) ? t : ((-b + h > 0.00001f) ? -b + h : -1.0f);
}

__device__ __forceinline__ float
ray_plane_intersection(const Ray &ray, const float3 &plane_normal,
                       const float3 &plane_point) {
  float denom = dot(plane_normal, ray.direction);
  if (fabsf(denom) > 0.00001f) {
    float3 p0l0 = plane_point - ray.origin;
    float t = dot(p0l0, plane_normal) / denom;
    return (t > 0.00001f) ? t : -1.0f;
  }
  return -1.0f;
}

__global__ void intersect_spheres(const Ray *__restrict__ rays,
                                  const Sphere *__restrict__ spheres,
                                  const int num_spheres,
                                  const float3 plane_normal,
                                  const float3 plane_point, const int width,
                                  const int height,
                                  Intersection *__restrict__ intersections) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const int pixel_index = y * width + x;
  const Ray &ray = rays[pixel_index];

  float closest_t = 1e38f;
  int closest_sphere_index = -1;

  // Check plane intersection first (likely to be hit often)
  float t_plane = ray_plane_intersection(ray, plane_normal, plane_point);
  if (t_plane > 0.0f && t_plane < closest_t) {
    closest_t = t_plane;
    closest_sphere_index = -1;
  }

// Check sphere intersections
#pragma unroll
  for (int i = 0; i < num_spheres; ++i) {
    float t = ray_sphere_intersection(ray, spheres[i]);
    if (t > 0.0f && t < closest_t) {
      closest_t = t;
      closest_sphere_index = i;
    }
  }

  intersections[pixel_index] = {closest_t, closest_sphere_index};
}
