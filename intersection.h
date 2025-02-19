#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "ray.h"
#include "sphere.h"
#include <cuda_runtime.h>

struct alignas(8) Intersection {
  float t;
  int sphere_index;
};

__global__ void intersect_spheres(const Ray *__restrict__ rays,
                                  const Sphere *__restrict__ spheres,
                                  const int num_spheres,
                                  const float3 plane_normal,
                                  const float3 plane_point, const int width,
                                  const int height,
                                  Intersection *__restrict__ intersections);

#endif
