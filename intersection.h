#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "ray.h"
#include "triangle.h"
#include <cuda_runtime.h>

typedef struct {
  float t;
  int triangle_index;
} Intersection;

__device__ float ray_triangle_intersection(Ray ray, Triangle triangle);

__global__ void intersect_triangles(Ray *rays, Triangle *triangles,
                                    int num_triangles, int width, int height,
                                    Intersection *intersections);

#endif