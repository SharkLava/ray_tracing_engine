#ifndef RAY_GENERATION_H
#define RAY_GENERATION_H

#include "ray.h"
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void generate_rays(float3 camera_pos, float3 camera_dir, float fov,
                              int width, int height, Ray *rays);

#endif