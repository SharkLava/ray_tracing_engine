#ifndef SHADING_H
#define SHADING_H

#include <vector_types.h>

#include "intersection.h"
#include "light.h"
#include "material.h"
#include "ray.h"
#include "triangle.h"

__global__ void shade_pixels(Ray *rays, Intersection *intersections,
                             Triangle *triangles, Material *materials,
                             Light light, int width, int height,
                             float3 *output_image);

#endif