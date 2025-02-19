#ifndef SHADING_H
#define SHADING_H

#include "intersection.h"
#include "light.h"
#include "material.h"

__global__ void shade_pixels(const Ray *__restrict__ rays,
                             const Intersection *__restrict__ intersections,
                             const Sphere *__restrict__ spheres,
                             const Material *__restrict__ materials,
                             const float3 plane_normal,
                             const Material plane_material, const Light light,
                             const int width, const int height,
                             float3 *__restrict__ output_image);

#endif
