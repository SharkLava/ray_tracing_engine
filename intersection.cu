#include <vector_types.h>
#include "intersection.h"
#include "ray.h"
#include "cuda_math.h"
#include "triangle.h"

__device__ float ray_triangle_intersection(Ray ray, Triangle triangle) {
    float3 edge1 = make_float3(triangle.v1.x - triangle.v0.x, triangle.v1.y - triangle.v0.y, triangle.v1.z - triangle.v0.z);
    float3 edge2 = make_float3(triangle.v2.x - triangle.v0.x, triangle.v2.y - triangle.v0.y, triangle.v2.z - triangle.v0.z);
    float3 h_res;
    float3 a = ray.direction;
    float3 b = edge2;
    h_res = cross(a, b);  // Using the cross product from cuda_math.h
    float3 h = h_res;
    float a_val = dot(edge1, h);  // Using the dot product from cuda_math.h

    if (a_val > -0.00001 && a_val < 0.00001)
        return -1;

    float f = 1.0f / a_val;
    float3 s = make_float3(ray.origin.x - triangle.v0.x, ray.origin.y - triangle.v0.y, ray.origin.z - triangle.v0.z);
    float u = f * dot(s, h);  // Using the dot product from cuda_math.h

    if (u < 0.0 || u > 1.0)
        return -1;

    float3 q = cross(s, edge1);  // Using the cross product from cuda_math.h
    float v = f * dot(ray.direction, q);  // Using the dot product from cuda_math.h

    if (v < 0.0 || u + v > 1.0)
        return -1;

    float t = f * dot(edge2, q);  // Using the dot product from cuda_math.h

    if (t > 0.00001)
        return t;
    else
        return -1;
}

__global__ void intersect_triangles(Ray* rays, Triangle* triangles, int num_triangles, int width, int height, Intersection* intersections) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_index = y * width + x;

    if (x < width && y < height) {
        float closest_t = 1e38f; 
        int closest_triangle_index = -1;

        for (int i = 0; i < num_triangles; ++i) {
            float t = ray_triangle_intersection(rays[pixel_index], triangles[i]);

            if (t > 0 && t < closest_t) {
                closest_t = t;
                closest_triangle_index = i;
            }
        }

        intersections[pixel_index].t = closest_t;
        intersections[pixel_index].triangle_index = closest_triangle_index;
    }
}
