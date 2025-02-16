#include "cuda_math.h"
#include "light.h"
#include "shading.h"
#include <vector_types.h>

__device__ float3 calculate_diffuse(Material material, Light light,
                                    float diffuse_intensity) {
  float3 color = multiply_float3(material.diffuse, light.color);
  return multiply_float3_scalar(color, diffuse_intensity);
}

__global__ void shade_pixels(Ray *rays, Intersection *intersections,
                             Triangle *triangles, Material *materials,
                             Light light, int width, int height,
                             float3 *output_image) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int pixel_index = y * width + x;

  if (x < width && y < height) {
    if (intersections[pixel_index].triangle_index != -1) {
      float3 intersection_point = make_float3(
          rays[pixel_index].origin.x +
              rays[pixel_index].direction.x * intersections[pixel_index].t,
          rays[pixel_index].origin.y +
              rays[pixel_index].direction.y * intersections[pixel_index].t,
          rays[pixel_index].origin.z +
              rays[pixel_index].direction.z * intersections[pixel_index].t);

      Triangle triangle = triangles[intersections[pixel_index].triangle_index];
      Material material = materials[intersections[pixel_index].triangle_index];

      // Calculate surface normal
      float3 edge1 = make_float3(triangle.v1.x - triangle.v0.x,
                                 triangle.v1.y - triangle.v0.y,
                                 triangle.v1.z - triangle.v0.z);
      float3 edge2 = make_float3(triangle.v2.x - triangle.v0.x,
                                 triangle.v2.y - triangle.v0.y,
                                 triangle.v2.z - triangle.v0.z);
      float3 normal = normalize(cross(edge1, edge2));

      // Calculate light direction
      float3 light_direction =
          normalize(make_float3(light.position.x - intersection_point.x,
                                light.position.y - intersection_point.y,
                                light.position.z - intersection_point.z));

      // Calculate diffuse intensity using dot product
      float diffuse_intensity = max(dot(normal, light_direction), 0.0f);
      float3 diffuse = calculate_diffuse(material, light, diffuse_intensity);

      output_image[pixel_index] = add_float3(material.ambient, diffuse);
    } else {
      output_image[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
    }
  }
}
