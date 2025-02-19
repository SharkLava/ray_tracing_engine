#include "cuda_math.h"
#include "shading.h"

__device__ __forceinline__ float3
calculate_diffuse(const Material &material, const Light &light,
                  const float diffuse_intensity) {
  return make_float3(material.diffuse.x * light.color.x * diffuse_intensity,
                     material.diffuse.y * light.color.y * diffuse_intensity,
                     material.diffuse.z * light.color.z * diffuse_intensity);
}

__global__ void shade_pixels(const Ray *__restrict__ rays,
                             const Intersection *__restrict__ intersections,
                             const Sphere *__restrict__ spheres,
                             const Material *__restrict__ materials,
                             const float3 plane_normal,
                             const Material plane_material, const Light light,
                             const int width, const int height,
                             float3 *__restrict__ output_image) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const int pixel_index = y * width + x;
  const Intersection &intersection = intersections[pixel_index];
  const Ray &ray = rays[pixel_index];

  // Early exit if no intersection
  if (intersection.t >= 1e38f) {
    output_image[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }

  // Calculate intersection point
  float3 intersection_point = ray.origin + ray.direction * intersection.t;
  float3 normal;
  const Material *material;

  if (intersection.sphere_index != -1) {
    const Sphere &sphere = spheres[intersection.sphere_index];
    material = &materials[intersection.sphere_index];
    normal = normalize(intersection_point - sphere.center);
  } else {
    material = &plane_material;
    normal = plane_normal;
  }

  // Calculate lighting
  float3 light_dir = normalize(light.position - intersection_point);
  float diffuse_intensity = fmaxf(dot(normal, light_dir), 0.0f);

  // Calculate final color
  float3 ambient = make_float3(material->ambient.x * light.color.x,
                               material->ambient.y * light.color.y,
                               material->ambient.z * light.color.z);

  float3 diffuse = calculate_diffuse(*material, light, diffuse_intensity);
  output_image[pixel_index] = ambient + diffuse;
}
