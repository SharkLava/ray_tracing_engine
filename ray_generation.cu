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
