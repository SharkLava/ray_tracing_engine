#include <vector_types.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <stdio.h>
#include "ray.h"
#include "triangle.h"
#include "material.h"
#include "intersection.h"
#include "cuda_utils.h"
#include "ray_generation.h"
#include "shading.h"
#include "light.h"
#include "cuda_math.h"

void saveImagePPM(const char* filename, float3* image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write PPM header
    file << "P6\n" << width << " " << height << "\n255\n";

    // Convert floating-point colors to bytes and write to file
    for (int i = 0; i < width * height; i++) {
        // Clamp colors to [0,1] range
        float r = std::min(std::max(image[i].x, 0.0f), 1.0f);
        float g = std::min(std::max(image[i].y, 0.0f), 1.0f);
        float b = std::min(std::max(image[i].z, 0.0f), 1.0f);

        // Convert to bytes
        unsigned char pixel[3] = {
            (unsigned char)(r * 255.0f),
            (unsigned char)(g * 255.0f),
            (unsigned char)(b * 255.0f)
        };

        file.write((char*)pixel, 3);
    }

    file.close();
    std::cout << "Image saved to " << filename << std::endl;
}

int main(int argc, char** argv) {
    // Set up image dimensions
    const int width = 800;
    const int height = 600;
    
    // Camera setup - moved closer to see the triangle better
    float3 camera_pos = make_float3(0.0f, 0.0f, -2.0f);
    float3 camera_dir = make_float3(0.0f, 0.0f, 1.0f);
    float fov = 90.0f;  // Wider FOV to see more

    // Create a simple triangle
    Triangle triangle;
    triangle.v0 = make_float3(-0.5f, -0.5f, 0.0f);  // Moved triangle closer to camera
    triangle.v1 = make_float3(0.5f, -0.5f, 0.0f);
    triangle.v2 = make_float3(0.0f, 0.5f, 0.0f);

    // Create material with brighter colors
    Material material;
    material.ambient = make_float3(0.2f, 0.2f, 0.2f);  // Increased ambient
    material.diffuse = make_float3(0.8f, 0.3f, 0.3f);  // Brighter red
    material.specular = make_float3(1.0f, 1.0f, 1.0f);
    material.shininess = 32.0f;

    // Create a light - moved to better illuminate the triangle
    Light light;
    light.position = make_float3(2.0f, 2.0f, -2.0f);
    light.color = make_float3(1.0f, 1.0f, 1.0f);

    std::cout << "Scene setup:" << std::endl;
    std::cout << "Triangle vertices: (" 
              << triangle.v0.x << "," << triangle.v0.y << "," << triangle.v0.z << "), ("
              << triangle.v1.x << "," << triangle.v1.y << "," << triangle.v1.z << "), ("
              << triangle.v2.x << "," << triangle.v2.y << "," << triangle.v2.z << ")" << std::endl;
    std::cout << "Camera position: (" 
              << camera_pos.x << "," << camera_pos.y << "," << camera_pos.z << ")" << std::endl;
    std::cout << "Light position: (" 
              << light.position.x << "," << light.position.y << "," << light.position.z << ")" << std::endl;

    // Allocate device memory
    Ray* d_rays;
    Triangle* d_triangles;
    Material* d_materials;
    Intersection* d_intersections;
    float3* d_output_image;

    CUDA_CHECK(cudaMalloc(&d_rays, width * height * sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(&d_triangles, sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_materials, sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d_intersections, width * height * sizeof(Intersection)));
    CUDA_CHECK(cudaMalloc(&d_output_image, width * height * sizeof(float3)));

    // Initialize output image to a debug color to verify memory is working
    float3 debug_color = make_float3(0.0f, 0.0f, 0.0f);
    CUDA_CHECK(cudaMemset(d_output_image, 0, width * height * sizeof(float3)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_triangles, &triangle, sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, &material, sizeof(Material), cudaMemcpyHostToDevice));

    // Set up grid and block dimensions
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);

    // Generate rays
    generate_rays<<<grid_size, block_size>>>(camera_pos, camera_dir, fov, width, height, d_rays);
    CUDA_CHECK(cudaGetLastError());

    // Debug: Copy back some rays to verify they're generated correctly
    Ray* h_debug_rays = new Ray[10];
    CUDA_CHECK(cudaMemcpy(h_debug_rays, d_rays, 10 * sizeof(Ray), cudaMemcpyDeviceToHost));
    std::cout << "\nFirst few rays:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "Ray " << i << " origin: (" 
                  << h_debug_rays[i].origin.x << "," 
                  << h_debug_rays[i].origin.y << "," 
                  << h_debug_rays[i].origin.z << ")" << std::endl;
        std::cout << "Ray " << i << " direction: (" 
                  << h_debug_rays[i].direction.x << "," 
                  << h_debug_rays[i].direction.y << "," 
                  << h_debug_rays[i].direction.z << ")" << std::endl;
    }
    delete[] h_debug_rays;

    // Perform ray-triangle intersection
    intersect_triangles<<<grid_size, block_size>>>(d_rays, d_triangles, 1, width, height, d_intersections);
    CUDA_CHECK(cudaGetLastError());

    // Debug: Check some intersections
    Intersection* h_debug_intersections = new Intersection[10];
    CUDA_CHECK(cudaMemcpy(h_debug_intersections, d_intersections, 10 * sizeof(Intersection), cudaMemcpyDeviceToHost));
    std::cout << "\nFirst few intersections:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "Intersection " << i << ": t=" << h_debug_intersections[i].t 
                  << ", triangle=" << h_debug_intersections[i].triangle_index << std::endl;
    }
    delete[] h_debug_intersections;

    // Shade pixels
    shade_pixels<<<grid_size, block_size>>>(d_rays, d_intersections, d_triangles, d_materials, 
                                          light, width, height, d_output_image);
    CUDA_CHECK(cudaGetLastError());

    // Allocate host memory for the output image
    float3* h_output_image = new float3[width * height];

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_image, d_output_image, width * height * sizeof(float3), 
                         cudaMemcpyDeviceToHost));

    // Print some pixel colors before saving
    std::cout << "\nFirst few pixels colors:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Pixel " << i << ": (" 
                  << h_output_image[i].x << ", "
                  << h_output_image[i].y << ", "
                  << h_output_image[i].z << ")" << std::endl;
    }

    // Save the image
    saveImagePPM("output.ppm", h_output_image, width, height);

    // Cleanup
    delete[] h_output_image;
    CUDA_CHECK(cudaFree(d_rays));
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_materials));
    CUDA_CHECK(cudaFree(d_intersections));
    CUDA_CHECK(cudaFree(d_output_image));

    return 0;
}
