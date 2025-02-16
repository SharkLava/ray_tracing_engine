#ifndef CUDA_MATH_H
#define CUDA_MATH_H

__device__ float dot(float3 a, float3 b);
__device__ float3 cross(float3 a, float3 b);
__device__ float length(float3 a);
__device__ float3 normalize(float3 a);

__device__ float3 multiply_float3(float3 a, float3 b);
__device__ float3 multiply_float3_scalar(float3 a, float scalar);
__device__ float3 add_float3(float3 a, float3 b);

#endif