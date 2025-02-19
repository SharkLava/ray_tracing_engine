#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <vector_types.h>

__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator*(const float3 &a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3 &a, const float3 &b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float length(const float3 &a) {
  return sqrtf(dot(a, a));
}

__device__ __forceinline__ float3 normalize(const float3 &a) {
  float inv_len = rsqrtf(dot(a, a));
  return a * inv_len;
}

#endif
