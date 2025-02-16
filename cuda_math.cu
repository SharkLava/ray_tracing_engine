#include "cuda_math.h"
#include <vector_types.h>

__device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross(float3 a, float3 b) {
  float3 result;
  result.x = a.y * b.z - a.z * b.y;
  result.y = a.z * b.x - a.x * b.z;
  result.z = a.x * b.y - a.y * b.x;
  return result;
}

__device__ float length(float3 a) { return sqrtf(dot(a, a)); }

__device__ float3 normalize(float3 a) {
  float len = length(a);
  return make_float3(a.x / len, a.y / len, a.z / len);
}

__device__ float3 multiply_float3(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 multiply_float3_scalar(float3 a, float scalar) {
  return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__device__ float3 add_float3(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
