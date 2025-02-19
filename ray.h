#ifndef RAY_H
#define RAY_H

#include "cuda_math.h"
#include <vector_types.h>

struct alignas(16) Ray {
  float3 origin;
  float3 direction;
};

#endif
