#ifndef RAY_H
#define RAY_H

#include "cuda_math.h"
#include <vector_types.h>

typedef struct {
  float3 origin;
  float3 direction;
} Ray;

#endif