#ifndef SPHERE_H
#define SPHERE_H

#include <vector_types.h>

struct alignas(16) Sphere {
  float3 center;
  float radius;
};

#endif
