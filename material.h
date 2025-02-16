#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector_types.h>

typedef struct {
  float3 ambient;
  float3 diffuse;
  float3 specular;
  float shininess;
} Material;

#endif