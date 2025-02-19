#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <fstream>
#include <iostream>
#include <vector_types.h>

inline void saveImagePPM(const char *filename, float3 *image, int width,
                         int height) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write PPM header
  file << "P6\n" << width << " " << height << "\n255\n";

  // Convert floating-point colors to bytes and write to file
  for (int i = 0; i < width * height; i++) {
    unsigned char pixel[3] = {
        static_cast<unsigned char>(std::min(std::max(image[i].x, 0.0f), 1.0f) *
                                   255.0f),
        static_cast<unsigned char>(std::min(std::max(image[i].y, 0.0f), 1.0f) *
                                   255.0f),
        static_cast<unsigned char>(std::min(std::max(image[i].z, 0.0f), 1.0f) *
                                   255.0f)};
    file.write(reinterpret_cast<char *>(pixel), 3);
  }

  file.close();
  std::cout << "Image saved to " << filename << std::endl;
}

#endif
