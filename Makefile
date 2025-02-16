# CUDA installation path
CUDA_PATH ?= /opt/cuda
CUDA_INC_DIR := $(CUDA_PATH)/include
CUDA_LIB_DIR := $(CUDA_PATH)/lib64

# Compiler and flags
NVCC := nvcc
NVCCFLAGS := -I$(CUDA_INC_DIR) -I. -L$(CUDA_LIB_DIR) -lcudart

# Output executable name
EXECUTABLE := main

# Source files
SOURCES := main.cu ray_generation.cu intersection.cu shading.cu cuda_math.cu cuda_utils.cu

# Object files
OBJECTS := $(SOURCES:.cu=.o)

# Default target
all: $(EXECUTABLE)

# Compile CUDA code
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Link the executable
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ $(NVCCFLAGS)

# Clean up object files and executable
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Run the executable
run: $(EXECUTABLE)
	./$(EXECUTABLE)

.PHONY: all clean run
