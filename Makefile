CUDA_PATH ?= /opt/cuda
CUDA_INC_DIR := $(CUDA_PATH)/include
CUDA_LIB_DIR := $(CUDA_PATH)/lib64

# Compiler and flags
NVCC := nvcc
NVCCFLAGS := -I$(CUDA_INC_DIR) -I. -L$(CUDA_LIB_DIR) -lcudart
GENCODE_FLAGS := -gencode arch=compute_50,code=sm_50 \
                 -gencode arch=compute_52,code=sm_52 \
                 -gencode arch=compute_60,code=sm_60 \
                 -gencode arch=compute_61,code=sm_61 \
                 -gencode arch=compute_70,code=sm_70 \
                 -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86

# Debug flags (uncomment for debugging)
# NVCCFLAGS += -g -G

# Optimization flags
NVCCFLAGS += -O3 $(GENCODE_FLAGS)

# Output executable name
EXECUTABLE := raytracer

# Source files
SOURCES := main.cu ray_generation.cu intersection.cu shading.cu

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
