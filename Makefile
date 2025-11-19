# Directories
THREADS_DIR := src/pthreads
OPENMP_DIR := src/openmp
MPI_DIR := src/mpi
CUDA_DIR := src/cuda
BUILD_DIR := build

# Source files
THREADS_SRC := $(wildcard $(THREADS_DIR)/*.c)
OPENMP_SRC := $(wildcard $(OPENMP_DIR)/*.c)
MPI_SRC := $(wildcard $(MPI_DIR)/*.c)
CUDA_SRC := $(wildcard $(CUDA_DIR)/*.cu)

# Output binaries
THREADS_BIN := $(patsubst $(THREADS_DIR)/%.c,$(BUILD_DIR)/%.exe,$(THREADS_SRC))
OPENMP_BIN := $(patsubst $(OPENMP_DIR)/%.c,$(BUILD_DIR)/%.exe,$(OPENMP_SRC))
MPI_BIN := $(patsubst $(MPI_DIR)/%.c,$(BUILD_DIR)/%.exe,$(MPI_SRC))
CUDA_BIN := $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/%.exe,$(CUDA_SRC))

# Compiler and flags
CC := gcc
THREADS_FLAGS := -pthread -Wall
OPENMP_FLAGS := -fopenmp -Wall
MPI_CC := mpicc
MPI_FLAGS := -Wall
NVCC := nvcc
CUDA_FLAGS := -O2

all: $(THREADS_BIN) $(OPENMP_BIN) $(MPI_BIN) $(CUDA_BIN)

$(BUILD_DIR)/%.exe: $(THREADS_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(THREADS_FLAGS) $< -o $@

$(BUILD_DIR)/%.exe: $(OPENMP_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(OPENMP_FLAGS) $< -o $@

$(BUILD_DIR)/%.exe: $(MPI_DIR)/%.c | $(BUILD_DIR)
	$(MPI_CC) $(MPI_FLAGS) $< -o $@

$(BUILD_DIR)/%.exe: $(CUDA_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean