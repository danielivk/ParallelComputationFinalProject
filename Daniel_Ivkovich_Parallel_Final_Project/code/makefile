###########################################################
# MPI+OMP+CUDA, Daniel Ivkovich
## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR= /usr/local/cuda

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC= mpicxx
CC_FLAGS= -fopenmp -Wall -I$(INC_DIR) -I$(CUDA_ROOT_DIR)/include
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -I$(INC_DIR) -I$(CUDA_ROOT_DIR)/include
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Make variables ##

# Target executable name:
EXE = $(OBJ_DIR)/executable

# Object files:
SRCS := $(shell find $(SRC_DIR) -name \*.c -or -name \*.cu -type f | rev | cut -d '/' -f1 |rev | cut -d '.' -f1)
OBJS := $(SRCS:%=$(OBJ_DIR)/%.o)
##########################################################

## Compile ##

# Link c and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .c file to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile main .cu file to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)
	
# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)

# Execute	
run:
	mpiexec -np 1 ./$(EXE)

run2:
	mpiexec -np 2 ./$(EXE)

run3:
	mpiexec -np 3 ./$(EXE)

run4:
	mpiexec -np 4 ./$(EXE)

