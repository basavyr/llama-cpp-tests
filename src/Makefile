# llama.cpp configs
# the build with static libraries is required
ROOT_DIR = /Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp
BUILD_DIR = $(ROOT_DIR)/build

# 1. Header Includes
# headers from include/, common/, and ggml/include/
INCLUDES = -I$(ROOT_DIR)/include \
           -I$(ROOT_DIR)/common \
           -I$(ROOT_DIR)/ggml/include \
           -I$(BUILD_DIR)/common

# 2. Library Paths (-L)
# point to where CMake built the static .a files
LIB_PATHS = -L$(BUILD_DIR)/src \
            -L$(BUILD_DIR)/common \
            -L$(BUILD_DIR)/ggml/src \
            -L$(BUILD_DIR)/ggml/src/ggml-metal \
            -L$(BUILD_DIR)/ggml/src/ggml-blas

# 3. Linked Libraries (-l)
# link all the split ggml components found in your build
LIBS = -lcommon \
       -lllama \
       -lggml \
       -lggml-base \
       -lggml-cpu \
       -lggml-metal \
       -lggml-blas

# 4. macOS Frameworks (Required for Metal/GPU)
FRAMEWORKS = -framework Accelerate \
             -framework Foundation \
             -framework Metal \
             -framework MetalKit

CXX = g++
CXX_FLAGS = -std=c++17 -O3 -Wall -Wextra

TARGET = simple-test

SOURCES = simple-test.cpp

# $(TARGET): $(SOURCES)
# 	$(CXX) $(CXX_FLAGS) -o $(TARGET).o $(SOURCES)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_PATHS) $(SOURCES) $(LIBS) $(FRAMEWORKS) -o $(TARGET)


clean:
	rm -f *.o *.out $(TARGET)