ANDROID_API :=31
CLANG_PATH := $(CLANG_PATH)

# set onnxruntime path
ONNXRUNTIME_DIR := $(ONNXRUNTIME_DIR)
ONNXRUNTIME_LIB_DIR := ${ONNXRUNTIME_DIR}/build/Android/Release
ONNXRUNTIME_INCLUDE_DIR := ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session


# Compiler settings - Can change to clang if desired
CXX := ${CLANG_PATH}/riscv64-linux-android${ANDROID_API}-clang++
# Include directories for header files
INCLUDES := -I. -I${ONNXRUNTIME_INCLUDE_DIR}
# Compiler flags, e.g. -g for debug, -O2 for optimise and -Wall for extra warnings.
CXXFLAGS := -g -O2 -Wall $(INCLUDES)
# Linker flags, e.g. -L<dir> if you have libraries in a nonstandard directory <dir>
LDFLAGS := -L${ONNXRUNTIME_LIB_DIR}
# Libraries to link into executable:
LDLIBS := -lonnxruntime

# The build target executable:
TARGET := run_resnet50

# Object files from the source files
OBJS := resnet50.o ../utils.o

.PHONY: all clean

# The default target - build the TARGET
all: $(TARGET) cosine_similarity

$(TARGET): $(OBJS)
	$(CXX)  -rdynamic $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)
	rm -rf $(OBJS)

cosine_similarity:
	$(CXX) -o cosine_similarity ../cosine_similarity.cpp


# To obtain object files
%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(TARGET) $(OBJS) cosine_similarity
