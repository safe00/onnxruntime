#pragma once

#include <assert.h>
#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <libgen.h>
#include <unistd.h>
#include <math.h>

#include "onnxruntime_c_api.h"

std::vector<float> readFloatsFromBinaryFile(const std::string& filepath);
void save_float_array(float *floatArray, size_t size, const char *filename);
float compute_cs(float *a, float *b, uint32_t size);
std::vector<std::vector<float>>test_model(const ORTCHAR_T *model_path, const char **input_names, float *input_data,
                  const int64_t *input_shape, int use_shl, std::string test_env);
