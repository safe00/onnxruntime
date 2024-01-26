// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <vector>
#include <iostream>
#include <fstream>

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

float compute_cs(float* a, float* b, uint32_t size) {
  double dot_sum = 0.0;
  double a_norm = 0.0;
  double b_norm = 0.0;
  float res = 0.0;

  for (size_t i = 0; i < size; i++) {
    dot_sum += (a[i] * b[i]);
    a_norm += (a[i] * a[i]);
    b_norm += (b[i] * b[i]);
  }
  res = dot_sum / (sqrt(a_norm * b_norm));
  return res;
}

void save_float_array(float* floatArray, size_t size, const char* filename) {
  FILE* file = fopen(filename, "wb");
  if (file == NULL) {
    return;
  }

  size_t elementsWritten = fwrite(floatArray, sizeof(float), size, file);
  if (elementsWritten != size) {
    return;
  }

  fclose(file);
}

std::vector<float> readFloatsFromBinaryFile(const std::string& filepath) {
  std::vector<float> buffer;
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filepath << std::endl;
    return buffer;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (size % sizeof(float) != 0) {
    std::cerr << "File size is not a multiple of float size: " << filepath << std::endl;
    return buffer;
  }

  buffer.resize(size / sizeof(float));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    std::cerr << "Failed to read float data from file: " << filepath << std::endl;
  }
  file.close();

  return buffer;
}

std::vector<std::vector<float>> test_model(const ORTCHAR_T* model_path, const char** input_names, float* input_data,
                                           const int64_t* input_shape, int use_shl, std::string test_env) {
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, test_env.c_str(), &env));
  assert(env != NULL);
  OrtSessionOptions* session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  if (use_shl) {
    ORT_ABORT_ON_ERROR(g_ort->OrtSessionOptionsAppendExecutionProvider_Shl(session_options, "base_api:CSINN_TH1520"));
  }

  OrtSession* session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

  OrtMemoryInfo* memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const size_t input_shape_len = 4;
  size_t model_input_ele_count = 1;

  for (size_t i = 0; i < input_shape_len; i++) {
    model_input_ele_count *= input_shape[i];
  }

  const size_t model_input_len = model_input_ele_count * sizeof(float);

  OrtValue* input_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, model_input_len, input_shape,
                                                           input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor));
  assert(input_tensor != NULL);
  int is_tensor;
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);

  size_t num_output_nodes;
  std::vector<const char*> output_node_names;
  std::vector<size_t> output_data_size;
  std::vector<std::vector<int64_t>> output_node_dims;
  std::vector<OrtValue*> output_tensors;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &num_output_nodes));
  output_node_names.resize(num_output_nodes);
  output_data_size.resize(num_output_nodes);
  output_node_dims.resize(num_output_nodes);
  output_tensors.resize(num_output_nodes);
  OrtAllocator* allocator;
  ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));
  for (size_t i = 0; i < num_output_nodes; i++) {
    // Get output node names
    char* output_name;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
    output_node_names[i] = output_name;

    OrtTypeInfo* typeinfo;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    // Get output shapes/dims
    size_t num_dims;
    ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(tensor_info, &num_dims));
    output_node_dims[i].resize(num_dims);
    ORT_ABORT_ON_ERROR(g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims[i].data(), num_dims));

    size_t tensor_size;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorShapeElementCount(tensor_info, &tensor_size));
    output_data_size[i] = tensor_size;

    if (typeinfo) g_ort->ReleaseTypeInfo(typeinfo);
  }

  ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), output_node_names.size(),
                                output_tensors.data()));

  std::vector<std::vector<float>> output;
  for (size_t i = 0; i < output_tensors.size(); i++) {
    auto output_tensor = output_tensors[i];
    assert(output_tensor != NULL);
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);
    float* output_tensor_data = NULL;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
    std::vector<float> output_data(output_tensor_data, output_tensor_data + output_data_size[i]);
    output.push_back(output_data);
    g_ort->ReleaseValue(output_tensor);
  }

  g_ort->ReleaseValue(input_tensor);
  free(input_data);

  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);

  return output;
}
