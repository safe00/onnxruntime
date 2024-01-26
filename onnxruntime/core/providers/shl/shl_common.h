// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <limits>

#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/shared/utils/utils.h"

extern "C" {
#include "csinn/csi_nn.h"
#include "shl_utils.h"
#if defined(RISCV_TH1520) || defined(RISCV_C920)
#ifndef __x86_64__
#include "shl_public/shl_c920.h"
#endif
#endif
}

namespace onnxruntime {
namespace shl_ep {
template <typename T>
T* GetShlParams(csinn_session* sess, const Node& node) {
  // TODO:: use node to select base api
  T* ret = (T*)csinn_alloc_params(sizeof(T), sess);
  if (sess->base_api == CSINN_TH1520) {
    ret->base.api = CSINN_TH1520;
  }
  ret->base.layout = CSINN_LAYOUT_NCHW;
  return ret;
}

template <typename T>
T* GetShlParams(csinn_session* sess) {
  T* ret = (T*)csinn_alloc_params(sizeof(T), sess);
  ret->base.layout = CSINN_LAYOUT_NCHW;
  return ret;
}

template <typename T, typename... Args>
bool is_one_of(const T& value, const Args&... args) {
  return ((value == args) || ...);
}

template <typename T, typename... Args>
bool all_equal(const T& first, const Args&... args) {
  return ((first == args) && ...);
}

void UpdateShlTensorDim(csinn_tensor* tensor, std::vector<int64_t> shape);
csinn_dtype_enum GetShlDtypeEnum(const ONNX_NAMESPACE::TypeProto_Tensor type);
csinn_dtype_enum GetShlDtypeEnum(enum ONNXTensorElementDataType type);
csinn_api_enum GetShlAPIEnum(std::string type);
csinn_quant_enum GetShlQDtypeEnum(std::string type);
csinn_rmode_enum GetShlRunModeEnum(std::string type);
csinn_debug_enum GetShlDebugLevelEnum(std::string type);
csinn_profiler_enum GetShlProfilerLevelEnum(std::string type);
std::vector<std::vector<int>> GetSupportedNodes(const onnxruntime::GraphViewer& graph_viewer, csinn_profiler_enum profile_level,
                                                csinn_api_enum base_api);
csinn_tensor* CreateShlTensor(const NodeArg* onnx_tensor, csinn_session* sess);
csinn_layout_enum GetShlWeightLayoutEnum(int dim_count);
float float16_to_float32(int16_t value);
std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>>
MarkfusibleNodes(const onnxruntime::GraphViewer& graph_viewer);
std::unordered_map<const Node*, std::string>
GetAllFusionNode(std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>> marked_fusible_map);

}  // namespace shl_ep
}  // namespace onnxruntime
