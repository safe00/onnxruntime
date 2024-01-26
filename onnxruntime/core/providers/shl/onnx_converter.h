// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "shl_common.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/optimizer/initializer.h"

#ifdef RISCV_C920
#define SHL_TARGET CSINN_C920
#endif

#ifdef RISCV_C906
#define SHL_TARGET CSINN_C906
#endif

#ifdef RISCV_TH1520
#define SHL_TARGET CSINN_TH1520
#endif

#ifdef REF_X86
#define SHL_TARGET CSINN_REF
#endif

#ifndef SHL_TARGET
#define SHL_TARGET CSINN_REF
#endif

namespace onnxruntime {
namespace shl_ep {

/**
 *  For convert from onnx::GraphViewer to shl csinn_session.
 */

class OnnxToShlConverter {
 public:
  OnnxToShlConverter(csinn_session* session, const std::unordered_map<std::string, std::string> config)
      : session_(session) {
    session_->debug_level = CSINN_DEBUG_LEVEL_ERROR;
    session_->profiler_level = CSINN_PROFILER_LEVEL_UNSET;

    if (config.count("debug_level")) {
      session_->debug_level = GetShlDebugLevelEnum(config.at("debug_level"));
    }
    if (config.count("profiler_level")) {
      session_->profiler_level = GetShlProfilerLevelEnum(config.at("profiler_level"));
    }
    session_->base_run_mode = CSINN_RM_CPU_GRAPH;
    if (config.count("base_run_mode")) {
      session_->base_run_mode = GetShlRunModeEnum(config.at("base_run_mode"));
    }

    // session_->model.save_mode = CSINN_RUN_ONLY;
    session_->base_api = SHL_TARGET;
    if (config.count("base_api")) {
      session_->base_api = GetShlAPIEnum(config.at("base_api"));
    }
    if (session_->base_api == CSINN_TH1520) {
      session_->base_run_mode = CSINN_RM_NPU_GRAPH;
    }

#ifndef __x86_64__
    if (session_->base_api == CSINN_REF) {
      std::cerr << "Currently using x86 reference implementation on non-x86 platform." << std::endl;
    }
#endif
  }

  ~OnnxToShlConverter() {}

  void Convert(const GraphViewer& graph_view);

 private:
  csinn_session* session_;
  std::map<std::string, csinn_tensor*> shl_tensor_map;
  using OperationFunc = void (OnnxToShlConverter::*)(const onnxruntime::Node& node);
  using FuseFunc = void (OnnxToShlConverter::*)(std::unordered_map<std::string, const Node*>);
  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>> marked_fusible_map;
  std::unordered_map<const Node*, std::string> all_fusible_nodes;
  void InitAllTensor(const GraphViewer& graph_view);
  void InitSHLSession(const GraphViewer& graph_viewer);

  void Add(const onnxruntime::Node& node);
  void AveragePool(const onnxruntime::Node& node);
  void AvgPool2d(const onnxruntime::Node& node);
  void Concat(const onnxruntime::Node& node);
  void Conv(const onnxruntime::Node& node);
  void Conv1D(const onnxruntime::Node& node){};
  void Conv2D(const onnxruntime::Node& node);
  void Conv3D(const onnxruntime::Node& node){};
  void Flatten(const onnxruntime::Node& node);
  void Gather(const onnxruntime::Node& node);
  void Gemm(const onnxruntime::Node& node);
  void GlobalAvgPool(const onnxruntime::Node& node);
  void MaxPool(const onnxruntime::Node& node);
  void MaxPool2d(const onnxruntime::Node& node);
  void Mul(const onnxruntime::Node& node);
  void Pow(const onnxruntime::Node& node);
  void ReLU(const onnxruntime::Node& node);
  void Resize(const onnxruntime::Node& node);
  void Reshape(const onnxruntime::Node& node);
  void Sigmoid(const onnxruntime::Node& node);
  void Softmax(const onnxruntime::Node& node);
  void Split(const onnxruntime::Node& node);
  void Sub(const onnxruntime::Node& node);
  void Transpose(const onnxruntime::Node& node);
  void Clip(const onnxruntime::Node& node);
  void MatMul(const onnxruntime::Node& node);
  void LeakyRelu(const onnxruntime::Node& node);
  void Erf(const onnxruntime::Node& node);
  void Div(const onnxruntime::Node& node);
  void DequantizeLinear(const onnxruntime::Node& node){};
  void QuantizeLinear(const onnxruntime::Node& node){};

  void FuseLayerNormalization(std::unordered_map<std::string, const Node*> fuse_nodes);
  void FuseDQDNodes(std::unordered_map<std::string, const Node*> fuse_nodes);
  void FuseDQDTensor(const GraphViewer& graph_viewer);

  template <typename... Args>
  void CheckQuantDtype(const csinn_tensor* first, const Args*... args) {
    if (session_->base_api != CSINN_TH1520) {
      return;
    }

    bool result = ((first->dtype == args->dtype) && ...);
    if (!result) {
      throw std::invalid_argument("When use th1520 backend, all input dtype must be same as output dtype. model can be convert by HHB tools");
    }

    return;
  }
  void CheckQuantDtype(const csinn_tensor* first, csinn_tensor** inputs, int num) {
    if (session_->base_api != CSINN_TH1520) {
      return;
    }
    for (int i = 0; i < num; i++) {
      if (first->dtype != inputs[i]->dtype) {
        throw std::invalid_argument("When use th1520 backend, all input dtype must be same as output dtype. model can be convert by HHB tools");
      }
    }
    return;
  }

  void SetPoolAttr(csinn_pool_params* params, const onnxruntime::Node& node);
  csinn_quant_info* GetQuantInfo(const Node* node);
  void ProcessQNodes(const Node* node);
  void ProcessDQNodes(const Node* node);
  void ProcessDQDNodes(const Node* q_node, const Node* dq_node);

  static inline std::unordered_map<std::string, OperationFunc> shl_ops_map = {
      {"Add", &OnnxToShlConverter::Add},
      {"AveragePool", &OnnxToShlConverter::AveragePool},
      {"Concat", &OnnxToShlConverter::Concat},
      {"Conv", &OnnxToShlConverter::Conv},
      {"Flatten", &OnnxToShlConverter::Flatten},
      {"Gather", &OnnxToShlConverter::Gather},
      {"Gemm", &OnnxToShlConverter::Gemm},
      {"GlobalAveragePool", &OnnxToShlConverter::GlobalAvgPool},
      {"MaxPool", &OnnxToShlConverter::MaxPool},
      {"Mul", &OnnxToShlConverter::Mul},
      {"Pow", &OnnxToShlConverter::Pow},
      {"Relu", &OnnxToShlConverter::ReLU},
      {"Reshape", &OnnxToShlConverter::Reshape},
      {"Resize", &OnnxToShlConverter::Resize},
      {"Sigmoid", &OnnxToShlConverter::Sigmoid},
      {"Softmax", &OnnxToShlConverter::Softmax},
      {"Erf", &OnnxToShlConverter::Erf},
      {"Split", &OnnxToShlConverter::Split},
      {"Sub", &OnnxToShlConverter::Sub},
      {"Div", &OnnxToShlConverter::Div},
      {"Transpose", &OnnxToShlConverter::Transpose},
      // {"Unsqueeze", &OnnxToShlConverter::Unsqueeze},
      {"Clip", &OnnxToShlConverter::Clip},
      {"MatMul", &OnnxToShlConverter::MatMul},
      {"LeakyRelu", &OnnxToShlConverter::LeakyRelu},
      {"DequantizeLinear", &OnnxToShlConverter::DequantizeLinear},
      {"QuantizeLinear", &OnnxToShlConverter::QuantizeLinear},
  };
  static inline std::unordered_map<std::string, FuseFunc> ops_fuse_map = {
      {"LayerNorm", &OnnxToShlConverter::FuseLayerNormalization},
      {"QDQFusion", &OnnxToShlConverter::FuseDQDNodes}};

  std::pair<FuseFunc, std::unordered_map<std::string, const Node*>> GetFusionFunc(const onnxruntime::Node& node, bool& clear) {
    static std::unordered_map<const Node*, std::unordered_map<std::string, const Node*>> all_fusible_func;
    if (clear) {
      all_fusible_func.clear();
    }
    if (all_fusible_func.empty()) {
      clear = false;
      for (const auto& iter : marked_fusible_map) {
        for (const auto& node_map : iter.second) {
          for (const auto& iter2 : node_map) {
            if (iter2.first == "key_node") {
              all_fusible_func.emplace(iter2.second, node_map);
              continue;
            }
          }
        }
      }
    }

    std::unordered_map<std::string, const Node*> node_map;
    auto iter = all_fusible_func.find(&node);
    if (iter != all_fusible_func.end()) {
      auto func = ops_fuse_map[all_fusible_nodes[&node]];
      return {func, iter->second};
    }

    return {static_cast<FuseFunc>(nullptr), std::unordered_map<std::string, const Node*>()};
  }

  void NodeConvert(const onnxruntime::Node& node, bool& clear_buffer) {
    if (all_fusible_nodes.count(&node)) {
      std::string is_key_node = all_fusible_nodes.at(&node);
      if (is_key_node != "") {
        FuseFunc func;
        std::unordered_map<std::string, const Node*> fusible_nodes;
        std::tie(func, fusible_nodes) = GetFusionFunc(node, clear_buffer);
        if (func) {
          (this->*(func))(fusible_nodes);
        }
      }
      return;
    }

    const auto& op_type = node.OpType();
    if (!shl_ops_map.count(op_type)) {
      throw std::invalid_argument("Unsupported operator " + op_type);
    }
    (this->*(shl_ops_map.at(op_type)))(node);
  }
};

}  // namespace shl_ep
}  // namespace onnxruntime
