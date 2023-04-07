// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "shl_common.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/optimizer/initializer.h"

extern "C" {
#include "csi_nn.h"
}

#ifdef RISCV_C920
#define SHL_TARGET CSINN_C920
#endif

#ifdef RISCV_C906
#define SHL_TARGET CSINN_C906
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
      : session_(session),
        shl_ops_map({
            {"Add", &OnnxToShlConverter::Add},
            {"AveragePool", &OnnxToShlConverter::AveragePool},
            {"Concat", &OnnxToShlConverter::Concat},
            {"Conv", &OnnxToShlConverter::Conv},
            {"Flatten", &OnnxToShlConverter::Flatten},
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
            {"Split", &OnnxToShlConverter::Split},
            {"Sub", &OnnxToShlConverter::Sub},
            {"Transpose", &OnnxToShlConverter::Transpose},
            {"Unsqueeze", &OnnxToShlConverter::Reshape},
            {"Clip", &OnnxToShlConverter::Clip},
            {"MatMul", &OnnxToShlConverter::MatMul},
            {"LeakyRelu", &OnnxToShlConverter::LeakyRelu},
        }) {
    session_->debug_level = CSINN_DEBUG_LEVEL_ERROR;
    session_->profiler_level = CSINN_PROFILER_LEVEL_UNSET;

    if (config.count("debug_level")) {
      session_->debug_level = GetShlDebugLevelEnum(config.at("debug_level"));
    }
    if (config.count("profiler_level")) {
      session_->profiler_level = GetShlProfilerLevelEnum(config.at("profiler_level"));
    }

    session_->base_run_mode = CSINN_RM_CPU_GRAPH;
    session_->model.save_mode = CSINN_RUN_ONLY;
    session_->base_api = SHL_TARGET;

#ifndef __x86_64__
    if (session_->base_api == CSINN_REF) {
      std::cerr << "Currently using x86 reference implementation on non-x86 platform." << std::endl;
    }
#endif
    session_->base_dtype = CSINN_DTYPE_FLOAT32;
    csinn_session_init(session_);
  }

  ~OnnxToShlConverter() {}

  void Convert(const GraphViewer& graph_view);

 private:
  csinn_session* session_;
  std::map<std::string, csinn_tensor*> shl_tensor_map;
  using OperationFunc = void (OnnxToShlConverter::*)(const onnxruntime::Node& node);
  std::map<std::string, OperationFunc> shl_ops_map;

  void InitAllTensor(const GraphViewer& graph_view);

  void Add(const onnxruntime::Node& node);
  void AveragePool(const onnxruntime::Node& node);
  void AvgPool2d(const onnxruntime::Node& node);
  void Concat(const onnxruntime::Node& node);
  void Conv(const onnxruntime::Node& node);
  void Conv1D(const onnxruntime::Node& node){};
  void Conv2D(const onnxruntime::Node& node);
  void Conv3D(const onnxruntime::Node& node){};
  void Flatten(const onnxruntime::Node& node);
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

  void SetPoolAttr(csinn_pool_params* params, const onnxruntime::Node& node);
  void NodeConvert(const onnxruntime::Node& node) {
    const auto& op_type = node.OpType();
    if (!shl_ops_map.count(op_type)) {
      throw std::invalid_argument("Unsupported operator " + op_type);
    }
    (this->*(shl_ops_map.at(op_type)))(node);
  }
};

}  // namespace shl_ep
}  // namespace onnxruntime
