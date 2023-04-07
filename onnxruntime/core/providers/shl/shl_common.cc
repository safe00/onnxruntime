// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shl_common.h"

namespace onnxruntime {
namespace shl_ep {
csinn_dtype_enum GetShlDtypeEnum(const ONNX_NAMESPACE::TypeProto_Tensor type) {
  if (type.has_elem_type()) {
    auto type_ = type.elem_type();
    switch (type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        return CSINN_DTYPE_FLOAT32;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        return CSINN_DTYPE_FLOAT16;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        return CSINN_DTYPE_UINT8;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        return CSINN_DTYPE_INT8;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        return CSINN_DTYPE_INT32;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        return CSINN_DTYPE_INT64;
      default:
        // TODO: support other type
        throw std::invalid_argument("The input of graph doesn't have valid type");
    }
  }
  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_DTYPE_FLOAT32;
}

csinn_layout_enum GetShlActLayoutEnum(int dim_count) {
  switch (dim_count) {
    case 1:
      return CSINN_LAYOUT_N;
    case 2:
      return CSINN_LAYOUT_NC;
    case 3:
      return CSINN_LAYOUT_NCW;
    case 4:
      return CSINN_LAYOUT_NCHW;
    case 5:
      return CSINN_LAYOUT_NCDHW;
    default:
      return CSINN_LAYOUT_NCHW;
  }
  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_LAYOUT_NCHW;
}

csinn_layout_enum GetShlWeightLayoutEnum(int dim_count) {
  switch (dim_count) {
    case 1:
      return CSINN_LAYOUT_O;
    case 2:
      return CSINN_LAYOUT_OI;
    case 3:
      return CSINN_LAYOUT_OIW;
    case 4:
      return CSINN_LAYOUT_OIHW;
    case 5:
      return CSINN_LAYOUT_OIDHW;
    default:
      return CSINN_LAYOUT_OIHW;
  }
  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_LAYOUT_OIHW;
}

csinn_quant_enum GetShlQDtypeEnum(std::string type) {
  if (type == "CSINN_QUANT_FLOAT32")
    return CSINN_QUANT_FLOAT32;
  else if (type == "CSINN_QUANT_FLOAT16")
    return CSINN_QUANT_FLOAT16;

  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_QUANT_FLOAT32;
}

csinn_debug_enum GetShlDebugLevelEnum(std::string type) {
  if (type == "CSINN_DEBUG_LEVEL_DEBUG")
    return CSINN_DEBUG_LEVEL_DEBUG;
  else if (type == "CSINN_DEBUG_LEVEL_INFO")
    return CSINN_DEBUG_LEVEL_INFO;
  else if (type == "CSINN_DEBUG_LEVEL_WARNING")
    return CSINN_DEBUG_LEVEL_WARNING;
  else if (type == "CSINN_DEBUG_LEVEL_ERROR")
    return CSINN_DEBUG_LEVEL_ERROR;
  else if (type == "CSINN_DEBUG_LEVEL_FATAL")
    return CSINN_DEBUG_LEVEL_FATAL;

  throw std::invalid_argument("Shl debug level get error.");
  return CSINN_DEBUG_LEVEL_INFO;
}

csinn_profiler_enum GetShlProfilerLevelEnum(std::string type) {
  if (type == "CSINN_PROFILER_LEVEL_UNSET")
    return CSINN_PROFILER_LEVEL_UNSET;
  else if (type == "CSINN_PROFILER_LEVEL_TIMER")
    return CSINN_PROFILER_LEVEL_TIMER;
  else if (type == "CSINN_PROFILER_LEVEL_DUMP")
    return CSINN_PROFILER_LEVEL_DUMP;
  else if (type == "CSINN_PROFILER_LEVEL_ALL")
    return CSINN_PROFILER_LEVEL_ALL;

  throw std::invalid_argument("Shl prifiler level get error.");
  return CSINN_PROFILER_LEVEL_UNSET;
}

std::pair<bool, std::string> IsNodeSupported(const GraphViewer& graph_viewer, const Node* node) {
  const auto& op = node->OpType();

  const std::vector<std::string> supported_types{
      "Add", "AveragePool", "Clip", "Concat",
      "Conv", "Flatten", "Gemm", "GlobalAveragePool",
      "MaxPool", "Mul", "Relu", "Sigmoid",
      "Split", "Transpose", "Reshape",
      "Unsqueeze", "MatMul", "LeakyRelu",
      // "Softmax", "Sub", "Pow", "Resize",
      // "LeakyRelu", "Slice", "Squeeze", "Gather",
      /*"QLinearConv", "QuantizeLinear", "DequantizeLinear"*/};
  if (std::find(supported_types.begin(), supported_types.end(), op) ==
      supported_types.end()) {
    return {false, "Unsupported operator"};
  }

  NodeAttrHelper helper(*node);

  if (op == "Conv") {
    const NodeArg* input = node->InputDefs()[0];
    auto in_shape = input->Shape();
    if (!is_one_of<int64_t>(in_shape->dim_size(), 4)) {
      return {false, "conv only supporte conv2d now"};
    }
  } else if (op == "AveragePool" || op == "MaxPool") {
    const NodeArg* input = node->InputDefs()[0];
    auto in_shape = input->Shape();
    if (!is_one_of<int64_t>(in_shape->dim_size(), 4)) {
      return {false, "conv only supporte pool2d now"};
    }
    if (helper.Get("auto_pad", "NOTSET") != "NOTSET") {
      return {false, "auto_pad is not supported"};
    }
    if (node->InputDefs().size() != 1) {
      return {false, "Argmax in maxpooling is not supported"};
    }
  } else if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
    const NodeArg* input = node->InputDefs()[0];
    auto in_shape = input->Shape();
    if (!is_one_of<int64_t>(in_shape->dim_size(), 4)) {
      return {false, "Only rank-4 tensor is supported"};
    }
  } else if (op == "Gemm") {
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);
    if (!(transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f)) {
      return {false,
              "Only transA == 0, transB == 1, alpha == 1.0 and beta == "
              "1.0 is supported."};
    }
  } else if (op == "Resize") {
    const NodeArg* input = node->InputDefs()[0];
    if (node->InputDefs().size() >= 2) {
      const NodeArg* roi = node->InputDefs()[1];
      auto roi_shape = roi->Shape();
      if (roi_shape != nullptr) {
        return {false, "Resize not supported roi"};
      }
    }

    auto in_shape = input->Shape();
    if (in_shape->dim_size() != 4) {
      return {false, "Resize only supports 4 dims."};
    }

    const auto mode = helper.Get("mode", "nearest");
    const auto nearest_mode = helper.Get("nearest_mode", "floor");
    const auto cubic_coeff_a = helper.Get("cubic_coeff_a", -0.75f);
    if (!is_one_of<std::string>(mode, "linear", "bilinear", "nearest")) {
      return {false, "Resize only supported input mode linear, bilinear and nearest"};
    }
    if (!(cubic_coeff_a == -0.75)) {
      return {false, "Resize only supported cubic_coeff_a == 0.75"};
    }
    if (!(nearest_mode == "floor")) {
      return {false, "Resize only supported nearest_mode == floor"};
    }

  } else if (op == "Reshape") {
    const NodeArg* shape = node->InputDefs()[1];
    bool is_const = graph_viewer.IsInitializedTensor(shape->Name());
    if (!is_const) {
      return {false, "Reshape only supported constant shape"};
    }
  } else if (op == "Split") {
    if (node->InputDefs().size() >= 2) {
      const NodeArg* split = node->InputDefs()[1];
      bool is_const = graph_viewer.IsInitializedTensor(split->Name());
      if (!is_const) {
        return {false, "Reshape only supported constant shape"};
      }
    }
  } else if (op == "Unsqueeze") {
    if (node->InputDefs().size() == 2) {
      const NodeArg* axis = node->InputDefs()[1];
      bool is_const = graph_viewer.IsInitializedTensor(axis->Name());
      if (!is_const) {
        return {false, "Unsqueeze only supported constant axis"};
      }
    }
  } else if (op == "Clip") {
    bool is_support = true;
    auto in_dtype = node->InputDefs()[0]->TypeAsProto()->tensor_type();
    if (in_dtype.has_elem_type()) {
      auto in_dtype_elem = in_dtype.elem_type();
      if (!(in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
            in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)) {
        return {false, "Clip only supported FLOAT32 and FLOAT16"};
      }
    } else {
      return {false, "Clip only supported FLOAT32 and FLOAT16"};
    }

    if (node->InputDefs().size() == 2) {
      const NodeArg* min_arg = node->InputDefs()[1];
      is_support &= graph_viewer.IsInitializedTensor(min_arg->Name());
    }
    if (node->InputDefs().size() == 3) {
      const NodeArg* max_arg = node->InputDefs()[2];
      is_support &= graph_viewer.IsInitializedTensor(max_arg->Name());
    }
    if (!is_support) {
      return {false, "Clip only supported constant min max value"};
    }
  }

  return {true, ""};
}

float float16_to_float32(int16_t value) {
  union FP32 {
    uint32_t u;
    float f;
  };

  const union FP32 magic = {(254UL - 15UL) << 23};
  const union FP32 was_inf_nan = {(127UL + 16UL) << 23};
  union FP32 out;

  out.u = (value & 0x7FFFU) << 13;
  out.f *= magic.f;
  if (out.f >= was_inf_nan.f) {
    out.u |= 255UL << 23;
  }
  out.u |= (value & 0x8000UL) << 16;

  return out.f;
}

std::vector<std::vector<int>> GetSupportedNodes(const onnxruntime::GraphViewer& graph_viewer) {
  std::vector<std::vector<int>> supported_node_vecs;
  std::vector<int> supported_node_vec;
  const std::vector<NodeIndex>& node_index = graph_viewer.GetNodesInTopologicalOrder();
  for (auto i : node_index) {
    bool supported;
    std::string error_msg;
    std::tie(supported, error_msg) = IsNodeSupported(graph_viewer, graph_viewer.GetNode(i));
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      const auto& op = graph_viewer.GetNode(i)->OpType();
      LOGS_DEFAULT(INFO) << op << ": " << error_msg;
      if (!supported_node_vec.empty()) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }
  if (!supported_node_vec.empty()) {
    supported_node_vecs.push_back(supported_node_vec);
  }

  return supported_node_vecs;
}

void UpdateShlTensorDim(csinn_tensor* tensor, std::vector<int64_t> shape) {
  tensor->dim_count = shape.size();
  std::transform(shape.begin(), shape.end(), tensor->dim, [](int64_t val) -> int32_t {
    return static_cast<int32_t>(val);
  });
}

csinn_tensor* CreateShlTensor(const NodeArg* onnx_tensor, csinn_session* sess) {
  csinn_tensor* shl_tensor = csinn_alloc_tensor(sess);

  shl_tensor->name = const_cast<char*>(onnx_tensor->Name().c_str());
  shl_tensor->dtype = shl_ep::GetShlDtypeEnum(onnx_tensor->TypeAsProto()->tensor_type());

  auto onnx_shape = onnx_tensor->Shape();
  if (onnx_shape != NULL) {
    auto tensor_shape = utils::GetTensorShapeFromTensorShapeProto(*onnx_shape).AsShapeVector();
    if (tensor_shape.size() == 0) {
      shl_tensor->dim_count = 1;
      shl_tensor->dim[0] = 1;
    } else {
      shl_tensor->dim_count = tensor_shape.size();
      for (int32_t i = 0; i < shl_tensor->dim_count; i++) {
        shl_tensor->dim[i] = tensor_shape[i];
      }
    }
  } else {
    shl_tensor->dim_count = 0;
    shl_tensor->dim[0] = 0;
  }

  shl_tensor->layout = GetShlActLayoutEnum(shl_tensor->dim_count);

  return shl_tensor;
};

}  // namespace shl_ep
}  // namespace onnxruntime
