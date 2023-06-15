// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shl_common.h"
#include "op_fusion.h"

namespace onnxruntime {
namespace shl_ep {
using FuseMarkerFn = std::function<std::vector<std::unordered_map<std::string, const Node*>>(const onnxruntime::GraphViewer& graph_viewer)>;

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

csinn_dtype_enum GetShlDtypeEnum(enum ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return CSINN_DTYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return CSINN_DTYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return CSINN_DTYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return CSINN_DTYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return CSINN_DTYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return CSINN_DTYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return CSINN_DTYPE_BOOL;
    default:
      // TODO: support other type
      throw std::invalid_argument("The input of graph doesn't have valid type");
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

csinn_rmode_enum GetShlRunModeEnum(std::string type) {
  if (type == "CSINN_RM_LAYER")
    return CSINN_RM_LAYER;
  else if (type == "CSINN_RM_CPU_GRAPH")
    return CSINN_RM_CPU_GRAPH;
  else if (type == "CSINN_RM_NPU_GRAPH")
    return CSINN_RM_NPU_GRAPH;
  else if (type == "CSINN_RM_CPU_BASE_HYBRID")
    return CSINN_RM_CPU_BASE_HYBRID;

  throw std::invalid_argument("Shl run mode get error.");
  return CSINN_RM_CPU_GRAPH;
}

csinn_api_enum GetShlAPIEnum(std::string type) {
  if (type == "CSINN_REF")
    return CSINN_REF;
  else if (type == "CSINN_GREF")
    return CSINN_GREF;
  else if (type == "CSINN_C906")
    return CSINN_C906;
  else if (type == "CSINN_C920")
    return CSINN_C920;
  else if (type == "CSINN_TH1520")
    return CSINN_TH1520;
  else if (type == "CSINN_C908")
    return CSINN_C908;
  else if (type == "CSINN_RVV")
    return CSINN_RVV;

  throw std::invalid_argument("Shl run mode get error.");
  return CSINN_C920;
}

void get_clip_value(const GraphViewer& graph_viewer, const onnxruntime::Node& node, float& min, float& max) {
  auto in_dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type();
  int32_t in_dtype_elem;
  if (in_dtype.has_elem_type()) {
    in_dtype_elem = in_dtype.elem_type();
  } else {
    throw std::invalid_argument("Shl clip get error dtype.");
  }

  if (in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    if (node.InputDefs().size() >= 2) {
      std::string min_name = node.InputDefs()[1]->Name();
      auto min_proto = graph_viewer.GetConstantInitializer(min_name, false);
      Initializer unpacked_min_proto(*min_proto);
      const uint8_t* data_buf = unpacked_min_proto.DataAsByteSpan().data();
      float* min_val = reinterpret_cast<float*>(const_cast<uint8_t*>(data_buf));
      min = *min_val;
    }
    if (node.InputDefs().size() == 3) {
      std::string max_name = node.InputDefs()[2]->Name();
      auto max_proto = graph_viewer.GetConstantInitializer(max_name, false);
      Initializer unpacked_max_proto(*max_proto);
      const uint8_t* data_buf = unpacked_max_proto.DataAsByteSpan().data();
      float* max_val = reinterpret_cast<float*>(const_cast<uint8_t*>(data_buf));
      max = *max_val;
    }
  } else if (in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    if (node.InputDefs().size() >= 2) {
      std::string min_name = node.InputDefs()[1]->Name();
      auto min_proto = graph_viewer.GetConstantInitializer(min_name, false);
      Initializer unpacked_min_proto(*min_proto);
      const uint8_t* data_buf = unpacked_min_proto.DataAsByteSpan().data();
      int16_t* min_value = reinterpret_cast<int16_t*>(const_cast<uint8_t*>(data_buf));
      min = float16_to_float32(min_value[0]);
    }
    if (node.InputDefs().size() == 3) {
      std::string max_name = node.InputDefs()[2]->Name();
      auto max_proto = graph_viewer.GetConstantInitializer(max_name, false);
      Initializer unpacked_max_proto(*max_proto);
      const uint8_t* data_buf = unpacked_max_proto.DataAsByteSpan().data();
      int16_t* max_value = reinterpret_cast<int16_t*>(const_cast<uint8_t*>(data_buf));
      max = float16_to_float32(max_value[0]);
    }
  }
}


std::pair<bool, std::string> IsNodeSupported(const GraphViewer& graph_viewer, const Node* node, csinn_api_enum base_api) {
  const auto& op = node->OpType();
  const std::vector<std::string> cpu_supported_ops{
      "Add", "AveragePool", "Clip", "Concat",
      "Conv", "Flatten", "Gemm", "GlobalAveragePool",
      "MaxPool", "Mul", "Relu", "Sigmoid",
      "Split", "Transpose", "Reshape",
      "Unsqueeze", "MatMul", "LeakyRelu",
      "Softmax", "Erf", "DequantizeLinear", "QuantizeLinear"
      // "Sqrt", "Div", "Sub", "Pow", "ReduceMean", "LayerNormalization",
      // "Sub", "Pow", "Resize",
      // "LeakyRelu", "Slice", "Squeeze", "Gather",
      /*"QLinearConv", "QuantizeLinear",*/};

  const std::vector<std::string> th1520_supported_ops{
      "Add", "AveragePool", "Clip", "Concat",
      "Conv", "Flatten", "Gemm", "GlobalAveragePool",
      "MaxPool", "Mul", "Relu", "Sigmoid",
      "Split", "Transpose", "Reshape",
      "Unsqueeze", "LeakyRelu", "DequantizeLinear", "QuantizeLinear"};

  const std::vector<std::string>& supported_ops = (base_api == CSINN_TH1520) ? th1520_supported_ops : cpu_supported_ops;
  if (std::find(supported_ops.begin(), supported_ops.end(), op) ==
      supported_ops.end()) {
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
      return {false, "pooling only supporte pool2d now"};
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

    if (base_api == CSINN_TH1520){
      float min = std::numeric_limits<float>::min();
      float max = std::numeric_limits<float>::max();
      get_clip_value(graph_viewer, *node, min, max);
      if (min != 0 || !is_one_of<float>(max, 6.0, std::numeric_limits<float>::max())) {
        return {false, "TH1520 only supported clip as relu/relu6."};
      }
    }
  } else if (op == "ReduceMean") {
    const auto noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0);
    if (noop_with_empty_axes) {
      return {false, "ReduceMean not supported noop_with_empty_axes"};
    }
    auto axes = helper.Get("axes", std::vector<int32_t>{0});
    if (node->InputDefs().size() == 2 || axes.size() > 1) {
      return {false, "ReduceMean not supported attribute axes"};
    }
  } else if (op == "DequantizeLinear") {
    // const NodeArg* input = node->InputDefs()[0];
    // bool is_const = graph_viewer.IsInitializedTensor(input->Name());

    // const auto output = node->OutputDefs()[0];
    // const auto& ouptut_nodes = graph_viewer.GetOutputs();

    // bool is_support = false;
    // for (auto out_node : ouptut_nodes) {
    //   if (out_node == output) {
    //     is_support = true;
    //   }
    // }
    // if (!is_const && !is_support) {
    //   return {false, "DequantizeLinear only supported constant input"};
    // }
    auto in_dtype = node->InputDefs()[0]->TypeAsProto()->tensor_type();
    if (in_dtype.has_elem_type()) {
      auto in_dtype_elem = in_dtype.elem_type();
      if (!(in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
            in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_INT8 ||
            in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_INT32)) {
        return {false, "DequantizeLinear only supported UINT8 and INT8"};
      }
    } else {
      return {false, "DequantizeLinear only supported UINT8 and INT8"};
    }

  } else if (op == "QuantizeLinear") {
    const NodeArg* input = node->InputDefs()[0];
    bool is_const = graph_viewer.IsInitializedTensor(input->Name());
    if (is_const) {
      return {false, "QuantizeLinear only supported noconstant input"};
    }
    auto in_dtype = node->InputDefs()[0]->TypeAsProto()->tensor_type();
    if (in_dtype.has_elem_type()) {
      auto in_dtype_elem = in_dtype.elem_type();
      if (in_dtype_elem != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        return {false, "QuantizeLinear only supported FLOAT"};
      }
    } else {
      return {false, "QuantizeLinear only supported FLOAT"};
    }
  } else if (op == "Transpose") {
    const NodeArg* input = node->InputDefs()[0];
    auto in_shape = input->Shape();
    if (base_api == CSINN_TH1520 && in_shape && in_shape->dim_size() != 4) {
      return {false, "transpose only support 4d shape"};
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

std::pair<bool, std::string> IsfusibleNode(std::unordered_map<const Node*, std::string> all_fusible_nodes, const Node* node) {
  if (all_fusible_nodes.count(node)) {
    return {true, "node is fusible."};
  }
  return {false, "node is nonfusible."};
}

bool CheckNodeQuantInfo(const onnxruntime::GraphViewer& graph_viewer, const Node* node) {
  if (is_one_of<std::string>(node->OpType(), "QuantizeLinear", "DequantizeLinear")) {
    return true;
  }

  auto in_size = node->GetInputEdgesCount();
  auto out_size = node->GetOutputEdgesCount();

  auto in_node_iterator = node->InputNodesBegin();
  auto out_node_iterator = node->OutputNodesBegin();
  for (uint i = 0; i < in_size; i++) {
    auto in_node = graph_viewer.GetNode(in_node_iterator->Index());
    if (in_node->OpType() != "DequantizeLinear") {
      return false;
    }
    ++in_node_iterator;
  }

  for (uint i = 0; i < out_size; i++) {
    auto out_node = graph_viewer.GetNode(out_node_iterator->Index());
    if (out_node->OpType() != "QuantizeLinear") {
      return false;
    }
    ++out_node_iterator;
  }

  return true;
}

std::vector<std::vector<int>> GetSupportedNodes(const onnxruntime::GraphViewer& graph_viewer, csinn_profiler_enum profile_level,
                                                csinn_api_enum base_api) {
  std::vector<std::vector<int>> supported_node_vecs;
  std::vector<int> supported_node_vec;
  auto marked_fusible_map = MarkfusibleNodes(graph_viewer);
  auto all_fusible_nodes = GetAllFusionNode(marked_fusible_map);
  const std::vector<NodeIndex>& node_index = graph_viewer.GetNodesInTopologicalOrder();
  for (auto i : node_index) {
    bool supported = false;
    bool is_fusible = false;
    std::string error_msg;
    auto crt_node = graph_viewer.GetNode(i);

    std::tie(supported, error_msg) = IsNodeSupported(graph_viewer, crt_node, base_api);
    if (!supported)
      std::tie(is_fusible, error_msg) = IsfusibleNode(all_fusible_nodes, crt_node);

    if (marked_fusible_map.count("QDQFusion")) {
      // check op input is quant
      supported = CheckNodeQuantInfo(graph_viewer, crt_node);
    }
    if (supported || is_fusible) {
      if (profile_level >= CSINN_PROFILER_LEVEL_TIMER) {
        if (!is_fusible && !supported_node_vec.empty()) {
          supported_node_vecs.push_back(supported_node_vec);
          supported_node_vec.clear();
        }
        supported_node_vec.push_back(i);
        if (!is_fusible) {
          supported_node_vecs.push_back(supported_node_vec);
          supported_node_vec.clear();
        }
      } else {
        supported_node_vec.push_back(i);
      }
    } else {
      const auto& op = crt_node->OpType();
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

std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>>
MarkfusibleNodes(const onnxruntime::GraphViewer& graph_viewer) {
  static std::unordered_map<std::string, FuseMarkerFn> fuse_markers{
      {"LayerNorm", shl_ep::LayerNormMarker},
      {"QDQFusion", shl_ep::QDQMarker},
  };

  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>> marked_fusible_map;
  for (const auto& iter : fuse_markers) {
    std::vector<std::unordered_map<std::string, const Node*>> fuse_nodes = iter.second(graph_viewer);
    if (fuse_nodes.size())
      marked_fusible_map.emplace(iter.first, fuse_nodes);
  }
  return marked_fusible_map;
}

std::unordered_map<const Node*, std::string>
GetAllFusionNode(std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>> marked_fusible_map) {
  std::unordered_map<const Node*, std::string> new_all_fusible_nodes;

  for (auto& fuse_func_iter : marked_fusible_map) {
    auto fused_func_name = fuse_func_iter.first;
    for (auto& node_map : fuse_func_iter.second) {
      for (auto& node_iter : node_map) {
        auto node_type = node_iter.first;
        auto node = node_iter.second;
        if (node_type == "key_node") {
          if (new_all_fusible_nodes.find(node) != new_all_fusible_nodes.end()) {
            new_all_fusible_nodes[node] = fused_func_name;
          } else {
            new_all_fusible_nodes.emplace(node, fused_func_name);
          }
        } else {
          if (new_all_fusible_nodes.find(node) == new_all_fusible_nodes.end())
            new_all_fusible_nodes.emplace(node, "");
        }
      }
    }
  }
  return new_all_fusible_nodes;
}

}  // namespace shl_ep
}  // namespace onnxruntime
