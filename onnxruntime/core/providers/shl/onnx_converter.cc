// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "onnx_converter.h"
#include "shl_common.h"

using std::string;
using std::vector;

namespace onnxruntime {
namespace shl_ep {

#define HAS(map, key) \
  (map.find(key) != map.end())

void OnnxToShlConverter::InitAllTensor(const GraphViewer& graph_viewer) {
  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  const std::vector<const NodeArg*>& all_nodes = graph_viewer.GetInputsIncludingInitializers();

  // input and constant
  for (const auto* node : all_nodes) {
    csinn_tensor* shl_tensor = shl_ep::CreateShlTensor(node, session_);
    shl_tensor_map[node->Name()] = shl_tensor;
  }

  // per layer output
  for (const auto& layer : graph_viewer.Nodes()) {
    for (const auto output : layer.OutputDefs()) {
      csinn_tensor* shl_tensor = shl_ep::CreateShlTensor(output, session_);
      shl_tensor_map[output->Name()] = shl_tensor;
    }
  }

  const auto& input_tensors = graph_viewer.GetInputs();
  const auto& output_tensors = graph_viewer.GetOutputs();

  // set constant buf
  for (const auto& tensor : init_tensors) {
    Initializer unpacked_tensor(*(tensor.second));
    const uint8_t* data_buf = unpacked_tensor.DataAsByteSpan().data();
    csinn_tensor* shl_tensor = shl_tensor_map.at(tensor.first);
    int size = csinn_tensor_byte_size(shl_tensor);
    shl_tensor->data = shl_mem_alloc(size);
    shl_tensor->layout = GetShlWeightLayoutEnum(shl_tensor->dim_count);
    memcpy(shl_tensor->data, data_buf, size);
    shl_tensor->is_const = 1;
  }

  csinn_set_input_number(input_tensors.size(), session_);
  csinn_set_output_number(output_tensors.size(), session_);

  auto set_sess_dynamic_shape = [session = session_](csinn_tensor* t) {
    for (int i = 0; i < t->dim_count; i++) {
      if (t->dim[i] < 0) {
        if (t->dim[i] == -1) {
          session->dynamic_shape = true;
          break;
        } else {
          throw std::invalid_argument("Error obtaining shape value.");
        }
      }
    }
  };

  for (uint i = 0; i < input_tensors.size(); i++) {
    auto tensor = input_tensors[i];
    auto shl_tensor = shl_tensor_map.at(tensor->Name());
    set_sess_dynamic_shape(shl_tensor);
    csinn_set_tensor_entry(shl_tensor, session_);
    csinn_set_input(i, shl_tensor, session_);
  }
}

void OnnxToShlConverter::Convert(const GraphViewer& graph_viewer) {
  // 1.  create all shl tensor and set constant data
  InitAllTensor(graph_viewer);

  // 2. set attr and build shl graph
  for (const auto& node : graph_viewer.Nodes()) {
    NodeConvert(node);
  }

  // 3. set output tensor
  const auto& output_tensors = graph_viewer.GetOutputs();
  for (uint i = 0; i < output_tensors.size(); i++) {
    auto tensor_name = output_tensors[i];
    auto shl_tensor = shl_tensor_map[tensor_name->Name()];
    csinn_set_output(i, shl_tensor, session_);
  }
  csinn_session_setup(session_);
}

void OnnxToShlConverter::Conv2D(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_conv2d_params>(session_);
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  const auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});
  const auto dilations = helper.Get("dilations", std::vector<int32_t>{1, 1});

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->group = helper.Get("group", 1);
  params->stride_height = strides[0];
  params->stride_width = strides[1];
  params->pad_top = pads[0];
  params->pad_left = pads[1];
  params->pad_down = pads[2];
  params->pad_right = pads[3];
  params->dilation_height = dilations[0];
  params->dilation_width = dilations[1];

  std::string bias;
  if (node.InputDefs().size() >= 3) {
    bias = node.InputDefs()[2]->Name();
  }

  std::string input = node.InputDefs()[0]->Name();
  std::string weight = node.InputDefs()[1]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  auto weight_tensor = shl_tensor_map.at(weight);
  auto bias_tensor = bias.size() ? shl_tensor_map.at(bias) : csinn_alloc_tensor(session_);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  csinn_conv2d_init(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
  csinn_conv2d(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
}

void OnnxToShlConverter::Conv(const onnxruntime::Node& node) {
  auto input = node.InputDefs()[0];
  switch (shl_tensor_map[input->Name()]->dim_count) {
    case 3:
      Conv1D(node);
      break;
    case 4:
      Conv2D(node);
      break;
    case 5:
      Conv3D(node);
      break;
    default:
      throw std::invalid_argument("Unsupported dims of Conv.");
      ;
  }
}

void OnnxToShlConverter::ReLU(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_relu_params>(session_);
  params->base.name = const_cast<char*>(node.Name().c_str());
  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  csinn_relu_init(input_tensor, output_tensor, params);
  csinn_relu(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::LeakyRelu(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  const auto alpha = helper.Get("alpha", 0.01f);
  auto params = shl_ep::GetShlParams<csinn_relu_params>(session_);
  params->base.name = const_cast<char*>(node.Name().c_str());
  params->n = alpha;

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  csinn_leaky_relu_init(input_tensor, output_tensor, params);
  csinn_leaky_relu(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::SetPoolAttr(csinn_pool_params* params, const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  const auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});
  const auto kernel_shape = helper.Get("kernel_shape", std::vector<int32_t>{1, 1});
  const auto count_include_pad = helper.Get("count_include_pad", 0);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->stride_height = strides[0];
  params->stride_width = strides[1];
  params->pad_top = pads[0];
  params->pad_left = pads[1];
  params->pad_down = pads[2];
  params->pad_right = pads[3];
  params->filter_height = kernel_shape[0];
  params->filter_width = kernel_shape[1];
  params->count_include_pad = count_include_pad;
  params->ceil_mode = helper.Get("ceil_mode", 0);
}

void OnnxToShlConverter::AvgPool2d(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_pool_params>(session_);
  SetPoolAttr(params, node);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_avgpool2d_init(input_tensor, output_tensor, params);
  csinn_avgpool2d(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::MaxPool2d(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_pool_params>(session_);
  SetPoolAttr(params, node);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_maxpool2d_init(input_tensor, output_tensor, params);
  csinn_maxpool2d(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::GlobalAvgPool(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_pool_params>(session_);
  SetPoolAttr(params, node);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_global_avgpool2d_init(input_tensor, output_tensor, params);
  csinn_global_avgpool2d(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::AveragePool(const onnxruntime::Node& node) {
  auto input = node.InputDefs()[0];
  switch (shl_tensor_map[input->Name()]->dim_count) {
    case 4:
      AvgPool2d(node);
      break;
    default:
      throw std::invalid_argument("Unsupported dims of AvgPool.");
  }
}

void OnnxToShlConverter::MaxPool(const onnxruntime::Node& node) {
  auto input = node.InputDefs()[0];
  switch (shl_tensor_map[input->Name()]->dim_count) {
    case 4:
      MaxPool2d(node);
      break;
    default:
      throw std::invalid_argument("Unsupported dims of AvgPool.");
  }
}

void OnnxToShlConverter::Softmax(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_softmax_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->axis = helper.Get("axis", -1);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_softmax_init(input_tensor, output_tensor, params);
  csinn_softmax(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Gemm(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_fc_params>(session_);

  std::string bias;
  if (node.InputDefs().size() >= 3) {
    bias = node.InputDefs()[2]->Name();
  }

  std::string input = node.InputDefs()[0]->Name();
  std::string weight = node.InputDefs()[1]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  auto weight_tensor = shl_tensor_map.at(weight);
  auto bias_tensor = bias.size() ? shl_tensor_map.at(bias) : csinn_alloc_tensor(session_);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  params->units = weight_tensor->dim[0];

  csinn_fullyconnected_init(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
  csinn_fullyconnected(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
}

void OnnxToShlConverter::MatMul(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_matmul_params>(session_);

  std::string input_a = node.InputDefs()[0]->Name();
  std::string input_b = node.InputDefs()[1]->Name();
  auto a_tensor = shl_tensor_map.at(input_a);
  auto b_tensor = shl_tensor_map.at(input_b);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  params->trans_a = false;
  params->trans_b = false;

  csinn_matmul_init(a_tensor, b_tensor, output_tensor, params);
  csinn_matmul(a_tensor, b_tensor, output_tensor, params);
}

void OnnxToShlConverter::Add(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_add_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_add(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Sub(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_sub_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_sub(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Mul(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_mul_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_mul(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Flatten(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_flatten_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_flatten_init(input_tensor, output_tensor, params);
  csinn_flatten(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Sigmoid(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_sigmoid_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_sigmoid_init(input_tensor, output_tensor, params);
  csinn_sigmoid(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Concat(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_concat_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->axis = helper.Get("axis", -1);
  params->inputs_count = node.InputDefs().size();

  csinn_tensor* inputs[params->inputs_count];

  for (int32_t i = 0; i < params->inputs_count; i++) {
    std::string input = node.InputDefs()[i]->Name();
    inputs[i] = shl_tensor_map.at(input);
  }

  if (params->axis < 0) {
    params->axis += inputs[0]->dim_count;
  }

  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_concat_init(inputs, output_tensor, params);
  csinn_concat(inputs, output_tensor, params);
}

void OnnxToShlConverter::Resize(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_resize_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());
  const auto coordinate_transformation_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  params->align_corners = coordinate_transformation_mode == "align_corners";
  const auto mode = helper.Get("mode", "nearest");

  if (mode == "bilinear") {
    params->resize_mode = CSINN_RESIZE_BILINEAR;
  } else if (mode == "linear") {
    params->resize_mode = CSINN_RESIZE_BILINEAR;
  } else if (mode == "nearest") {
    params->resize_mode = CSINN_RESIZE_NEAREST_NEIGHBOR;
  } else {
    throw std::invalid_argument("Unsupported dims of AvgPool.");
  }

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_resize_init(input_tensor, output_tensor, params);
  csinn_resize(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Reshape(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_reshape_params>(session_);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  params->shape = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * 8));

  for (int i = 0; i < output_tensor->dim_count; i++) {
    params->shape[i] = output_tensor->dim[i];
  }
  params->shape_num = output_tensor->dim_count;

  csinn_reshape_init(input_tensor, output_tensor, params);
  csinn_reshape(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Transpose(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_transpose_params>(session_);
  const auto perm = helper.Get("perm", vector<int>{0, 1, 2});

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->permute_num = perm.size();
  params->permute = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * perm.size()));

  for (uint i = 0; i < perm.size(); i++) {
    params->permute[i] = perm[i];
  }

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  csinn_transpose_init(input_tensor, output_tensor, params);
  csinn_transpose(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Split(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_split_params>(session_);
  params->base.name = const_cast<char*>(node.Name().c_str());
  params->output_num = node.OutputDefs().size();
  if (node.InputDefs().size() == 2) {
    std::string split = node.InputDefs()[1]->Name();
    csinn_tensor* split_tensor = shl_tensor_map.at(split);
    int split_length = split_tensor->dim[0] - 1;
    params->split_index = reinterpret_cast<int32_t*>(malloc(split_length * sizeof(int32_t)));
    int64_t* index = reinterpret_cast<int64_t*>(split_tensor->data);
    for (int i = 0; i < split_length; i++) {
      if (i == 0) {
        params->split_index[i] = index[i];
      } else {
        params->split_index[i] = params->split_index[i - 1] + index[i];
      }
    }
  }

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  csinn_tensor* output_tensors[params->output_num];

  for (int i = 0; i < params->output_num; i++) {
    std::string output = node.OutputDefs()[i]->Name();
    output_tensors[i] = shl_tensor_map.at(output);
  }
  auto axis = helper.Get("axis", 0);
  params->axis = axis < 0 ? axis + input_tensor->dim_count : axis;
  csinn_split_init(input_tensor, output_tensors, params);
  csinn_split(input_tensor, output_tensors, params);
}

void OnnxToShlConverter::Pow(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_);
  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_power_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_power(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Clip(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_clip_params>(session_);
  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
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
      auto min_tensor = shl_tensor_map.at(min_name);
      float* min_value = reinterpret_cast<float*>(min_tensor->data);
      params->min_value = min_value[0];
    }
    if (node.InputDefs().size() == 3) {
      std::string max_name = node.InputDefs()[2]->Name();
      auto max_tensor = shl_tensor_map.at(max_name);
      float* max_value = reinterpret_cast<float*>(max_tensor->data);
      params->max_value = max_value[0];
    }
  } else if (in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    if (node.InputDefs().size() >= 2) {
      std::string min_name = node.InputDefs()[1]->Name();
      auto min_tensor = shl_tensor_map.at(min_name);
      int16_t* min_value = reinterpret_cast<int16_t*>(min_tensor->data);
      params->min_value = float16_to_float32(min_value[0]);
    }
    if (node.InputDefs().size() == 3) {
      std::string max_name = node.InputDefs()[2]->Name();
      auto max_tensor = shl_tensor_map.at(max_name);
      int16_t* max_value = reinterpret_cast<int16_t*>(max_tensor->data);
      params->max_value = float16_to_float32(max_value[0]);
    }
  }

  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  csinn_clip_init(input_tensor, output_tensor, params);
  csinn_clip(input_tensor, output_tensor, params);
}

}  // namespace shl_ep
}  // namespace onnxruntime
