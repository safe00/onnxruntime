// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shl_execution_provider.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/memcpy.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

constexpr const char* SHL = "SHL";
constexpr const char* SHL_CPU = "SHLCpu";

namespace shl_ep {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kShlExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kShlExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kShlExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kShlExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

Status RegisterShlKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kShlExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kShlExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetShlKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterShlKernels(*kernel_registry));

  return kernel_registry;
}
}  // namespace shl_ep

struct ShlFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  std::string node_name;
};

ShlExecutionProvider::ShlExecutionProvider(const std::unordered_map<std::string, std::string>& config)
    : IExecutionProvider{onnxruntime::kShlExecutionProvider}, config_(config) {
  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(SHL, OrtAllocatorType::OrtDeviceAllocator));
      },
      0};

  InsertAllocator(CreateAllocator(default_memory_info));

  AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(SHL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }};

  InsertAllocator(CreateAllocator(cpu_memory_info));
  profiler_level = config_.count("profiler_level") ? shl_ep::GetShlProfilerLevelEnum(config_.at("profiler_level")) : CSINN_PROFILER_LEVEL_UNSET;
}

std::vector<std::vector<int>> ShlExecutionProvider::GetSupportedNodes(
    const onnxruntime::GraphViewer& graph_viewer) const {
  csinn_api_enum base_api = CSINN_REF;
#ifdef RISCV_C920
  base_api =CSINN_C920;
#endif
#ifdef RISCV_TH1520
  base_api = CSINN_TH1520;
#endif
  if (config_.count("base_api")) {
    base_api = shl_ep::GetShlAPIEnum(config_.at("base_api"));
  }

  return shl_ep::GetSupportedNodes(graph_viewer, profiler_level, base_api);
}

std::vector<std::unique_ptr<ComputeCapability>>
ShlExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                    const IKernelLookup& /*kernel_lookup*/) const {
  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Handle If and Loop operators
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() &&
        tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "Shl: Initializers with external data"
                               " location are not currently supported";
      return result;
    }
  }

  std::set<const NodeArg*> all_node_inputs;
  for (const auto& node : graph_viewer.Nodes()) {
    for (const auto input : node.InputDefs()) {
      all_node_inputs.insert(input);
    }
  }
  const auto graph_outputs = graph_viewer.GetOutputs();

  const auto supported_nodes_vector = GetSupportedNodes(graph_viewer);

  int counter = 0;

  for (const auto& group : supported_nodes_vector) {
    if (!group.empty()) {
      std::unordered_set<size_t> node_set;
      node_set.reserve(group.size());
      for (const auto& index : group) {
        node_set.insert(index);
      }
      std::unique_ptr<IndexedSubGraph> sub_graph =
          std::make_unique<IndexedSubGraph>();
      // Find inputs and outputs of the subgraph
      std::unordered_map<const NodeArg*, int>
          fused_inputs, fused_outputs, fused_outputs_to_add;
      std::unordered_set<const NodeArg*> erased;
      int input_order = 0;
      int output_order = 0;

      for (const auto& index : group) {
        sub_graph->nodes.push_back(index);
        const auto& node = graph_viewer.GetNode(index);

        for (const auto& input : node->InputDefs()) {
          const auto& it = fused_outputs.find(input);

          if (it != fused_outputs.end()) {
            fused_outputs.erase(it);
            erased.insert(input);
          }
          // only when input is neither in output list nor erased list, add the
          // input to input list
          else if (erased.find(input) == erased.end()) {
            fused_inputs[input] = input_order++;
          }
        }

        // For output searching, there is a special case:
        // If node's OutputEdges are more than its outputs, meaning certain
        // output is used more than once,
        // if the output is connected to nodes that don't belong to the
        // subgraph, the output need to be added to the output list
        if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
          for (auto it = node->OutputEdgesBegin(),
                    end = node->OutputEdgesEnd();
               it != end; ++it) {
            const auto& node_idx = it->GetNode().Index();
            const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];

            if (node_set.find(node_idx) != node_set.end()) {
              const auto& iter = fused_inputs.find(output);

              if (iter != fused_inputs.end()) {
                fused_inputs.erase(iter);
                erased.insert(output);
              } else if (erased.find(output) == erased.end()) {
                fused_outputs[output] = output_order++;
              }
            } else {
              fused_outputs_to_add[output] = output_order++;
            }
          }
        } else {
          for (const auto& output : node->OutputDefs()) {
            const auto& it = fused_inputs.find(output);

            if (it != fused_inputs.end()) {
              fused_inputs.erase(it);
              erased.insert(output);
            }
            // only when output is neither in input list nor erased list,
            // add the output to output list
            else if (erased.find(output) == erased.end()) {
              fused_outputs[output] = output_order++;
            }
          }
        }
      }

      fused_outputs.insert(
          fused_outputs_to_add.begin(), fused_outputs_to_add.end());

      // Sort inputs and outputs by the order they were added
      std::map<int, const NodeArg*> inputs, outputs;

      for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
        if (it->first->Type() != nullptr)
          inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }

      for (auto it = fused_outputs.begin(),
                end = fused_outputs.end();
           it != end; ++it) {
        for (const auto& x : all_node_inputs) {
          if (x->Name() == it->first->Name()) {
            outputs.insert(
                std::pair<int, const NodeArg*>(it->second, it->first));
            break;
          }
        }
        if (std::find(graph_outputs.begin(),
                      graph_outputs.end(), it->first) != graph_outputs.end()) {
          outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
        }
      }

      // Assign inputs and outputs to subgraph's meta_def
      auto meta_def =
          std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();

      if (profiler_level >= CSINN_PROFILER_LEVEL_TIMER) {
        const auto& node = graph_viewer.GetNode(group[0]);
        std::string out_name;
        for (auto out : outputs) {
          if (out.second) {
            out_name = out.second->Name();
            break;
          }
        }
        meta_def->name = "SHL_" + node->OpType() + "_(" + out_name + ")";
      } else {
        meta_def->name = "SHL_" + std::to_string(counter++);
      }
      meta_def->domain = kMSDomain;

      for (const auto& input : inputs) {
        meta_def->inputs.push_back(input.second->Name());
      }

      for (const auto& output : outputs) {
        meta_def->outputs.push_back(output.second->Name());
      }

      meta_def->since_version = 1;
      sub_graph->SetMetaDef(std::move(meta_def));

      result.push_back(
          std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

Status ShlExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                     std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_view = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    // build shl tensor and graph
    csinn_session* current_sess = csinn_alloc_session();
    shl_ep::OnnxToShlConverter converter(current_sess, config_);
    converter.Convert(graph_view);

    std::unordered_map<std::string, size_t> names2index;
    const auto& input_defs = fused_node.InputDefs();
    names2index.reserve(input_defs.size());
    for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
      names2index[input_defs[i]->Name()] = i;
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context,
                                         FunctionState* state) {
      std::unique_ptr<ShlFuncState> p = std::make_unique<ShlFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, context->node_name};
      *state = p.release();
      return 0;
    };

    compute_info.release_state_func = [current_sess](FunctionState state) {
      if (state) {
        ShlFuncState* p = static_cast<ShlFuncState*>(state);
        csinn_session_deinit(current_sess);
        csinn_free_session(current_sess);
        delete p;
      }
    };

    compute_info.compute_func = [=](FunctionState state,
                                    const OrtApi* /*api*/,
                                    OrtKernelContext* context) {
      auto get_shape_from_shl_tensor = [](const csinn_tensor* tensor) -> std::vector<int64_t> {
        std::vector<int64_t> shape(tensor->dim_count);
        std::transform(tensor->dim, tensor->dim + tensor->dim_count, shape.begin(), [](int32_t val) -> int64_t {
          if (val < 0){
            LOGS_DEFAULT(WARNING) << "Shl: Tensor shape cannot contain any negative value. use 1 instead of " << val;
            val = 1;
          }
          return static_cast<int64_t>(val);
        });
        return shape;
      };

      Ort::KernelContext ctx(context);
      const size_t n_outputs = ctx.GetOutputCount();
      int input_num = csinn_get_input_number(current_sess);

      std::vector<void *> malloced_input_pr;
      for (int i = 0; i < input_num; i++) {
        csinn_tensor* shl_input = csinn_alloc_tensor(current_sess);
        csinn_get_input(i, shl_input, current_sess);
        size_t index = names2index.at(shl_input->name);
        auto input_tensor = ctx.GetInput(index);
        auto input_tensor_shape_info = input_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = input_tensor_shape_info.GetShape();
        if (shl_input->dim_count == 0) {
          current_sess->dynamic_shape = true;
        }
        if (current_sess->dynamic_shape) {
          shl_ep::UpdateShlTensorDim(shl_input, shape);
        }
        auto in_data_dtype = shl_ep::GetShlDtypeEnum(input_tensor_shape_info.GetElementType());

        if (shl_input->dtype != in_data_dtype){
          csinn_tensor* shl_input_tmp = csinn_alloc_tensor(NULL);
          csinn_tensor_copy(shl_input_tmp, shl_input);
          shl_input_tmp->dtype = in_data_dtype;
          shl_input_tmp->data = const_cast<void*>(input_tensor.GetTensorRawData());
          shl_input->data = shl_mem_alloc_aligned(csinn_tensor_byte_size(shl_input), 0);
          malloced_input_pr.push_back(shl_input->data);
        #ifdef RISCV_TH1520
          #ifdef __x86_64__
            csinn_tensor_data_convert(shl_input, shl_input_tmp);
            csinn_free_tensor(shl_input_tmp);
          #else
            void* quant_data = shl_ep::shl_c920_f32_to_input_dtype(i, reinterpret_cast<float*>(shl_input_tmp->data), current_sess);
            memcpy(shl_input->data, quant_data, csinn_tensor_byte_size(shl_input));
            shl_mem_free(quant_data);
          #endif
        #else
          csinn_tensor_data_convert(shl_input, shl_input_tmp);
          csinn_free_tensor(shl_input_tmp);
        #endif
        } else {
          shl_input->data = const_cast<void*>(input_tensor.GetTensorRawData());
        }
        csinn_update_input(i, shl_input, current_sess);
        csinn_free_tensor(shl_input);
      }

      uint64_t start_time, end_time;
      if (config_.count("debug_level") && shl_ep::GetShlDebugLevelEnum(config_.at("debug_level")) >= CSINN_DEBUG_LEVEL_INFO){
        start_time = shl_get_timespec();
        csinn_session_run(current_sess);
        end_time = shl_get_timespec();
        printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time-start_time))/1000000,
                      1000000000.0/((float)(end_time-start_time)));
      } else {
        csinn_session_run(current_sess);
      }

      for (size_t i = 0; i < n_outputs; i++) {
        csinn_tensor* shl_output = csinn_alloc_tensor(current_sess);
        csinn_get_output(i, shl_output, current_sess);
        auto output_shape = get_shape_from_shl_tensor(shl_output);
        const auto output_tensor = ctx.GetOutput(i, output_shape);
        void* output_buf = const_cast<void*>(output_tensor.GetTensorRawData());
        auto out_data_dtype = shl_ep::GetShlDtypeEnum(output_tensor.GetTensorTypeAndShapeInfo().GetElementType());
        if (shl_output->dtype != out_data_dtype){
          csinn_tensor* shl_output_tmp = csinn_alloc_tensor(NULL);
          csinn_tensor_copy(shl_output_tmp, shl_output);
          shl_output_tmp->dtype = out_data_dtype;
          int out_size = csinn_tensor_byte_size(shl_output_tmp);
        #ifdef RISCV_TH1520
          #ifdef __x86_64__
            shl_output_tmp->data = malloc(out_size);
            csinn_tensor_data_convert(shl_output_tmp, shl_output);
            memcpy(output_buf, shl_output_tmp->data, out_size);
            shl_mem_free(shl_output_tmp->data);
          #else
            void* output_float = shl_ep::shl_c920_output_to_f32_dtype(i, shl_output->data, current_sess);
            memcpy(output_buf, output_float, out_size);
            shl_mem_free(output_float);
          #endif
        #else
          shl_output_tmp->data = malloc(out_size);
          csinn_tensor_data_convert(shl_output_tmp, shl_output);
          memcpy(output_buf, shl_output_tmp->data, out_size);
          shl_mem_free(shl_output_tmp->data);
        #endif
          csinn_free_tensor(shl_output_tmp);
        } else {
          int out_size = csinn_tensor_byte_size(shl_output);
          memcpy(output_buf, shl_output->data, out_size);
        }

        if (current_sess->base_api != CSINN_TH1520){
          shl_mem_free(shl_output->data);
        }
        csinn_free_tensor(shl_output);
      }

      for (auto x : malloced_input_pr){
        shl_mem_free(x);
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::shared_ptr<KernelRegistry> ShlExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = shl_ep::GetShlKernelRegistry();
  return kernel_registry;
}

}  // namespace onnxruntime
