// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "op_fusion.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace shl_ep {

/**
Layer Normalization will fuse LayerNormalization into one node :
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                               ^
                      |                                               |
                      +-----------------------------------------------+
It also handles cases of duplicated sub nodes exported from older version of PyTorch :
+---------------------+
|                     v
|          +-------> Sub ---------------------------------------------+
|          |                                                          |
|          |                                                          v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
|                     ^
|                     |
+---------------------+

In recent pytorch, Cast nodes may be inserted before Pow to ensure that both inputs 'base' and 'power' are the same type
due to restriction in older opsets. Therefore, Layer Normalization will also handle the case below :
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Cast --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                              |                                                ^
                              |                                                |
                              +------------------------------------------------+
+---------------------+       Cast
|                     |        |
|                     v        v
X --> ReduceMean --> Sub -->  Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                                ^
                      |                                                |
                      +------------------------------------------------+

When using Apex O2, a Cast node may be inserted between Div and Mul, Layer Normalization will also handle the case below:
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
                      |                                               ^
                      |                                               |
                      +-----------------------------------------------+

OR

         +---------------------+
         |                     |
         |                     v
X --> Cast --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
                               |                                               ^
                               |                                               |
                               +-----------------------------------------------+

Logically since LayerNormalization supports input and scale/bias in different data types, and during the kernel execution,
data are casted to float/double to calculate for precision, so if there is any Cast Ops in the sub-graph, we can remove it.
Such Cast Op can be the input of the sub-graph, or an Cast Op between the Div and Mul nodes.
*/
std::vector<std::unordered_map<std::string, const Node*>>
LayerNormMarker(const onnxruntime::GraphViewer& graph_viewer) {
  std::vector<std::unordered_map<std::string, const Node*>> marked_nodes;

  const auto& graph = graph_viewer.GetGraph();
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* p_reduce_mean = graph.GetNode(node_index);
    if (p_reduce_mean == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*p_reduce_mean, "ReduceMean", {1, 11, 13, 18}) ||
        (p_reduce_mean->GetOutputEdgesCount() != 1 && p_reduce_mean->GetOutputEdgesCount() != 2) ||
        graph.NodeProducesGraphOutput(*p_reduce_mean)) {
      continue;
    }

    // Loop through the children of current "ReduceMean" node. See if they match ["Sub"] or ["Sub", "Sub"]
    int subCnt = 0;
    const Node* p_sub_node = nullptr;
    const Node* p_sub_node_dup = nullptr;
    for (auto iter = p_reduce_mean->OutputNodesBegin(); iter != p_reduce_mean->OutputNodesEnd(); ++iter) {
      if ((*iter).OpType().compare("Sub") == 0) {
        if (subCnt == 0) {
          p_sub_node = &(*iter);
        } else {
          p_sub_node_dup = &(*iter);
        }
        subCnt++;
      } else {
        // doesn't match layer norm pattern. break.
        subCnt = -1;
        break;
      }
    }

    if (subCnt != 1 && subCnt != 2) {
      continue;
    }

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*p_sub_node, "Sub", {7, 13, 14}) ||
        p_sub_node->GetExecutionProviderType() != p_reduce_mean->GetExecutionProviderType()) {
      continue;
    }

    // Apex O2 pattern specific match starts...
    // Logically since we support input and scale/bias in different data types, those Cast Ops in sub-graph
    // can be removed. This is one possible place a Cast Op can exist, that is the input of the sub-graph.
    // Make sure it consumes by the sub-graph only.
    const NodeArg* p_reduce_mean_input = p_reduce_mean->InputDefs()[0];
    const NodeArg* p_sub_input = nullptr;
    for (const NodeArg* node_arg : p_sub_node->InputDefs()) {
      if (node_arg != p_reduce_mean->OutputDefs()[0]) {
        p_sub_input = node_arg;
        break;
      }
    }

    if (!p_reduce_mean_input || !p_sub_input || p_reduce_mean_input != p_sub_input) {
      continue;
    }

    if (p_sub_node_dup) {
      const NodeArg* p_sub_dup_input = nullptr;
      for (const NodeArg* node_arg : graph.GetNode(p_sub_node_dup->Index())->InputDefs()) {
        if (node_arg != p_reduce_mean->OutputDefs()[0]) {
          p_sub_dup_input = node_arg;
          break;
        }
      }
      if (!p_sub_dup_input || p_reduce_mean_input != p_sub_dup_input) {
        continue;
      }
    }

    const Node* p_reduce_mean_input_node = graph_utils::GetInputNode(*p_reduce_mean, 0);
    bool has_leading_cast = false;
    if (p_reduce_mean_input_node) {
      const Node& reduce_mean_input_node = *graph.GetNode(p_reduce_mean_input_node->Index());
      // If input to the 1st ReduceMean is a Cast, and the Cast has same consumer count as subCnt + 1
      if (graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean_input_node, "Cast", {9, 13}) &&
          optimizer_utils::CheckOutputEdges(graph, reduce_mean_input_node, static_cast<size_t>(subCnt) + 1)) {
        // nodes_to_remove.insert(nodes_to_remove.begin(), reduce_mean_input_node);
        has_leading_cast = true;
      }
    }
    // Apex O2 pattern specific match ends...

    // Find the "Div" node after "Sub". It's possible that there is "Cast" node after "Sub" node.
    const Node* p_cast1 = nullptr;
    if (!p_sub_node_dup && p_sub_node->GetOutputEdgesCount() == 1) {
      const Node& cast_node = *graph.GetNode(p_sub_node->OutputNodesBegin()->Index());
      if (graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {9, 13}) &&
          optimizer_utils::CheckOutputEdges(graph, cast_node, 2u)) {
        p_cast1 = &cast_node;
        // nodes_to_remove.push_back(cast_node);
      }
    }

    if (!optimizer_utils::CheckOutputEdges(graph, *p_sub_node, subCnt == 1 && !p_cast1 ? 2u : 1u)) {
      continue;
    }

    const Node* p_div = nullptr;
    p_div = graph_utils::FirstChildByType(p_cast1 ? *p_cast1 : *p_sub_node, "Div");

    // Find the sub_dup node if exist
    if (p_sub_node_dup != nullptr) {
      const Node& sub_node_dup = *graph.GetNode(p_sub_node_dup->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(sub_node_dup, "Sub", {7, 13, 14}) ||
          !optimizer_utils::CheckOutputEdges(graph, *p_sub_node, 1)) {
        continue;
      }
      // nodes_to_remove.push_back(sub_node_dup);
      // Find Div node after the duplicated sub node if it's not found after the first sub node.
      if (p_div == nullptr) {
        p_div = graph_utils::FirstChildByType(sub_node_dup, "Div");
      }
    }

    if (p_div == nullptr) {
      continue;
    }
    const Node& div_node = *graph.GetNode(p_div->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div_node, "Div", {7, 13, 14}) ||
        !optimizer_utils::CheckOutputEdges(graph, div_node, 1)) {
      continue;
    }
    // nodes_to_remove.push_back(div_node);

    // Traceback the div node to find sqrt --> div
    const Node* p_sqrt = graph_utils::FirstParentByType(div_node, "Sqrt");
    if (p_sqrt == nullptr) {
      continue;
    }
    const Node& sqrt_node = *graph.GetNode(p_sqrt->Index());

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sqrt_node, "Sqrt", {6, 13}) ||
        !optimizer_utils::CheckOutputEdges(graph, sqrt_node, 1) ||
        sqrt_node.GetInputEdgesCount() == 0) {
      continue;
    }
    // nodes_to_remove.push_back(sqrt_node);

    // Traceback the sqrt node to find add --> sqrt
    const Node& add2_node = *graph.GetNode(sqrt_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7, 13, 14}) ||
        !optimizer_utils::CheckOutputEdges(graph, add2_node, 1)) {
      continue;
    }
    // nodes_to_remove.push_back(add2_node);
    // Traceback the add node to find reduceMean --> add
    const Node* p_reduce_mean2 = nullptr;

    p_reduce_mean2 = graph_utils::FirstParentByType(add2_node, "ReduceMean");
    if (p_reduce_mean2 == nullptr) {
      continue;
    }
    const Node& reduce_mean2_node = *graph.GetNode(p_reduce_mean2->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce_mean2_node, "ReduceMean", {1, 11, 13, 18}) ||
        !optimizer_utils::CheckOutputEdges(graph, reduce_mean2_node, 1) ||
        reduce_mean2_node.GetInputEdgesCount() == 0) {
      continue;
    }
    // nodes_to_remove.push_back(reduce_mean2_node);

    // Traceback the reduceMean node to find pow --> reduceMean
    const Node& pow_node = *graph.GetNode(reduce_mean2_node.InputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow_node, "Pow", {7, 12, 13, 15}) ||
        !optimizer_utils::CheckOutputEdges(graph, pow_node, 1)) {
      continue;
    }
    // nodes_to_remove.push_back(pow_node);

    // check if Cast node exists: either between sub and pow, or as second input to pow
    const Node* p_cast2 = graph_utils::FirstParentByType(pow_node, "Cast");
    if (p_cast2 != nullptr && p_cast2 != p_cast1) {
      const Node& cast_node = *graph.GetNode(p_cast2->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(cast_node, "Cast", {9, 13}) ||
          !optimizer_utils::CheckOutputEdges(graph, cast_node, 1)) {
        continue;
      }
      // nodes_to_remove.push_back(cast_node);
    } else if (!p_cast2) {
      const Node* p_sub2_node = graph_utils::FirstParentByType(pow_node, "Sub");
      if (!p_sub2_node || (p_sub2_node != p_sub_node && p_sub2_node != p_sub_node_dup)) {
        continue;
      }
    }

    // Apex O2 pattern specific match starts...
    // Logically since we support input and scale/bias in different data types, those Cast Ops in sub-graph
    // can be removed. This is one possible place a Cast Op can exist, that is between Div and Mul nodes.
    // div --> mul or div --> cast --> mul
    const Node* mul_node_p = graph.GetNode(div_node.OutputNodesBegin()->Index());
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*mul_node_p, "Cast", {9, 13}) &&
        optimizer_utils::CheckOutputEdges(graph, *mul_node_p, 1)) {
      // nodes_to_remove.push_back(*mul_node_p);
      mul_node_p = graph.GetNode(mul_node_p->OutputNodesBegin()->Index());
    }
    // Apex O2 pattern specific match ends...

    const Node& mul_node = *mul_node_p;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        !optimizer_utils::CheckOutputEdges(graph, mul_node, 1)) {
      continue;
    }
    // nodes_to_remove.push_back(mul_node);

    // mul --> add
    // Need not check output edges of last node since they will be moved to fused node.
    auto* last_add_node_p = graph.GetNode(mul_node.OutputNodesBegin()->Index());
    const Node& last_add_node = *last_add_node_p;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(last_add_node, "Add", {7, 13, 14})) {
      continue;
    }
    // nodes_to_remove.push_back(last_add_node);

    // // get axes attributes
    const onnxruntime::NodeAttributes& attributes = p_reduce_mean->GetAttributes();
    std::vector<int64_t> axes_values;
    // TODO: modify this codes when opset >= 18 (axes is an input).
    if (attributes.find("axes") != attributes.end()) {
      axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    } else if (p_reduce_mean->InputDefs().size() == 2) {
      auto axes = p_reduce_mean->InputDefs()[1];
      auto axes_const = graph.GetConstantInitializer(axes->Name(), true);
      if (axes_const != nullptr) {
        Initializer initializer{*axes_const, graph.ModelPath()};
        axes_values.insert(axes_values.end(), initializer.DataAsSpan<int64_t>().begin(), initializer.DataAsSpan<int64_t>().end());
      }
    }

    // Get the inputs for the new LayerNormalization node.
    // scale and bias could be multi-dims; we only support it for training at the moment
    // because SkipLayerNorm kernel, for example, has dependency on single dim size
    const NodeArg* scale = nullptr;
    const NodeArg* bias = nullptr;
    for (size_t i = 0; i < mul_node.InputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(mul_node.InputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, mul_node.InputDefs()[i])) {
        // Scale must be 1d.
        if (mul_node.InputDefs()[i]->Shape()->dim_size() == 1) {
          scale = mul_node.InputDefs()[i];
        }
      }
    }

    for (size_t i = 0; i < last_add_node.InputDefs().size(); i++) {
      if (graph_utils::NodeArgIsConstant(graph, *(last_add_node.InputDefs()[i])) ||
          graph_utils::IsGraphInput(graph, last_add_node.InputDefs()[i])) {
        // Bias must be 1d.
        if (last_add_node.InputDefs()[i]->Shape()->dim_size() == 1) {
          bias = last_add_node.InputDefs()[i];
        }
      }
    }
    if (scale == nullptr || bias == nullptr) {
      continue;
    }

    // Scale and bias must have the same shape.
    bool same_dim = true;
    for (int i = 0; i < scale->Shape()->dim_size(); i++) {
      if (scale->Shape()->dim(i).dim_value() != bias->Shape()->dim(i).dim_value()) {
        same_dim = false;
        break;
      }
    }
    if (!same_dim)
      continue;

    std::unordered_map<std::string, const Node*> layernorm_nodes;

    const Node* first_input = has_leading_cast ? graph.GetNode(p_reduce_mean_input_node->Index())
                                               : p_reduce_mean;
    layernorm_nodes["key_node"] = p_reduce_mean;
    layernorm_nodes["first_node"] = first_input;
    layernorm_nodes["sub"] = graph.GetNode(p_sub_node->Index());
    layernorm_nodes["div"] = graph.GetNode(p_div->Index());
    layernorm_nodes["sqrt"] = graph.GetNode(p_sqrt->Index());
    layernorm_nodes["add1"] = graph.GetNode(sqrt_node.InputNodesBegin()->Index());
    layernorm_nodes["mean2"] = graph.GetNode(p_reduce_mean2->Index());
    layernorm_nodes["pow"] = graph.GetNode(reduce_mean2_node.InputNodesBegin()->Index());
    layernorm_nodes["mul"] = mul_node_p;
    layernorm_nodes["add2"] = graph.GetNode(mul_node.OutputNodesBegin()->Index());
    if (p_cast1 != nullptr) {
      layernorm_nodes["cast1"] = graph.GetNode(p_sub_node->OutputNodesBegin()->Index());
    }

    if (p_cast2 != nullptr) {
      layernorm_nodes["cast2"] = graph.GetNode(p_cast2->Index());
    }
    if (p_sub_node_dup != nullptr) {
      layernorm_nodes["sub2"] = graph.GetNode(p_sub_node_dup->Index());
    }

    marked_nodes.push_back(layernorm_nodes);
  }
  return marked_nodes;
}
}  // namespace shl_ep
}  // namespace onnxruntime
