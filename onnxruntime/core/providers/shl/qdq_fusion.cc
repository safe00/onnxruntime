// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "op_fusion.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace shl_ep {

/**
QDQ will fuse qaunt->node->dquant into one node :
*/

std::vector<std::unordered_map<std::string, const Node*>>
QDQMarker(const onnxruntime::GraphViewer& graph_viewer) {
  std::vector<std::unordered_map<std::string, const Node*>> marked_nodes;
  std::unordered_map<std::string, const Node*> dqd_nodes;

  // Get QDQ NodeUnits first
  QDQ::SelectorManager selector_mgr;
  const auto qdq_selections = selector_mgr.GetQDQSelections(graph_viewer);

  const auto add_node_to_map = [&](const std::vector<NodeIndex>& node_indices, std::string key) {
    int count = 0;
    for (const auto& node_idx : node_indices) {
      auto new_key = key + "_" + std::to_string(count++);
      const auto* node = graph_viewer.GetNode(node_idx);
      dqd_nodes.insert({new_key, node});
    }
  };

  const auto node_in_map = [&](const Node* node) {
    for (auto map_iter: dqd_nodes) {
      if (map_iter.second == node){
        return true;
      }
    }
    return false;
  };


  for (const auto& qdq_selection : qdq_selections) {
    dqd_nodes.clear();
    add_node_to_map(qdq_selection.q_nodes, "q_nodes");
    add_node_to_map(qdq_selection.dq_nodes, "dq_nodes");
    dqd_nodes["key_node"] = graph_viewer.GetNode(qdq_selection.target_node);

    // process q -> dq
    for (auto node_index : qdq_selection.dq_nodes) {
      auto dq_node = graph_viewer.GetNode(node_index);
      auto in_size = dq_node->GetInputEdgesCount();
      auto out_size = dq_node->GetOutputEdgesCount();
      if (in_size == 1 && out_size == 1) {
        auto pre_index = dq_node->InputNodesBegin()->Index();
        auto pre_node = graph_viewer.GetNode(pre_index);
        if (pre_node->OpType() == "QuantizeLinear") {
          const auto& pre_out_name = dq_node->InputDefs()[0]->Name();
          dqd_nodes[pre_out_name] = pre_node;
        }
      }
    }

    for (auto node_index : qdq_selection.q_nodes) {
      auto q_node = graph_viewer.GetNode(node_index);
      auto in_size = q_node->GetInputEdgesCount();
      auto out_size = q_node->GetOutputEdgesCount();
      if (in_size == 1 && out_size == 1) {
        auto out_index = q_node->OutputNodesBegin()->Index();
        auto out_node = graph_viewer.GetNode(out_index);
        if (out_node->OpType() == "DequantizeLinear") {
          const auto& out_name = out_node->InputDefs()[0]->Name();
          if (!node_in_map(out_node))
            dqd_nodes[out_name] = out_node;
        }
      }
    }

    if (!dqd_nodes.empty())
      marked_nodes.push_back(dqd_nodes);
  }

  return marked_nodes;
}
}  // namespace shl_ep
}  // namespace onnxruntime
