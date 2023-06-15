// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace shl_ep {

/**
@Class LayerNormFusion

Rewrite graph fusing Layer Normalization subgraph to a single LayerNormalization node.

The formula corresponding to LayerNorm activation subgraph:
(x - mean(x, axis)) / sqrt(var(x, axis)) * scale + bias, where x is the input.

*/
std::vector<std::unordered_map<std::string, const Node*>> LayerNormMarker(const onnxruntime::GraphViewer& graph_viewer);
std::vector<std::unordered_map<std::string, const Node*>> QDQMarker(const onnxruntime::GraphViewer& graph_viewer);


}  // namespace shl_ep
}  // namespace onnxruntime
