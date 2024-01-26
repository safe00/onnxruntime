// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef SHL_EXECUTION_PROVIDER_OPTIONS_H
#define SHL_EXECUTION_PROVIDER_OPTIONS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"



namespace onnxruntime {

namespace shl_ep {

class ShlEPOptionsHelper {
public:
  static std::string whitespace_trimming(const std::string& str);
  static ProviderOptions FromOptionsString(const char* options);
};

}  // namespace shl
}  // namespace onnxruntime

#endif  // SHL_EXECUTION_PROVIDER_OPTIONS_H
