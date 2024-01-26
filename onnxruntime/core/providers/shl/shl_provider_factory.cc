// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shl/shl_provider_factory.h"
#include "shl_ep_options.h"
#include "shl_execution_provider.h"
#include "shl_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct ShlProviderFactory : public IExecutionProviderFactory {
  ShlProviderFactory(const ProviderOptions& config) : config_(config) {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  ProviderOptions config_;
};

std::unique_ptr<IExecutionProvider> ShlProviderFactory ::CreateProvider() {
  return std::make_unique<ShlExecutionProvider>(config_);
};

std::shared_ptr<IExecutionProviderFactory> ShlProviderFactoryCreator::Create(const ProviderOptions& config) {
  return std::make_shared<ShlProviderFactory>(config);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Shl, _In_ OrtSessionOptions* options, _In_ const char* opt_str) {
  onnxruntime::ProviderOptions shl_options = onnxruntime::shl_ep::ShlEPOptionsHelper::FromOptionsString(opt_str);
  options->provider_factories.push_back(onnxruntime::ShlProviderFactoryCreator::Create(shl_options));
  return nullptr;
}
