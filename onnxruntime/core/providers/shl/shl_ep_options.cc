// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
#include <regex>

#include "core/common/common.h"
#include "core/common/cpuid_info.h"
#include "core/framework/provider_options_utils.h"

#include "shl_ep_options.h"


namespace onnxruntime {
namespace shl_ep {

std::string ShlEPOptionsHelper::whitespace_trimming(const std::string& str) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  size_t start = str.find_first_not_of(WHITESPACE);
  if (start == std::string::npos) {
    return "";
  } else {
    size_t end = str.find_last_not_of(WHITESPACE);
    ORT_ENFORCE(end != std::string::npos);
    return str.substr(start, end + 1);
  }
}

ProviderOptions ShlEPOptionsHelper::FromOptionsString(const char* opt_str) {
  std::string settings{opt_str};
  ProviderOptions options;
  if (!settings.empty()) {
    const std::string& str = settings;

    // tokenize settings
    std::regex reg("\\s*,\\s*");
    std::sregex_token_iterator iter(str.begin(), str.end(), reg, -1);
    std::sregex_token_iterator iter_end;
    std::vector<std::string> pairs(iter, iter_end);

    ORT_ENFORCE(pairs.size() > 0);

    for(const auto& pair : pairs) {
      auto pos_colon = pair.find(':');
      ORT_ENFORCE(pos_colon != std::string::npos, "Invalid key value pair.");
      std::string key = pair.substr(0, pos_colon);
      std::string value = pair.substr(pos_colon + 1);

      // trim leading and trailing spaces from key/value
      key = whitespace_trimming(key);
      value = whitespace_trimming(value);
      options[key] = value;
    }
  }

  return options;
}


}  // namespace shl
}  // namespace onnxruntime
