
#include "../utils.h"
#include <unistd.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <cstdlib>

char* MODEL_PATH = getenv("MODEL_PATH");
std::string model_path{MODEL_PATH};

std::string get_current_working_dir() {
  char buff[PATH_MAX];
  getcwd(buff, PATH_MAX);
  std::string current_working_dir(buff);
  return current_working_dir;
}

std::string join_path(std::string& dir, std::string& path) {
  std::string out{""};
  if (dir.empty()) {
    out += "./" + path;
  } else if (dir.back() == '/') {
    out += dir + path;
  } else {
    out += dir + "/" + path;
    ;
  }
  return out;
}

void model_test(bool use_shl) {
  std::string model_name{"mobilenetv2-12-sim-qdq.onnx"};
  model_path = join_path(model_path, model_name);
  std::string curt_path = get_current_working_dir();
  const char* input_names[] = {"input"};
  const int64_t input_shape[] = {1, 3, 224, 224};

  int in_length = 1 * 3 * 224 * 224;
  float* input_data = (float*)malloc(4 * in_length);
  // create array
  for (int i = 0; i < in_length; i++) {
    input_data[i] = (float)rand() / (float)RAND_MAX;  // 0~1
  }

  std::vector<std::vector<float>> shl_out = test_model(model_path.c_str(), input_names, input_data, input_shape, use_shl, "mv2");

  if (use_shl) {
    for (size_t i = 0; i < shl_out.size(); i++) {
      std::string out_file_path = curt_path + "/shl_mv2_out" + std::to_string(i) + ".bin";
      save_float_array(shl_out[i].data(), shl_out[i].size(), out_file_path.c_str());
    }
  } else {
    for (size_t i = 0; i < shl_out.size(); i++) {
      std::string out_file_path = curt_path + "/cpu_mv2_out" + std::to_string(i) + ".bin";
      save_float_array(shl_out[i].data(), shl_out[i].size(), out_file_path.c_str());
    }
  }
}

int main(int argc, char* argv[]) {
  srand(0);  // set random seed
  bool use_shl = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--use_shl") {
      use_shl = true;
      break;
    }
  }
  model_test(use_shl);

  return 0;
}
