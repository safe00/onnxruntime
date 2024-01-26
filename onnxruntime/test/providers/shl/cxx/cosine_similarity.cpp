#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <cassert>

#define ASSERT_WITH_STREAM(expr) \
  (!(expr)) &&                   \
      AssertWithStream(__FILE__, __LINE__).GetStream()

class AssertWithStream {
 public:
  AssertWithStream(const char* file, int line) {
    oss << "Assertion failed: "
        << "In file: " << file << ", Line: " << line << "\n";
  }

  std::ostringstream& GetStream() {
    return oss;
  }

  ~AssertWithStream() {
    std::cerr << oss.str();
    assert(false);
  }

 private:
  std::ostringstream oss;
};

std::vector<float> readFloatsFromBinaryFile(const std::string& filepath) {
  std::vector<float> buffer;
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filepath << std::endl;
    return buffer;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (size % sizeof(float) != 0) {
    std::cerr << "File size is not a multiple of float size: " << filepath << std::endl;
    return buffer;
  }

  buffer.resize(size / sizeof(float));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    std::cerr << "Failed to read float data from file: " << filepath << std::endl;
  }

  return buffer;
}

float calculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
  if (vec1.size() != vec2.size()) {
    std::cerr << "Vectors are of different lengths." << std::endl;
    return 0.0f;
  }

  double dotProduct = 0.0f;
  double normA = 0.0f;
  double normB = 0.0f;

  for (size_t i = 0; i < vec1.size(); ++i) {
    dotProduct += vec1[i] * vec2[i];
    normA += vec1[i] * vec1[i];
    normB += vec2[i] * vec2[i];
  }

  if (normA == 0.0f || normB == 0.0f) {
    std::cerr << "One or both vectors are zero vectors." << std::endl;
    return 0.0f;
  }

  return dotProduct / (sqrt(normA) * sqrt(normB));
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <file1.bin> <file2.bin> threshold" << std::endl;
    return 1;
  }

  std::string file1Path = argv[1];
  std::string file2Path = argv[2];
  float threshold = std::stof(argv[3]);

  std::vector<float> data1 = readFloatsFromBinaryFile(file1Path);
  std::vector<float> data2 = readFloatsFromBinaryFile(file2Path);

  if (data1.empty() || data2.empty()) {
    std::cerr << "Failed to read data from one or both files." << std::endl;
    return 1;
  }

  float cosineSimilarity = calculateCosineSimilarity(data1, data2);

  ASSERT_WITH_STREAM(cosineSimilarity >= threshold) << "current threshold is " << std::to_string(cosineSimilarity) << std::endl;
  ;

  return 0;
}
