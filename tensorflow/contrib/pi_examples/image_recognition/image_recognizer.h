//
// Created by kerner on 11/3/17.
//

#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace PI {
namespace recognition {

class ImageRecognizer {
 public:
  struct Parameters {
    // graph to be executed
    std::string graph;
    // name of file containing labels
    std::string labels;
    // resize image to this width in pixels
    int32_t input_width;
    // resize image to this height in pixels
    int32_t input_height;
    // scale pixel values to this mean
    int32_t input_mean;
    // scale pixel values to this std deviation
    int32_t input_std;
    // name of input layer, default is Mul
    std::string input_layer;
    // name of output layer, default is softmax
    std::string output_layer;

    Parameters(const std::string& graph, const std::string& labels,
               const int32_t input_width, const int32_t input_height,
               const int32_t input_mean, const int32_t input_std,
               const std::string& input_layer, const std::string& output_layer);
    virtual ~Parameters();
  };

  ImageRecognizer(const Parameters& params);
  virtual ~ImageRecognizer();

  std::vector<std::pair<std::string, float>> Recognize(
      uint8_t* image_data, const int image_width, const int image_height,
      const int image_channels);

 private:
  // PIMPL
  class Impl;
  std::shared_ptr<Impl> impl_;
};

std::shared_ptr<ImageRecognizer> CreateImageRecognizer(
    const std::string& graph, const std::string& labels,
    const int32_t input_width, const int32_t input_height,
    const int32_t input_mean, const int32_t input_std,
    const std::string& input_layer = "Mul",
    const std::string& output_layer = "softmax");
}  // namespace recognition
}  // namespace PI
