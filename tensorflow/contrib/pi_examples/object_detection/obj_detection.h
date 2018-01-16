/**
 * Author : Chao Yu
 * Date   : 11/01/2018
 * Note   : Object detection header file.
 */

#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace PI {
namespace detection {

class ObjectDetection {
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
    std::vector<std::string> output_layer;

    Parameters(const std::string& graph, const std::string& labels,
               const int32_t input_width, const int32_t input_height,
               const int32_t input_mean, const int32_t input_std,
               const std::string& input_layer,
               const std::vector<std::string>& output_layer);
    virtual ~Parameters();
  };

  ObjectDetection(const Parameters& params);
  virtual ~ObjectDetection();

  struct Objects {
    std::string classes;
    float scroes;
    float box_top;
    float box_left;
    float box_bottom;
    float box_right;
  };

  std::vector<Objects> Detect(const std::string& file_name,
                              const int image_width, const int image_height);

 private:
  // PIMPL
  class Impl;
  std::shared_ptr<Impl> impl_;
};

std::shared_ptr<ObjectDetection> CreateObjectDetection(
    const std::string& graph, const std::string& labels,
    const int32_t input_width, const int32_t input_height,
    const int32_t input_mean, const int32_t input_std,
    const std::string& input_layer = "image_tensor:0",
    const std::vector<std::string>& output_layer = {
        "detection_boxes:0", "detection_scores:0", "detection_classes:0",
        "num_detections:0"});
}  // namespace detection
}  // namespace PI
