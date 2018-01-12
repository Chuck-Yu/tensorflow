/**
  * Author : Chao Yu
  * Date   : 10/01/2018
  * Note   : Object detection based on TensforFlow.
  */

#include <fstream>
#include <utility>
#include "obj_detection.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
// using tensorflow::Flag;
// using tensorflow::Tensor;
// using tensorflow::Status;
// using tensorflow::string;
// using tensorflow::int32;

namespace PI {
namespace detection {
  class ObjectDetection::Impl {
    public:
      // Impl Start
      Impl(const Parameters & params);
      ~Impl();

      std::vector<std::pair<std::string, float>> Detect(
        const std::string& file_name);

    private:
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

      std::unique_ptr<tensorflow::Session> session;
      std::vector<std::string> str_labels;

      class ObjectDetection;
      tensorflow::Status LoadGraph(std::string graph_file_name,
        std::unique_ptr<tensorflow::Session> *session);
      tensorflow::Status ReadLabelsFile(
        const std::string& file_name, std::vector<std::string>* result);
      tensorflow::Status ReadTensorFromImageFile(
        const std::string& file_name, const int input_height,
        const int input_width, const float input_mean,
        const float input_std,
        std::vector<tensorflow::Tensor>* out_tensors);
      tensorflow::Status ReadEntireFile(
        tensorflow::Env* env, const std::string& filename,
        tensorflow::Tensor* output);

  };

  ObjectDetection::Impl::Impl(const Parameters &params)
    : graph(params.graph),
      labels(params.labels),
      input_width(params.input_width),
      input_height(params.input_height),
      input_mean(params.input_mean),
      input_std(params.input_std),
      input_layer(params.input_layer),
      output_layer(params.output_layer) {
    std::string graph_path = tensorflow::io::JoinPath("", graph);
    tensorflow::Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
    }

    // Read labels
    tensorflow::Status read_labels_status =
    ReadLabelsFile(labels, &str_labels);
    if (!read_labels_status.ok()) {
      LOG(ERROR) << read_labels_status;
    }
  }

  ObjectDetection::Impl::~Impl() {}

  // Takes a file name, and loads a list of labels from it, one per line, and
  // returns a vector of the strings. It pads with empty strings so the length
  // of the result is a multiple of 16, because our model expects that.
  tensorflow::Status ObjectDetection::Impl::ReadLabelsFile(
    const std::string& file_name, std::vector<std::string>* result) {
    std::ifstream file(file_name);
    if (!file) {
      return tensorflow::errors::NotFound("Labels file ", file_name,
                                          " not found.");
    }
    result->clear();
    std::string line;
    while (std::getline(file, line)) {
      result->push_back(line);
    }
    // *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
      result->emplace_back();
    }
    return tensorflow::Status::OK();
  }

  tensorflow::Status ObjectDetection::Impl::ReadEntireFile(tensorflow::Env* env, const std::string& filename,
                               tensorflow::Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    std::string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
      return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                          "' expected ", file_size, " got ",
                                          data.size());
    }
    output->scalar<std::string>()() = data.ToString();
    return tensorflow::Status::OK();
  }

  // Given an image file name, read in the data, try to decode it as an image,
  // resize it to the requested size, and then scale the values as desired.
  tensorflow::Status ObjectDetection::Impl::ReadTensorFromImageFile(const std::string& file_name, const int input_height,
                                 const int input_width, const float input_mean,
                                 const float input_std,
                                 std::vector<tensorflow::Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    std::string input_name = "file_reader";
    std::string output_name = "normalized";

    // read file_name into a tensor named input
    tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
        ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
        Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::StringPiece(file_name).ends_with(".png")) {
      image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                               DecodePng::Channels(wanted_channels));
    } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
      // gif decoder returns 4-D tensor, remove the first dim
      image_reader =
          Squeeze(root.WithOpName("squeeze_first_dim"),
                  DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else {
      // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
      image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                DecodeJpeg::Channels(wanted_channels));
    }

    auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
    return tensorflow::Status::OK();
  }

  // Reads a model graph definition from disk, and creates a session object you
  // can use to run it.
  tensorflow::Status ObjectDetection::Impl::LoadGraph(std::string graph_file_name,
                   std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                          graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
      return session_create_status;
    }
    return tensorflow::Status::OK();
  }



  std::vector<std::pair<std::string, float>> ObjectDetection::Impl::Detect(
      const std::string& file_name) {
    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
    std::vector<std::pair<std::string, float>> det_result;

    std::vector<tensorflow::Tensor> resized_tensors;
    std::string image_path = tensorflow::io::JoinPath("", file_name);
    tensorflow::Status read_tensor_status =
        ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                                input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
      LOG(ERROR) << read_tensor_status;
      return det_result;
    }
    const tensorflow::Tensor& resized_tensor = resized_tensors[0];

    LOG(INFO) <<"image shape:" << resized_tensor.shape().DebugString()<< ",len:" << resized_tensors.size() << ",tensor type:"<< resized_tensor.dtype();
    // << ",data:" << resized_tensor.flat<tensorflow::uint8>();
    // Actually run the image through the model.
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session->Run({{input_layer, resized_tensor}},
                                     output_layer, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return det_result;
    }

    // LOG(INFO) << "size:" << outputs.size() << ",image_width:" << imgWidth << ",image_height:" << imgHeight << endl;

    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();

    LOG(INFO) << "num_detections:" << num_detections(0) << "," << outputs[0].shape().DebugString();

    for(size_t i = 0; i < num_detections(0) && i < 20;++i)
    {
      if(scores(i) > 0.5)
      {
        LOG(INFO) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
        // det_result.push_back(std::make_pair(str_labels[(int)(classes(i)) - 1], scores(i), boxes(0,i,0), boxes(0,i,1), boxes(0,i,2), boxes(0,i,3)));
        det_result.push_back(std::make_pair(str_labels[(int)(classes(i)) - 1], scores(i)));
      }
    }

    return det_result;
  }

  // Parameters start
  ObjectDetection::Parameters::Parameters(
      const std::string &graph, const std::string &labels,
      const int32_t input_width, const int32_t input_height,
      const int32_t input_mean, const int32_t input_std,
      const std::string &input_layer, const std::vector<std::string> &output_layer)
      : graph(graph),
        labels(labels),
        input_width(input_width),
        input_height(input_height),
        input_mean(input_mean),
        input_std(input_std),
        input_layer(input_layer),
        output_layer(output_layer) {}

  ObjectDetection::Parameters::~Parameters() {}
  // Parameters end

  // ObjectDetection start
  ObjectDetection::ObjectDetection(const Parameters &params)
      : impl_(std::make_shared<Impl>(params)) {}
  ObjectDetection::~ObjectDetection() {}

  std::vector<std::pair<std::string, float>> ObjectDetection::Detect(
      const std::string& file_name) {
    return impl_->Detect(file_name);
  }
  // ObjectDetection end

  std::shared_ptr<ObjectDetection> CreateObjectDetection(
      const std::string &graph, const std::string &labels,
      const int32_t input_width, const int32_t input_height,
      const int32_t input_mean, const int32_t input_std,
      const std::string &input_layer /*= "Mul"*/,
      const std::vector<std::string> &output_layer /*= "softmax"*/) {
    ObjectDetection::Parameters parameters(graph, labels, input_width,
                                          input_height, input_mean, input_std,
                                          input_layer, output_layer);
    return std::make_shared<ObjectDetection>(parameters);
  }

} // detection
} // PI
