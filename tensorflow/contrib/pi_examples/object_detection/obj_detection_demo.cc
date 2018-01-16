/**
 * Author : Chao Yu
 * Date   : 11/01/2018
 * Note   : Object detection demo code.
 */

#include <stdio.h>
#include <setjmp.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <jpeglib.h>
#include "obj_detection.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*
// Error handling for JPEG decoding.
void CatchError(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf* jpeg_jmpbuf = reinterpret_cast<jmp_buf*>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// Decompresses a JPEG file from disk.
bool LoadJpegFile(std::string file_name, std::vector<uint8_t>* data, int* width,
                  int* height, int* channels) {
  struct jpeg_decompress_struct cinfo;
  FILE* infile;
  JSAMPARRAY buffer;
  int row_stride;

  if ((infile = fopen(file_name.c_str(), "rb")) == NULL) {
    std::cout << "Can't open " << file_name << std::endl;
    return false;
  }

  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;  // recovery point in case of error
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    fclose(infile);
    return false;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);
  *width = cinfo.output_width;
  *height = cinfo.output_height;
  *channels = cinfo.output_components;
  data->resize((*height) * (*width) * (*channels));

  row_stride = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                      row_stride, 1);
  while (cinfo.output_scanline < cinfo.output_height) {
    uint8_t* row_address =
        &((*data)[cinfo.output_scanline * row_stride]);
    jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(row_address, buffer[0], row_stride);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);
  return true;
}
*/

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  std::string image(argv[1]);
  // std::string image = "/path/to/images";
  std::string graph =
      "my/obj_detection/ssd_mobilenet_v1_coco_11_06_2017/"
      "frozen_inference_graph.pb";
  std::string labels = "my/obj_detection/labels.txt";
  int32_t input_width = 299;
  int32_t input_height = 299;
  int32_t input_mean = 0;
  int32_t input_std = 255;
  std::string input_layer = "image_tensor:0";
  std::vector<std::string> output_layer = {
      "detection_boxes:0", "detection_scores:0", "detection_classes:0",
      "num_detections:0"};

  // Load image
  std::vector<uint8_t> image_data;
  int image_width = 576;
  int image_height = 768;
  int image_channels;
  std::cout << "image: " << image << std::endl;

  // cv::Mat matImage = cv::imread(image, cv::IMREAD_COLOR);
  // image_width = matImage.size().width;    // matImage.cols
  // image_height = matImage.size().height;  // matImage.rows

  // bool load_file_status = LoadJpegFile(image, &image_data, &image_width,
  //                                 &image_height, &image_channels);
  std::cout << "Loaded JPEG: " << image_width << "x" << image_height << "x"
            << image_channels << std::endl;
  // if (!load_file_status) {
    // std::cerr << "Load image failed!" << std::endl;
    // return -1;
  // }

  auto objectDetection = PI::detection::CreateObjectDetection(
      graph, labels, input_width, input_height, input_mean, input_std,
      input_layer, output_layer);

  /*
  int imgWidth;
  int imgHeight;
  // Show the images
  cv::Mat matImage = cv::imread(image, cv::IMREAD_COLOR);
  imgWidth = matImage.size().width;    // matImage.cols
  imgHeight = matImage.size().height;  // matImage.rows
  // cv::imwrite("/tmp/matImage.jpg", matImage);
  */

  // Get detection results
  auto objects = objectDetection->Detect(image, image_width, image_height);

  for (PI::detection::ObjectDetection::Objects obj : objects) {
     std::cout << obj.classes << ": " << obj.scroes << std::endl;
     std::cout << obj.box_top << ": " << obj.box_left << std::endl;
     std::cout << obj.box_bottom << ": " << obj.box_right << std::endl;
  }

  /*// Config TEXT size etc.
    int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 1;
    int thickness = 1;

    for(size_t i = 0; i < num_detections(0) && i < 20;++i)
    {
      if(scores(i) > 0.5)
      {
        LOG(INFO) << i << ",score:" << scores(i) << ",class:" << classes(i)<<
    ",box:" << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< ","
    << boxes(0,i,3);
        // and its top left corner...
        cv::Point pt1(imgWidth*boxes(0, i, 1), imgHeight*boxes(0, i, 0));
        // and its bottom right corner.
        cv::Point pt2(imgWidth*boxes(0, i, 3), imgHeight*boxes(0, i, 2));
        // These two calls...
        cv::rectangle(matImage, pt1, pt2, cv::Scalar(0, 255, 0), 2);

        cv::putText(matImage, str_labels[(int)(classes(i)) - 1], pt1, fontFace,
    fontScale, cv::Scalar(255, 0, 0), thickness, 4, 0);
      }
    }

    cv::imshow("matImage", matImage);

    cv::waitKey();
  */
  return 0;
}
