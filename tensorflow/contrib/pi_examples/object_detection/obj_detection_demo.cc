/**
  * Author : Chao Yu
  * Date   : 11/01/2018
  * Note   : Object detection demo code.
  */

#include "obj_detection.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[]){

  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  std::string image(argv[1]);
  // std::string image = "/path/to/images";
  std::string graph ="my/obj_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb";
  std::string labels ="my/obj_detection/labels.txt";
  int32_t input_width = 299;
  int32_t input_height = 299;
  int32_t input_mean = 0;
  int32_t input_std = 255;
  std::string input_layer = "image_tensor:0";
  std::vector<std::string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };

  auto objectDetection = PI::detection::CreateObjectDetection(
    graph,
    labels,
    input_width,
    input_height,
    input_mean,
    input_std,
    input_layer,
    output_layer
  );

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
  auto objects = objectDetection->Detect(image);

  for (std::pair<std::string, float> obj : objects) {
    std::cout << obj.first << ": " << obj.second << std::endl;
    // LOG(INFO) << obj.second << ": " << obj.third;
    // LOG(INFO) << "Box: " << obj.fourth << "," << obj.fifth << "," << obj.sixth << "," << obj.seventh;
  }

/*// Config TEXT size etc.
  int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
  double fontScale = 1;
  int thickness = 1;

  for(size_t i = 0; i < num_detections(0) && i < 20;++i)
  {
    if(scores(i) > 0.5)
    {
      LOG(INFO) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
      // and its top left corner...
      cv::Point pt1(imgWidth*boxes(0, i, 1), imgHeight*boxes(0, i, 0));
      // and its bottom right corner.
      cv::Point pt2(imgWidth*boxes(0, i, 3), imgHeight*boxes(0, i, 2));
      // These two calls...
      cv::rectangle(matImage, pt1, pt2, cv::Scalar(0, 255, 0), 2);

      cv::putText(matImage, str_labels[(int)(classes(i)) - 1], pt1, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness, 4, 0);
    }
  }

  cv::imshow("matImage", matImage);

  cv::waitKey();
*/
  return 0;
}
