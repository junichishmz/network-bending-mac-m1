#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

torch::Tensor erode(torch::Tensor image, int64_t dilation_size) {
  image = image.to(torch::kCPU);
  cv::Mat image_mat(image.size(0),
                    image.size(1),
                    CV_32FC1,
                    image.data_ptr<float>());
  
  int dilation_type = cv::MORPH_ELLIPSE;
  cv::Mat output_mat;
  cv::Mat element = cv::getStructuringElement( dilation_type,
                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       cv::Point( dilation_size, dilation_size ) );
  cv::erode( image_mat, output_mat, element );

  torch::Tensor output =
    torch::from_blob(output_mat.ptr<float>(),{image.size(0), image.size(1)});
  return output.clone().to(torch::kCPU);
}

static auto registry =
  torch::RegisterOperators("my_ops::erode", &erode);