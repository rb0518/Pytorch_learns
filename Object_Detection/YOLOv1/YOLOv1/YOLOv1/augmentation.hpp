#pragma once
#include <tuple>
#include <vector>
#include <random>
#include <memory>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
// For Original Header
#include "transforms.hpp"


// ----------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl)
// ----------------------------------------------------
#define YOLOAugmentation std::make_shared<YOLOAugmentationImpl>
class YOLOAugmentationImpl : public transforms::ComposeImpl {
private:
	double jitter;
	double flip_rate, scale_rate, blur_rate, brightness_rate, hue_rate, saturation_rate, shift_rate, crop_rate;
	std::vector<std::mt19937> mt;
	void deepcopy(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_flip(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_scale(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_blur(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_brightness(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_hue(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_saturation(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_shift(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
	void random_crop(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2);
public:
	YOLOAugmentationImpl(const double jitter_ = 0.2, const double flip_rate_ = 0.5, const double scale_rate_ = 0.5, const double blur_rate_ = 0.5, const double brightness_rate_ = 0.5, const double hue_rate_ = 0.5, const double saturation_rate_ = 0.5, const double shift_rate_ = 0.5, const double crop_rate_ = 0.5);
	bool type() override { return CV_MAT; }
	void forward(cv::Mat& data_in1, std::tuple<torch::Tensor, torch::Tensor>& data_in2, cv::Mat& data_out1, std::tuple<torch::Tensor, torch::Tensor>& data_out2) override;
};

