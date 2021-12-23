#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>


class VOCSegDataset : public torch::data::Dataset<VOCSegDataset>
{
public:
	explicit VOCSegDataset(std::string data_root, std::string run_mode, int64_t num_class);

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;
private:
	void createColormap();
	void Resize(cv::Mat& src, cv::Mat& dst, int width, int height, float probability);
	void HorizontalFlip(cv::Mat& src, cv::Mat& dst, float probability);
	void VerticalFlip(cv::Mat& src, cv::Mat& dst, float probability);
	void RandomScaleRotate(cv::Mat& src, cv::Mat& dst, float probability, 
		float rotate_limit, float scale_limit, int interpolation, int boder_mode);
private:
	std::string data_root_;
	std::string run_mode_;
	std::string list_file_;
	int64_t num_class_;

	std::vector<std::string> image_files_;
	std::vector<std::string> label_files_;
	std::map<int, int> colormap_;
};


