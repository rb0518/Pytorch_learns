#pragma once

#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <map>

struct trainTricks {
	unsigned int freeze_epochs = 0;
	std::vector<unsigned int> decay_epochs = { 0 };
	float dice_ce_ratio = 0.5;

	float horizontal_flip_prob = 0;
	float vertical_flip_prob = 0;
	float scale_rotate_prob = 0;

	float scale_limit = 0.1;
	float rotate_limit = 45;
	int interpolation = cv::INTER_LINEAR;
	int border_mode = cv::BORDER_CONSTANT;
};


class SegDataset : public torch::data::Dataset<SegDataset>
{
public:
	SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
		std::vector<std::string> list_labels, std::vector<std::string> name_list, 
		trainTricks trichs, bool isTrain = false);

	torch::optional<size_t> size() const override {
		return list_labels_.size();
	};

	torch::data::Example<> get(size_t index) override;
private:
	void draw_mask(std::string json_path, cv::Mat& mask);
	std::vector<std::string> list_images_;
	std::vector<std::string> list_labels_;
	std::vector<std::string> list_names_;
	std::map<std::string, int> name2index_;
	std::map<std::string, cv::Scalar> name2color_;
	int resize_width_;
	int resize_height_;
	bool is_train_;
	trainTricks tricks_;
};

