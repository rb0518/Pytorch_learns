#pragma once
#include <torch/torch.h>
#include "InterFace.h"

class VGGImpl : public Backbone
{
public:
	VGGImpl(std::vector<int> cfg, int num_classes = 1000, bool batch_norm = false);
	torch::Tensor forward(torch::Tensor x);

	std::vector<torch::Tensor> features(torch::Tensor x, int encoder_depth = 5) override;
	torch::Tensor features_at(torch::Tensor x, int stage_num) override;
	void make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list) override;
	void load_pretrained(std::string pretrained_path) override;

private:
	torch::nn::Sequential features_{ nullptr };
	torch::nn::AdaptiveAvgPool2d avgpool{ nullptr };
	torch::nn::Sequential classifier;
	std::vector<int> cfg = {};
	bool batch_norm;
};


TORCH_MODULE(VGG);
