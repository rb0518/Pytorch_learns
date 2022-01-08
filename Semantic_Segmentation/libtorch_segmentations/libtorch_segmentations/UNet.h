#pragma once
#include "ResNet.h"
#include "VGG.h"
#include "UNetDecoder.h"

class UNetImpl : public torch::nn::Module
{
public:
	UNetImpl() {};
	~UNetImpl() {
	};

	UNetImpl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "",
		int encoder_depth = 5, std::vector<int> decoder_channels = { 256, 128, 64, 32, 16 }, 
		bool use_attention = false);
	torch::Tensor forward(torch::Tensor x);

private:
	std::shared_ptr<Backbone> encoder_;
	UNetDecoder decoder_{ nullptr };
	SegmentationHead segmentation_head_{ nullptr };
	int num_classes_ = 1;
	std::vector<int> basic_channels_ = { 3, 64, 64, 128, 256, 512 };
	std::vector<int> bottle_channels_ = { 3, 64, 256, 512, 1024, 2048 };
	std::map<std::string, std::vector<int>> name2layers_ = getParams();
};

TORCH_MODULE(UNet);

