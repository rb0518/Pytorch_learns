#pragma once

#include "PSPNetDecoder.h"
#include "ResNet.h"
#include "VGG.h"

class PSPNetImpl : public torch::nn::Module
{
public:
	PSPNetImpl() {};
	~PSPNetImpl() {};
	PSPNetImpl(int _num_classes, std::string encoder_name = "resnet18",
		std::string pretrained_path = "", int _encoder_depth = 3,
		int psp_out_channels = 512, bool psp_use_batchnorm = true,
		float psp_dropout = 0.2, double upsampling = 8);
	torch::Tensor forward(torch::Tensor x);
private:
	std::shared_ptr<Backbone> encoder;
	PSPDecoder decoder{ nullptr };
	SegmentationHead segmentation_head{ nullptr };
	int num_classes = 1;
	int encoder_depth = 3;
	std::vector<int> BasicChannels = { 3, 64, 64, 128, 256, 512 };
	std::vector<int> BottleChannels = { 3, 64, 256, 512, 1024, 2048 };
	std::map < std::string, std::vector<int>> name2layers = getParams();
};
TORCH_MODULE(PSPNet);

