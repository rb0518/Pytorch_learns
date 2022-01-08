#include "UNet.h"
#include <glog/logging.h>

UNetImpl::UNetImpl(int num_classes, std::string encoder_name /* = "resnet18" */, std::string pretrained_path /* = "" */,
	int encoder_depth /* = 5 */, std::vector<int> decoder_channels /* =  */, bool use_attention /* = false */) 
{
	num_classes_ = num_classes;

	auto encoder_param = encoder_params();

	CHECK(encoder_param.contains(encoder_name)) << "encoder name must in {resnet18, resnet34, resnet50, resnet101, resnet150, \
				resnext50_32x4d, resnext101_32x8d, vgg11, vgg11_bn, vgg13, vgg13_bn, \
				vgg16, vgg16_bn, vgg19, vgg19_bn,}";

	std::vector<int> encoder_channels = encoder_param[encoder_name]["out_channels"];
	if (encoder_param[encoder_name]["class_type"] == "resnet")
	{
	//	encoder_ = new ResNetImpl(encoder_param[encoder_name]["layers"], 1000, encoder_name);
		encoder_ = std::make_shared<ResNetImpl>(encoder_param[encoder_name]["layers"], 1000, encoder_name);
	}
	else if (encoder_param[encoder_name]["class_type"] == "vgg")
	{
//		encoder_ = new VGGImpl(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
		encoder_ = std::make_shared<VGGImpl>(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
	}
	else
	{
		LOG(WARNING) << "unknown error in backbone initialization";
	}

	encoder_->load_pretrained(pretrained_path);
	decoder_ = UNetDecoder(encoder_channels, decoder_channels, encoder_depth, use_attention, /*use_center = */ false);
	segmentation_head_ = SegmentationHead(decoder_channels[decoder_channels.size() - 1], num_classes, 1, 1);

	register_module("encoder", encoder_/*std::shared_ptr<Backbone>(encoder_)*/);
	register_module("decoder", decoder_);
	register_module("segmentation_head", segmentation_head_);
}

torch::Tensor UNetImpl::forward(torch::Tensor x)
{
	std::vector<torch::Tensor> features = encoder_->features(x);
	x = decoder_->forward(features);
	x = segmentation_head_->forward(x);
	return x;
}