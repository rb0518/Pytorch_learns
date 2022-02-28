#include "UNet.h"
#include <glog/logging.h>

UNetImpl::UNetImpl(int num_classes, std::string encoder_name /* = "resnet18" */, std::string pretrained_path /* = "" */,
	int encoder_depth /* = 5 */, std::vector<int> decoder_channels /* = { 256, 128, 64, 32, 16 } */, bool use_attention /* = false */) 
{
	// -- 2022-1-18 Refer to the python UNET modeland rewrite the C + +code ---
#if 1
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
	decoder_ = UNetsModule::UNetDecoder(encoder_channels, decoder_channels, encoder_depth, use_attention, /*use_center = */ false);
	segmentation_head_ = SegmentationHead(decoder_channels[decoder_channels.size() - 1], num_classes, 1, 1);

	register_module("encoder", encoder_/*std::shared_ptr<Backbone>(encoder_)*/);
	register_module("decoder", decoder_);
	register_module("segmentation_head", segmentation_head_);
#else
	inconv = InConv(3, 64);

	down1 = DownScale(64, 128);
	down2 = DownScale(128, 256);
	down3 = DownScale(256, 512);
	down4 = DownScale(512, 512);

	up1 = UpScale(1024, 256, false);
	up2 = UpScale(512, 128, false);
	up3 = UpScale(256, 64, false);
	up4 = UpScale(128, 64, false);

	outconv = OutConv(64, num_classes);

	register_module("inconv", inconv);
	register_module("down1", down1);
	register_module("down2", down2);
	register_module("down3", down3);
	register_module("down4", down4);

	register_module("up1", up1);
	register_module("up2", up2);
	register_module("up3", up3);
	register_module("up4", up4);

	register_module("outconv", outconv);
#endif
}

torch::Tensor UNetImpl::forward(torch::Tensor x)
{
#if 1
	std::vector<torch::Tensor> features = encoder_->features(x);
	x = decoder_->forward(features);
	x = segmentation_head_->forward(x);
	return x;
#else
	auto x1 = inconv->forward(x);

	auto x2 = down1->forward(x1);
	auto x3 = down2->forward(x2);
	auto x4 = down3->forward(x3);
	auto x5 = down4->forward(x4);

	x = up1->forward(x5, x4);
	x = up2->forward(x, x3);
	x = up3->forward(x, x2);
	x = up4->forward(x, x1);

	x = outconv(x);

	return x;
#endif
}