#include "PSPNet.h"

PSPNetImpl::PSPNetImpl(int _num_classes, std::string encoder_name /* = "resnet18" */,
	std::string pretrained_path /* = "" */, int _encoder_depth /* = 3 */,
	int psp_out_channels /* = 512 */, bool psp_use_batchnorm /* = true */,
	float psp_dropout /* = 0.2 */, double upsampling /* = 8 */)
{
	num_classes = _num_classes;
	encoder_depth = _encoder_depth;

	auto encoder_param = encoder_params();

	CHECK(encoder_param.contains(encoder_name)) << "encoder name must in {resnet18, resnet34, resnet50, resnet101, resnet150, \
				resnext50_32x4d, resnext101_32x8d, vgg11, vgg11_bn, vgg13, vgg13_bn, \
				vgg16, vgg16_bn, vgg19, vgg19_bn,}";

	std::vector<int> encoder_channels = encoder_param[encoder_name]["out_channels"];
	if (encoder_param[encoder_name]["class_type"] == "resnet")
	{
		//	encoder_ = new ResNetImpl(encoder_param[encoder_name]["layers"], 1000, encoder_name);
		encoder = std::make_shared<ResNetImpl>(encoder_param[encoder_name]["layers"], 1000, encoder_name);
	}
	else if (encoder_param[encoder_name]["class_type"] == "vgg")
	{
		//		encoder_ = new VGGImpl(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
		encoder = std::make_shared<VGGImpl>(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
	}
	else
	{
		LOG(WARNING) << "unknown error in backbone initialization";
	}

	encoder->load_pretrained(pretrained_path);
	decoder = PSPDecoder(encoder_channels, psp_out_channels, psp_dropout, psp_use_batchnorm);
	segmentation_head = SegmentationHead(psp_out_channels, num_classes, 3, upsampling);

	register_module("encoder", encoder);
	register_module("decoder", decoder);
	register_module("segmentation_haed", segmentation_head);
}

torch::Tensor PSPNetImpl::forward(torch::Tensor x)
{
	std::vector<torch::Tensor> features = encoder->features(x, encoder_depth);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}