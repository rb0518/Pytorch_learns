#include "UNetDecoder.h"
#include "util.h"
namespace UNetsModule {

SCSEModuleImpl::SCSEModuleImpl(int in_channels, int reduction /* = 16 */, bool use_attention /* = false */)
{
	use_attention_ = use_attention;
	cSE = torch::nn::Sequential(
		torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
		torch::nn::Conv2d(conv_options(in_channels, in_channels / reduction, 1)),
		torch::nn::ReLU(torch::nn::ReLUOptions(true)),
		torch::nn::Conv2d(conv_options(in_channels/reduction, in_channels, 1)),
		torch::nn::Sigmoid()
	);
	sSE = torch::nn::Sequential(torch::nn::Conv2d(conv_options(in_channels, 1, 1)), torch::nn::Sigmoid());
	register_module("cSE", cSE);
	register_module("sSE", sSE);
}

torch::Tensor SCSEModuleImpl::forward(torch::Tensor x)
{
	if (use_attention_ == false) 
	{
		return x;
	}

	return x * cSE->forward(x) + x * sSE->forward(x);
}

Conv2dReLUImpl::Conv2dReLUImpl(int in_channels, int out_channels, int kernel_size /* = 3 */, int padding /* = 1 */)
{
	conv2d = torch::nn::Conv2d(conv_options(in_channels, out_channels, kernel_size, 1, padding));
	bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
	register_module("conv2d", conv2d);
	register_module("bn", bn);
}

torch::Tensor Conv2dReLUImpl::forward(torch::Tensor x)
{
//	std::cout << "Conv2dReLU forward: x " << x.sizes() << std::endl;
	x = conv2d->forward(x);
	x = bn->forward(x);

	return x;
}

DecoderBlockImpl::DecoderBlockImpl(int in_channels, int skip_channels, int out_channels, bool skip, bool attention) {
	conv1 = UNetsModule::Conv2dReLU(in_channels /*+ skip_channels*/, out_channels, 3, 1);
	conv2 = UNetsModule::Conv2dReLU(out_channels, out_channels, 3, 1);
	register_module("conv1", conv1);
	register_module("conv2", conv2);
	upsample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2,2 })).mode(torch::kNearest));

	attention1 = SCSEModule(in_channels /*+ skip_channels*/, 16, attention);
	attention2 = SCSEModule(out_channels, 16, attention);
	register_module("attention1", attention1);
	register_module("attention2", attention2);
	is_skip = skip;
}

torch::Tensor DecoderBlockImpl::forward(torch::Tensor x, torch::Tensor skip) {
	x = upsample->forward(x);
	if (is_skip)
	{
		std::cout << "is_skip: x: " << x.sizes() << "  skip:" << skip.sizes() << std::endl;
		x = torch::cat({ x, skip }, 1);
		x = attention1->forward(x);
	}
	x = conv1->forward(x);
	x = conv2->forward(x);
	x = attention2->forward(x);
	return x;
}


torch::nn::Sequential CenterBlock(int in_channels, int out_channels)
{
	return torch::nn::Sequential(UNetsModule::Conv2dReLU(in_channels, out_channels, 3, 1),
		UNetsModule::Conv2dReLU(out_channels, out_channels, 3, 1));
}

UNetDecoderImpl::UNetDecoderImpl(std::vector<int> encoder_channels, std::vector<int> decoder_channels,
	int n_blocks /* = 5 */, bool use_attention /* = false */, bool use_center /* = false */)
{
#if 1
	CHECK(n_blocks == decoder_channels.size()) << "Model depth not equal to decoder_channels";

	// 1 reverse encoder_channels 
//	std::cout << "1: " << decoder_channels << std::endl;
	std::reverse(std::begin(encoder_channels), std::end(encoder_channels));
//	std::cout << "2 reverse : " << encoder_channels << std::endl;

	int head_channels = encoder_channels[0];
	std::vector<int> out_channels = decoder_channels;
//	std::cout << "2: " << out_channels << "   h: " <<  head_channels << std::endl;

	decoder_channels.pop_back();

//	std::cout << "3: " << out_channels << "   h: " << head_channels << std::endl;
	decoder_channels.insert(decoder_channels.begin(), head_channels);
//	std::cout << "4: " << out_channels << "   h: " << head_channels << std::endl;

	std::vector<int> in_channels = decoder_channels;
	encoder_channels.erase(encoder_channels.begin());
//	std::cout << "5 erase : " << encoder_channels << std::endl;

	std::vector<int> skip_channels = encoder_channels;
	skip_channels[skip_channels.size() - 1] = 0;
//	std::cout << "6 erase : " << skip_channels << std::endl;

	if (use_center)
	{
		center = CenterBlock(head_channels, head_channels);
	}
	else
	{	// torch::nn::Identity 直接返回输入, 用来保持程序模块结构一致
		center = torch::nn::Sequential(torch::nn::Identity());
	}

	for (int i = 0; i < in_channels.size() - 1; i++)
	{
		blocks->push_back(DecoderBlock(in_channels[i], skip_channels[i], out_channels[i], false, use_attention));
	}
	blocks->push_back(DecoderBlock(in_channels[in_channels.size() - 1], skip_channels[in_channels.size() - 1],
		out_channels[in_channels.size() - 1], false, use_attention));

	register_module("center", center);
	register_module("block", blocks);
#else
	if (n_blocks != decoder_channels.size()) std::cout << "Model depth not equal to your provided `decoder_channels`";
	std::reverse(std::begin(encoder_channels), std::end(encoder_channels));

	// computing blocks input and output channels
	int head_channels = encoder_channels[0];
	std::vector<int> out_channels = decoder_channels;
	decoder_channels.pop_back();
	decoder_channels.insert(decoder_channels.begin(), head_channels);
	std::vector<int> in_channels = decoder_channels;
	encoder_channels.erase(encoder_channels.begin());
	std::vector<int> skip_channels = encoder_channels;
	skip_channels[skip_channels.size() - 1] = 0;

	if (use_center)  center = CenterBlock(head_channels, head_channels);
	else center = torch::nn::Sequential(torch::nn::Identity());
	//the last DecoderBlock of blocks need no skip tensor
	for (int i = 0; i < in_channels.size() - 1; i++) {
		blocks->push_back(DecoderBlock(in_channels[i], skip_channels[i], out_channels[i], false, use_attention));
	}
	blocks->push_back(DecoderBlock(in_channels[in_channels.size() - 1], skip_channels[in_channels.size() - 1],
		out_channels[in_channels.size() - 1], false, use_attention));

	register_module("center", center);
	register_module("blocks", blocks); 
#endif
}

torch::Tensor UNetDecoderImpl::forward(std::vector<torch::Tensor> features)
{
	std::reverse(std::begin(features), std::end(features));
	torch::Tensor head = features[0];
	features.erase(features.begin());
	auto x = center->forward(head);
	for (int i = 0; i < blocks->size(); i++)
	{
		x = blocks[i]->as<DecoderBlock>()->forward(x, features[i]);
	}
	return x;
}

// -- 2022-1-18 Refer to the python UNET model and rewrite the C + +code ---
DoubleConvImpl::DoubleConvImpl(int in_channel, int out_channel, int mid_channel /* = -1 */)
{
	if (mid_channel == -1)
	{
		mid_channel = out_channel;
	}

	double_conv = torch::nn::Sequential(
		torch::nn::Conv2d(conv_options(in_channel, mid_channel, 3).bias(false).padding(1)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(mid_channel)),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
		torch::nn::Conv2d(conv_options(mid_channel, out_channel, 3).bias(false).padding(1)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channel)),
		torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
		);

	register_module("double_conv", double_conv);
}

torch::Tensor DoubleConvImpl::forward(torch::Tensor x)
{
	x = double_conv->forward(x);
	return x;
}

InConvImpl::InConvImpl(int in_channel, int out_channel)
{
	double_conv = DoubleConv(in_channel, out_channel);
	
	register_module("double_conv", double_conv);
}

torch::Tensor InConvImpl::forward(torch::Tensor x)
{
	x = double_conv->forward(x);
	return x;
}

DownScaleImpl::DownScaleImpl(int in_channel, int out_channel)
{
	maxpool_conv = torch::nn::Sequential(
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
		DoubleConv(in_channel, out_channel)
	);

	register_module("maxpool", maxpool_conv);
}

torch::Tensor DownScaleImpl::forward(torch::Tensor x)
{
	x = maxpool_conv->forward(x);
	return x;
}

UpScaleImpl::UpScaleImpl(int in_channel, int out_channel, bool bilinear)
{
	bilinear_ = bilinear;
	up_upsample = torch::nn::Upsample(
		torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2 })).mode(torch::kBilinear).align_corners(true));
	up_convtrans = torch::nn::ConvTranspose2d(
		torch::nn::ConvTranspose2dOptions(in_channel/2, in_channel/2, 2).stride(2));
	double_conv = DoubleConv(in_channel, out_channel);

	register_module("upsample", up_upsample);
	register_module("convtrans", up_convtrans);
	register_module("double_conv", double_conv);
}

torch::Tensor UpScaleImpl::forward(torch::Tensor x1, torch::Tensor x2)
{
	if (bilinear_)
		x1 = up_upsample->forward(x1);
	else
		x1 = up_convtrans->forward(x1);

	auto diffX = x2.sizes()[2] - x1.sizes()[2];	// NCWH
	auto diffY = x2.sizes()[3] - x1.sizes()[3];

	x1 = torch::nn::functional::pad(x1, torch::nn::functional::PadFuncOptions({ diffX / 2, diffX - diffX / 2, diffY / 2, diffY - diffY / 2 }));
	
	auto x = torch::cat({ x2, x1 }, 1);
	x = double_conv->forward(x);

	return x;
}

OutConvImpl::OutConvImpl(int in_channel, int out_channel)
{
	conv = torch::nn::Conv2d(conv_options(in_channel, out_channel, 1));

	register_module("conv", conv);
}

torch::Tensor OutConvImpl::forward(torch::Tensor x)
{
	return conv->forward(x);
}

}
