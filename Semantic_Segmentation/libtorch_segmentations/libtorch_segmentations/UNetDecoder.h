#pragma once
#include <torch/torch.h>

class SCSEModuleImpl : public torch::nn::Module
{
public:
	SCSEModuleImpl(int in_channels, int reduction = 16, bool use_attention = false);
	torch::Tensor forward(torch::Tensor x);
private:
	bool use_attention_ = false;
	torch::nn::Sequential cSE{ nullptr };
	torch::nn::Sequential sSE{ nullptr };
};
TORCH_MODULE(SCSEModule);

class Conv2dReLUImpl : public torch::nn::Module
{
public:
	Conv2dReLUImpl(int in_channels, int out_channels, int kernel_size = 3, int padding = 1);
	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::Conv2d conv2d_{ nullptr };
	torch::nn::BatchNorm2d batch_norm_{ nullptr };
};
TORCH_MODULE(Conv2dReLU);


class DecoderBlockImpl : public torch::nn::Module 
{
public:
	DecoderBlockImpl(int in_channels, int skip_channels, int out_channels, bool skip = true, bool attention = false);
	torch::Tensor forward(torch::Tensor x, torch::Tensor skip);
private:
	Conv2dReLU conv1{ nullptr };
	Conv2dReLU conv2{ nullptr };
	SCSEModule attention1{ nullptr };
	SCSEModule attention2{ nullptr };
	torch::nn::Upsample upsample{ nullptr };
	bool is_skip = true;
}; TORCH_MODULE(DecoderBlock);

torch::nn::Sequential CenterBlock(int in_channels, int out_channels);

class UNetDecoderImpl : public torch::nn::Module
{
public:
	UNetDecoderImpl(std::vector<int> encoder_channels, std::vector<int> decoder_channels, int n_block = 5,
		bool use_attention = false, bool use_center = false);
	torch::Tensor forward(std::vector<torch::Tensor> features);
private:
	torch::nn::Sequential center_{ nullptr };
	torch::nn::ModuleList blocks_ = torch::nn::ModuleList();
};
TORCH_MODULE(UNetDecoder);