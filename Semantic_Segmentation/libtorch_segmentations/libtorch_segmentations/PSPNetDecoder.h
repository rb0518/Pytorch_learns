#pragma once
#include <torch/torch.h>

class PSPBlockImpl : public torch::nn::Module
{
public:
	PSPBlockImpl(int in_channels, int out_channels, int pool_size, bool use_batchnorm = true);
	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::Sequential pool{ nullptr };
	torch::nn::Sequential conv{ nullptr };
};
TORCH_MODULE(PSPBlock);

class PSPModuleImpl : public torch::nn::Module 
{
public:
	PSPModuleImpl(int in_channels, std::vector<int> sizes, bool use_batchnorm = true);
	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::ModuleList blocks;
};
TORCH_MODULE(PSPModule);

class PSPDecoderImpl : public torch::nn::Module
{
public:
	PSPDecoderImpl(std::vector<int> encoder_channels, int out_channels, double _dropout = 0.2,
		bool use_batchnorm = true, int encoder_depth = 3);
	torch::Tensor forward(std::vector<torch::Tensor> xs);
private:
	int encoder_depth_ = 3;
	PSPModule psp{ nullptr };
	torch::nn::Sequential conv{ nullptr };
	torch::nn::Dropout2d dropout{ nullptr };
};
TORCH_MODULE(PSPDecoder);