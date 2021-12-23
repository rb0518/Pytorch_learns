#pragma once
#include <torch/torch.h>
#include <torch/script.h>

struct VGG : public torch::nn::Module
{
	VGG(bool nBN) : in_channels(3)
	{
		conv2d_1 = register_module("conv2d_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).padding(1)));
		relu_1 = register_module("relu_1", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_2 = register_module("conv2d_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
		relu_2 = register_module("relu_1", torch::nn::ReLU(torch::nn::ReLUOptions(true)));

		conv2d_3 = register_module("conv2d_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(3)));
		relu_3 = register_module("relu_3", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_4 = register_module("conv2d_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(3)));
		relu_4 = register_module("relu_4", torch::nn::ReLU(torch::nn::ReLUOptions(true)));

		conv2d_5 = register_module("conv2d_5", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
		relu_5 = register_module("relu_5", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_6 = register_module("conv2d_6", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
		relu_6 = register_module("relu_6", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_7 = register_module("conv2d_7", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
		relu_7 = register_module("relu_7", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_8 = register_module("conv2d_8", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
		relu_8 = register_module("relu_8", torch::nn::ReLU(torch::nn::ReLUOptions(true)));

		conv2d_9 = register_module("conv2d_9", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
		relu_9 = register_module("relu_9", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_10 = register_module("conv2d_10", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_10 = register_module("relu_10", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_11 = register_module("conv2d_11", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_11 = register_module("relu_11", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_12 = register_module("conv2d_12", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_12 = register_module("relu_12", torch::nn::ReLU(torch::nn::ReLUOptions(true)));

		conv2d_13 = register_module("conv2d_13", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_13 = register_module("relu_13", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_14 = register_module("conv2d_14", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_14 = register_module("relu_14", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_15 = register_module("conv2d_15", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_15 = register_module("relu_15", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_16 = register_module("conv2d_16", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_16 = register_module("relu_16", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	}

	torch::Tensor forward(torch::Tensor input)
	{
		xmaxpool_1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))(relu_2(conv2d_2(relu_1(conv2d_1(input)))));
		xmaxpool_2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))(relu_4(conv2d_4(relu_3(conv2d_3(xmaxpool_1)))));
		xmaxpool_3 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))(relu_8(conv2d_8(relu_7(conv2d_7(relu_6(conv2d_6(relu_5(conv2d_5(xmaxpool_2)))))))));
		xmaxpool_4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))(relu_12(conv2d_12(relu_11(conv2d_11(relu_10(conv2d_10(relu_9(conv2d_9(xmaxpool_3)))))))));
		xmaxpool_5 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))(relu_12(conv2d_16(relu_16(conv2d_15(relu_15(conv2d_14(relu_13(conv2d_13(xmaxpool_4)))))))));

		return xmaxpool_5;
	}

public:
	torch::nn::Conv2d conv2d_1{ nullptr }, conv2d_2{ nullptr}, conv2d_3{ nullptr }, conv2d_4{ nullptr };
	torch::nn::Conv2d conv2d_5{ nullptr }, conv2d_6{ nullptr }, conv2d_7{ nullptr }, conv2d_8{ nullptr };
	torch::nn::Conv2d conv2d_9{ nullptr }, conv2d_10{ nullptr }, conv2d_11{ nullptr }, conv2d_12{ nullptr };
	torch::nn::Conv2d conv2d_13{ nullptr }, conv2d_14{ nullptr }, conv2d_15{ nullptr }, conv2d_16{ nullptr };

	torch::nn::ReLU relu_1{ nullptr }, relu_2{ nullptr }, relu_3{ nullptr }, relu_4{ nullptr };
	torch::nn::ReLU relu_5{ nullptr }, relu_6{ nullptr }, relu_7{ nullptr }, relu_8{ nullptr };
	torch::nn::ReLU relu_9{ nullptr }, relu_10{ nullptr }, relu_11{ nullptr }, relu_12{ nullptr };
	torch::nn::ReLU relu_13{ nullptr }, relu_14{ nullptr }, relu_15{ nullptr }, relu_16{ nullptr };

	torch::Tensor xmaxpool_1, xmaxpool_2, xmaxpool_3, xmaxpool_4, xmaxpool_5;
private:
	int in_channels;
};

struct FCN8s : public torch::nn::Module
{
	FCN8s(int c) :n_class(c), in_channels(3)
	{
		//VGGNet8s
		conv2d_1 = register_module("conv2d_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).padding(1)));
		relu_1 = register_module("relu_1", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_2 = register_module("conv2d_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
		relu_2 = register_module("relu_2", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		maxpool_1 = register_module("maxpool_1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

		conv2d_3 = register_module("conv2d_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
		relu_3 = register_module("relu_3", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_4 = register_module("conv2d_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
		relu_4 = register_module("relu_4", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		maxpool_2 = register_module("maxpool_2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

		conv2d_5 = register_module("conv2d_5", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
		relu_5 = register_module("relu_5", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_6 = register_module("conv2d_6", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
		relu_6 = register_module("relu_6", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_7 = register_module("conv2d_7", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
		relu_7 = register_module("relu_7", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		maxpool_3 = register_module("maxpool_3", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

		conv2d_9 = register_module("conv2d_9", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
		relu_9 = register_module("relu_9", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_10 = register_module("conv2d_10", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_10 = register_module("relu_10", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_11 = register_module("conv2d_11", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_11 = register_module("relu_11", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		maxpool_4 = register_module("maxpool_4", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

		conv2d_13 = register_module("conv2d_13", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_13 = register_module("relu_13", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_14 = register_module("conv2d_14", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_14 = register_module("relu_14", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		conv2d_15 = register_module("conv2d_15", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		relu_15 = register_module("relu_15", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		maxpool_5 = register_module("maxpool_5", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
	
		// FCN
		convTranspose2d_1 = register_module("convTranspose2d_1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 512, 3).stride(2).padding(1).dilation(1).output_padding(1)));
		relu_fcn_1 = register_module("relu_fcn_1", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		batchNorm2d_1 = register_module("batchNorm2d_1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512)));
		convTranspose2d_2 = register_module("convTranspose2d_2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 256, 3).stride(2).padding(1).dilation(1).output_padding(1)));
		relu_fcn_2 = register_module("relu_fcn_2", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		batchNorm2d_2 = register_module("batchNorm2d_2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)));
		convTranspose2d_3 = register_module("convTranspose2d_3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).dilation(1).output_padding(1)));
		relu_fcn_3 = register_module("relu_fcn_3", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		batchNorm2d_3 = register_module("batchNorm2d_3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)));
		convTranspose2d_4 = register_module("convTranspose2d_4", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 3).stride(2).padding(1).dilation(1).output_padding(1)));
		relu_fcn_4 = register_module("relu_fcn_4", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		batchNorm2d_4 = register_module("batchNorm2d_4", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
		convTranspose2d_5 = register_module("convTranspose2d_5", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 3).stride(2).padding(1).dilation(1).output_padding(1)));
		relu_fcn_5 = register_module("relu_fcn_5", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
		batchNorm2d_5 = register_module("batchNorm2d_5", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)));
		classifier = register_module("conv2d", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, n_class, 1)));
	}
	~FCN8s()
	{

	}

	torch::Tensor forward(torch::Tensor input)
	{
		//VGGNet 
		xmaxpool_1 = maxpool_1((relu_2(conv2d_2(relu_1(conv2d_1(input))))));
		xmaxpool_2 = maxpool_2((relu_4(conv2d_4(relu_3(conv2d_3(xmaxpool_1))))));
		xmaxpool_3 = maxpool_3(((relu_7((conv2d_7((relu_6(conv2d_6(relu_5(conv2d_5(xmaxpool_2)))))))))));
		xmaxpool_4 = maxpool_4((relu_11((conv2d_11((relu_10(conv2d_10(relu_9(conv2d_9(xmaxpool_3))))))))));
		xmaxpool_5 = maxpool_5((relu_15((conv2d_15((relu_14(conv2d_14(relu_13(conv2d_13(xmaxpool_4))))))))));
		//FCN
		score = relu_fcn_1(convTranspose2d_1(xmaxpool_5));
		score = score.add(xmaxpool_4);
		score = batchNorm2d_1(score);
		score = relu_fcn_2(convTranspose2d_2(score));
		score = score.add(xmaxpool_3);
		score = batchNorm2d_2(score);
		score = batchNorm2d_3(relu_fcn_3(convTranspose2d_3(score)));
		score = batchNorm2d_4(relu_fcn_4(convTranspose2d_4(score)));
		score = batchNorm2d_5(relu_fcn_5(convTranspose2d_5(score)));
		score = classifier(score);
		return score;
	}

	//FCN
	torch::Tensor score;
	int n_class;
	torch::nn::ReLU relu_fcn_1{ nullptr }, relu_fcn_2{ nullptr }, relu_fcn_3{ nullptr }, relu_fcn_4{ nullptr }, relu_fcn_5{ nullptr };
	torch::nn::ConvTranspose2d convTranspose2d_1{ nullptr }, convTranspose2d_2{ nullptr }, convTranspose2d_3{ nullptr }, convTranspose2d_4{ nullptr }, convTranspose2d_5{ nullptr };
	torch::nn::Conv2d classifier{ nullptr };
	torch::nn::BatchNorm2d batchNorm2d_1{ nullptr }, batchNorm2d_2{ nullptr }, batchNorm2d_3{ nullptr }, batchNorm2d_4{ nullptr }, batchNorm2d_5{ nullptr };

	//VGGNet
	int  in_channels;
	torch::Tensor xmaxpool_1, xmaxpool_2, xmaxpool_3, xmaxpool_4, xmaxpool_5;

	torch::nn::Conv2d conv2d_1{ nullptr }, conv2d_2{ nullptr }, conv2d_3{ nullptr }, conv2d_4{ nullptr }, conv2d_5{ nullptr }, conv2d_6{ nullptr };
	torch::nn::Conv2d conv2d_7{ nullptr }, conv2d_8{ nullptr }, conv2d_9{ nullptr }, conv2d_10{ nullptr }, conv2d_11{ nullptr }, conv2d_12{ nullptr };
	torch::nn::Conv2d conv2d_13{ nullptr }, conv2d_14{ nullptr }, conv2d_15{ nullptr }, conv2d_16{ nullptr };
	torch::nn::ReLU relu_1{ nullptr }, relu_2{ nullptr }, relu_3{ nullptr }, relu_4{ nullptr }, relu_5{ nullptr }, relu_6{ nullptr };
	torch::nn::ReLU relu_7{ nullptr }, relu_8{ nullptr }, relu_9{ nullptr }, relu_10{ nullptr }, relu_11{ nullptr }, relu_12{ nullptr };
	torch::nn::ReLU relu_13{ nullptr }, relu_14{ nullptr }, relu_15{ nullptr }, relu_16{ nullptr };
	torch::nn::MaxPool2d maxpool_1{ nullptr }, maxpool_2{ nullptr }, maxpool_3{ nullptr }, maxpool_4{ nullptr }, maxpool_5{ nullptr };
};

