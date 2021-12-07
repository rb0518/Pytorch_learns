#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

struct Residual : torch::nn::Module
{
	int in_channel_r;
	int out_channel_r;
	int _stride_r;

	Residual(int in_channel, int out_channel, int _stride = 1)
	{
		conv1 = torch::nn::Module::register_module("conv1", torch::nn::Conv2d(
			torch::nn::Conv2dOptions(in_channel, out_channel, { 3, 3 }).padding(1).stride({ _stride, _stride })));
		
		c1b = register_module("c1b", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(
			out_channel).eps(1e-5).momentum(0.1).affine(true).track_running_stats(true)));
		
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(
			out_channel, out_channel, { 3,3 }).padding(1).stride({ 1, 1 })));

		c2b = register_module("c2b", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(
			out_channel).eps(1e-5).momentum(0.1).affine(true).track_running_stats(true)));
		//注意conv3 1*1卷积层与conv1卷积层的stride须保持一致
		conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(
			in_channel, out_channel, { 1, 1 }).stride({ _stride, _stride })));

		c3b = register_module("c3b", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(
			out_channel).eps(1e-5).momentum(0.1).affine(true).track_running_stats(true)));


		in_channel_r = in_channel;
		out_channel_r = out_channel;
	} 

	torch::Tensor forward(torch::Tensor input)
	{
		namespace F = torch::nn::functional;


		auto x = conv1->forward(input);
		x = c1b->forward(x);
		x = F::relu(x);


		x = conv2->forward(x);
		x = c2b->forward(x);


		torch::Tensor x1;
		//如果输入通道数不等于输出通道数，将对输入进行1*1卷积
		if (in_channel_r != out_channel_r)
		{
			x1 = conv3->forward(input);
			x1 = c3b->forward(x1);
		}
		else
		{
			x1 = input;
		}


		x = x + x1;  //两层卷积结果与输入信号之和
		x = F::relu(x);


		return x;
	}

	torch::nn::Conv2d		conv1	{ nullptr };
	torch::nn::BatchNorm2d	c1b		{ nullptr };
	torch::nn::Conv2d		conv2	{ nullptr };
	torch::nn::BatchNorm2d  c2b		{ nullptr };
	torch::nn::Conv2d		conv3	{ nullptr };
	torch::nn::BatchNorm2d  c3b     { nullptr };
};

struct Resnet34 : torch::nn::Module
{
    Resnet34(int in_channel, int num_class = 10)
    {
        //注册模块1
        conv1 = register_module("conv1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, 64, { 3, 3 }).padding(1).stride({ 1, 1 })),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64).eps(1e-5).momentum(0.1).affine(true).track_running_stats(true)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ 3, 3 }).stride({ 1, 1 }).padding(1)))
        );

        //注册模块2
        conv2 = register_module("conv2", torch::nn::Sequential(
            Residual(64, 64),    //重复调用残差模块，也相当于将多个残差模块装入顺序容器中
            Residual(64, 64),
            Residual(64, 64))
        );

        //注册模块3
        conv3 = register_module("conv3", torch::nn::Sequential(
            Residual(64, 128, 2),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128))
        );

        //注册模块4
        conv4 = register_module("conv4", torch::nn::Sequential(
            Residual(128, 256, 2),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256))
        );

        //注册模块5
        conv5 = register_module("conv5", torch::nn::Sequential(
            Residual(256, 512, 2),
            Residual(512, 512),
            Residual(512, 512))
        );


        //注册全连接层的Affine层，由于模块6只有1个池化层和1个全连接层，就不使用顺序容器实现了，且池化层不需要在此注册，放在前向传播函数中实现即可
        fc = register_module("fc", torch::nn::Linear(512, num_class));
    }




    ~Resnet34()
    {


    }




    //前向传播函数
    torch::Tensor forward(torch::Tensor input)
    {
        namespace F = torch::nn::functional;

            //模块1
        auto x = conv1->forward(input);
        //模块2
        x = conv2->forward(x);
        //模块3
        x = conv3->forward(x);
        //模块4
        x = conv4->forward(x);
        //模块5
        x = conv5->forward(x);
        //模块6
        x = F::avg_pool2d(x, F::AvgPool2dFuncOptions(4));  //池化窗口4*4，stride默认与窗口尺寸一致4*4，padding默认为0
        x = x.view({ x.size(0), -1 });  //一维展开
        x = fc->forward(x);   //Affine层，Softmax层不需要在此处计算，因为后面的CrossEntropyLoss交叉熵函数本身包含了Softmax计算


        return x;
    }


    // Use one of many "standard library" modules.
    torch::nn::Sequential conv1{ nullptr };
    torch::nn::Sequential conv2{ nullptr };
    torch::nn::Sequential conv3{ nullptr };
    torch::nn::Sequential conv4{ nullptr };
    torch::nn::Sequential conv5{ nullptr };
    torch::nn::Linear fc{ nullptr };
};


