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
		//ע��conv3 1*1�������conv1������stride�뱣��һ��
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
		//�������ͨ�������������ͨ�����������������1*1���
		if (in_channel_r != out_channel_r)
		{
			x1 = conv3->forward(input);
			x1 = c3b->forward(x1);
		}
		else
		{
			x1 = input;
		}


		x = x + x1;  //����������������ź�֮��
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
        //ע��ģ��1
        conv1 = register_module("conv1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, 64, { 3, 3 }).padding(1).stride({ 1, 1 })),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64).eps(1e-5).momentum(0.1).affine(true).track_running_stats(true)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ 3, 3 }).stride({ 1, 1 }).padding(1)))
        );

        //ע��ģ��2
        conv2 = register_module("conv2", torch::nn::Sequential(
            Residual(64, 64),    //�ظ����òв�ģ�飬Ҳ�൱�ڽ�����в�ģ��װ��˳��������
            Residual(64, 64),
            Residual(64, 64))
        );

        //ע��ģ��3
        conv3 = register_module("conv3", torch::nn::Sequential(
            Residual(64, 128, 2),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128))
        );

        //ע��ģ��4
        conv4 = register_module("conv4", torch::nn::Sequential(
            Residual(128, 256, 2),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256))
        );

        //ע��ģ��5
        conv5 = register_module("conv5", torch::nn::Sequential(
            Residual(256, 512, 2),
            Residual(512, 512),
            Residual(512, 512))
        );


        //ע��ȫ���Ӳ��Affine�㣬����ģ��6ֻ��1���ػ����1��ȫ���Ӳ㣬�Ͳ�ʹ��˳������ʵ���ˣ��ҳػ��㲻��Ҫ�ڴ�ע�ᣬ����ǰ�򴫲�������ʵ�ּ���
        fc = register_module("fc", torch::nn::Linear(512, num_class));
    }




    ~Resnet34()
    {


    }




    //ǰ�򴫲�����
    torch::Tensor forward(torch::Tensor input)
    {
        namespace F = torch::nn::functional;

            //ģ��1
        auto x = conv1->forward(input);
        //ģ��2
        x = conv2->forward(x);
        //ģ��3
        x = conv3->forward(x);
        //ģ��4
        x = conv4->forward(x);
        //ģ��5
        x = conv5->forward(x);
        //ģ��6
        x = F::avg_pool2d(x, F::AvgPool2dFuncOptions(4));  //�ػ�����4*4��strideĬ���봰�ڳߴ�һ��4*4��paddingĬ��Ϊ0
        x = x.view({ x.size(0), -1 });  //һάչ��
        x = fc->forward(x);   //Affine�㣬Softmax�㲻��Ҫ�ڴ˴����㣬��Ϊ�����CrossEntropyLoss�����غ������������Softmax����


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


