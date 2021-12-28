#pragma once
#include <iostream>
#include <string>
#include <torch/torch.h>
#include "VOCSegDataset.h"
#include "fcn.h"

class FCN_Train
{
public:
	typedef struct tagSettings {
		std::string device_name;
		std::string str_data_root;
		int num_class;
		int epochs;
		int batch_size;
		int auto_step;		// 自动调整学习阈值和存储的步长
		int check_step;		// 输出学习信息的步长
		float lr_init;		// learning rate first value
		float lr_gamma;		// learning rate adjust value

		std::string out_root;

		void showInfo()
		{
			std::cout << "You settings: " << std::endl;
			std::cout << "   device: " << device_name << std::endl;
			std::cout << "   Dataset root: " << str_data_root << std::endl;
			std::cout << "   Output root: " << out_root << std::endl;
			std::cout << "   Batch size: " << batch_size << " total epochs: " << epochs << std::endl;
			std::cout << "   learn rate init: " << lr_init << " learn rate adjust value: " << lr_gamma << std::endl;
		}

	}Settings;
public:
	explicit FCN_Train(const std::string& deviceType, const Settings& sets);
	~FCN_Train() { ; }

	void Run();
private:
	Settings sets_;
};

