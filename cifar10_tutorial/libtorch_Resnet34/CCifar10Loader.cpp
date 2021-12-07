#include "CCifar10Loader.h"

#include <filesystem>
#include <iostream>
#include <sstream>

#include <fstream>


CCifar10Loader::CCifar10Loader(const std::string& folder, DATATYPE type)
	: folder_(folder), datatype_(type)
{
	read_cifar10_bin();
}

void CCifar10Loader::read_cifar10_bin()
{
	const int image_total = 10000;
	const int train_file_total = 5;
	const int test_file_total = 1;
	images_.clear();
	labels_id_.clear();
	
	int filetotal = (_TRAIN_ == datatype_) ? train_file_total : test_file_total;

	std::vector<std::vector<std::string>> filenames = { {"data_batch_1.bin",
															"data_batch_2.bin",
															"data_batch_3.bin",
															"data_batch_4.bin",
															"data_batch_5.bin"},
														{"test_batch.bin"} };

	for(int i = 0; i < filetotal; i++)
	{
		auto filepathname = std::filesystem::path(folder_);

		filepathname.append(filenames[datatype_][i]);
		std::ifstream readfile;
		readfile.open(filepathname, std::ios::in | std::ios::binary);
		if (readfile)
		{
			for (int j = 0; j < image_total; j++)
			{
				uchar readbuffer[3073];
				readfile.read((char*)readbuffer, 3073 * sizeof(uchar));
				labels_id_.push_back(readbuffer[0]);
				
				cv::Mat matChannels[3];
				matChannels[0] = cv::Mat(32, 32, CV_8UC1);
				matChannels[1] = cv::Mat(32, 32, CV_8UC1);
				matChannels[2] = cv::Mat(32, 32, CV_8UC1);

				for (int k = 0; k < 3; k++)
				{
					memcpy(matChannels[k].data, readbuffer + 1 + (2 - k) * 1024, 1024 * sizeof(uchar));
				}

				auto matRGB = cv::Mat(32, 32, CV_8UC3);
				cv::merge(&matChannels[0], 3, matRGB);
				images_.push_back(matRGB.clone());
			}

			std::cout << "Load data from file: " << filepathname.string() << " success." << std::endl;
		}
		else
		{
			std::cout << "open file : " << filepathname << "error !" << std::endl;
		}
	}
}
