#ifndef _SEGMENTOR_H_
#define _SEGMENTOR_H_

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "SegDataset.h"
#include "util.h"
#include "loss.hpp"

// Instantiate a class with float and double specifications.
#define INSTANTIATE_SEG_CLASS(classname) \
  template class Segmentor<##classname> 


template <class Model>
class Segmentor
{
public:
	Segmentor() {};
	~Segmentor() {};

	void Initialize(int gpu_id, int width, int height, std::vector<std::string>&& name_list,
		std::string encoder_name, std::string pretrained_path);

	void Train(float learning_rate, int epochs, int batch_size,
		std::string train_val_path, std::string image_type, std::string save_path);

	void LoadWeight(std::string weight_path);

	void Predict(cv::Mat& image, const std::string& which_class);

	void SetTrainTricks(trainTricks& tricks);
private:
	int width_ = 512;
	int height_ = 512;
	std::vector<std::string> name_list_;
	torch::Device device_ = torch::Device(torch::kCPU);
	Model model_;
	trainTricks tricks_;
};
#endif

