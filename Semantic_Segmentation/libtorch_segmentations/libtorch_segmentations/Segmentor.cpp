#include "Segmentor.h"

#include "UNet.h"
#include "FPN.h"

INSTANTIATE_SEG_CLASS(UNet);
INSTANTIATE_SEG_CLASS(FPN);

template <class Model>
void Segmentor<Model>::Initialize(int gpu_id, int width, int height, std::vector<std::string>&& name_list,
	std::string encoder_name, std::string pretrained_path)
{
	width_ = width;
	height_ = height;
	name_list = name_list;
	CHECK(std::filesystem::exists(std::filesystem::path(pretrained_path))) << "pretrained_path is invalid";
	CHECK(name_list.size() >= 2) << "Class name is less than 1";
	if (gpu_id >= torch::getNumGPUs()) std::cout << "GPU id exceed max number of gpu";
	if (gpu_id >= 0) {
		device_ = torch::Device(torch::kCUDA, gpu_id);
	}

	model_ = Model(name_list.size(), encoder_name, pretrained_path);

	size_t num_params = 0;
	for (auto param : model_->parameters())
	{
		num_params += param.numel();
	}

	std::ofstream ofs("d://test.txt");
	ofs << "Total number of parameters: " << (float)num_params / 1e6f << "M" << std::endl << std::endl;
	ofs << model_ << std::endl;
	ofs.close();

	model_->to(device_);
}

template <class Model>
void Segmentor<Model>::Train(float learning_rate, int epochs, int batch_size,
	std::string train_val_path, std::string image_type, std::string save_path)
{
	std::string train_dir = (std::filesystem::path(train_val_path).append("train")).string();
	std::string val_dir = (std::filesystem::path(train_val_path).append("val")).string();

	std::vector<std::string> list_image_train = {};
	std::vector<std::string> list_label_train = {};
	std::vector<std::string> list_image_val = {};
	std::vector<std::string> list_label_val = {};

	load_seg_data_from_folder(train_dir, image_type, list_image_train, list_label_train);
	load_seg_data_from_folder(val_dir, image_type, list_image_val, list_label_val);

	auto custom_dataset_train = SegDataset(width_, height_, list_image_train, list_label_train,
		name_list_, tricks_, true).map(torch::data::transforms::Stack<>());
	auto custom_dataset_val = SegDataset(width_, height_, list_image_val, list_label_val,
		name_list_, tricks_, true).map(torch::data::transforms::Stack<>());

	auto options = torch::data::DataLoaderOptions();
	options.drop_last(true);
	options.batch_size(batch_size);
	auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(custom_dataset_train), options);
	auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(custom_dataset_val), options);

	float best_loss = 1e10;
	LOG(INFO) << "Start training...";
	std::cout << "train path: " << train_dir << std::endl;
	std::cout << "val path: " << val_dir << std::endl;
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		float loss_sum = 0;
		int batch_count = 0;
		float loss_train = 0;
		float dice_coef_sum = 0;

		for (auto decay_epoch : tricks_.decay_epochs)
		{
			if (decay_epoch - 1 == epoch)
				learning_rate /= 10;
		}

		torch::optim::Adam optimizer(model_->parameters(), learning_rate);
		if (epoch < tricks_.freeze_epochs)
		{
			for (auto mm : model_->named_parameters())
			{
				if (strstr(mm.key().data(), "encoder"))
				{
					mm.value().set_requires_grad(false);
				}
				else
				{
					mm.value().set_requires_grad(true);
				}
			}
		}
		else
		{
			for (auto mm : model_->named_parameters())
			{
				mm.value().set_requires_grad(true);
			}
		}

		model_->train();

		for (auto& batch : *data_loader_train)
		{
			auto data = batch.data;
			auto target = batch.target;

			data = data.to(torch::kF32).to(device_).div(255.0);
			target = target.to(torch::kLong).to(device_).squeeze(1);
			// 			std::cout << "data.sizes:" << data.sizes() << std::endl;
			// 			std::cout << "target.sizes: " << target.sizes() << std::endl;
			optimizer.zero_grad();

			torch::Tensor prediction = model_->forward(data);

			 			std::cout << "target.sizes:" << target.sizes() << std::endl;
			 			std::cout << "prediction.sizes: " << prediction.sizes() << std::endl;
			torch::Tensor ce_loss = CELoss(prediction, target);

			torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), name_list_.size());
			auto loss = dice_loss * tricks_.dice_ce_ratio + ce_loss * (1 - tricks_.dice_ce_ratio);
			// Compute gradients
			loss.backward();
			optimizer.step();
			loss_sum += loss.item().toFloat();
			dice_coef_sum += (1 - dice_loss).item().toFloat();
			batch_count++;
			loss_train = loss_sum / batch_count / batch_size;
			auto dice_coef = dice_coef_sum / batch_count;

			std::cout << "Epoch: " << epoch << ", Train loss:" << loss_train <<
				"  Dice coefficient: " << dice_coef << std::endl;

		}

		model_->eval();
		loss_sum = 0;
		batch_count = 0;
		dice_coef_sum = 0;
		float loss_val = 0;

		for (auto& batch : *data_loader_val)
		{
			auto data = batch.data;
			auto target = batch.target;
			data = data.to(torch::kF32).to(device_).div(255.0);
			target = target.to(torch::kLong).to(device_).squeeze(1);

			torch::Tensor prediction = model_->forward(data);

			torch::Tensor ce_loss = CELoss(prediction, target);
			torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), name_list_.size());
			auto loss = dice_loss * tricks_.dice_ce_ratio + ce_loss * (1 - tricks_.dice_ce_ratio);
			loss_sum += loss.template item<float>();
			dice_coef_sum += (1 - dice_loss).item().toFloat();
			batch_count++;
			loss_val = loss_sum / batch_count / batch_size;
			auto dice_coef = dice_coef_sum / batch_count;

			std::cout << "Epoch: " << epoch << " , Validation Loss: " << loss_val <<
				" , Dice coefficient: " << dice_coef << std::endl;
		}

		if (loss_val < best_loss)
		{
			torch::save(model_, save_path);
			best_loss = loss_val;
		}
	}
}

template <class Model>
void Segmentor<Model>::LoadWeight(std::string weight_path)
{
	torch::load(model_, weight_path);
	model_->to(device_);
	model_->eval();
}

template <class Model>
void Segmentor<Model>::Predict(cv::Mat& image, const std::string& which_class)
{
	cv::Mat srcImg = image.clone();
	int which_class_index = -1;
	for (int i = 0; i < name_list_.size(); i++)
	{
		if (name_list_[i] == which_class)
		{
			which_class_index = i;
			break;
		}
	}

	CHECK(which_class_index != -1) << which_class << " not in the name list";
	int image_width = image.cols;
	int image_height = image.rows;

	cv::resize(image, image, cv::Size(width_, height_));
	torch::Tensor tensor_image = torch::from_blob(image.data, { 1, height_, width_, 3 }, torch::kByte);
	tensor_image = tensor_image.to(device_);
	tensor_image = tensor_image.permute({ 0, 3, 1, 2 });		// HWC==>CHW
	tensor_image = tensor_image.to(torch::kFloat);
	tensor_image = tensor_image.div(255.0);

	try {
		at::Tensor output = model_->forward({ tensor_image });
	}
	catch (const std::exception& e)
	{
		LOG(WARNING) << e.what();
	}
	at::Tensor output = model_->forward({ tensor_image });
	output = torch::softmax(output, 1).mul(255.0).toType(torch::kByte);
	image = cv::Mat::ones(cv::Size(width_, height_), CV_8UC1);

	at::Tensor re = output[0][which_class_index].to(torch::kCPU).detach();
	memcpy(image.data, re.data_ptr(), width_ * height_ * sizeof(unsigned char));
	cv::resize(image, image, cv::Size(image_width, image_height));

	//	cv::imwrite("prediction.jpg", image);
	cv::imshow("prediction", image);
	cv::imshow("srcImage", srcImg);
	cv::waitKey();
	cv::destroyAllWindows();
}

template <class Model>
void Segmentor<Model>::SetTrainTricks(trainTricks& tricks)
{
	this->tricks_ = tricks;
}

