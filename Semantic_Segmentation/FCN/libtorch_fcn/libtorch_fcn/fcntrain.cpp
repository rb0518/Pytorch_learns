#include "fcntrain.h"
#include <filesystem>
#include <glog/logging.h>

FCN_Train::FCN_Train(const std::string& devicetype, const Settings& sets)
	: sets_(sets)
{
	sets_.showInfo();
}

void FCN_Train::Run()
{
	torch::DeviceType device = (sets_.device_name == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
/*	std::cout << device << std::endl;*/

	auto train_dataset = VOCSegDataset(sets_.str_data_root, "train", 21).map(torch::data::transforms::Stack<>());
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset),
		torch::data::DataLoaderOptions().batch_size(sets_.batch_size).workers(1));

	auto val_dataset = VOCSegDataset(sets_.str_data_root, "val", 21).map(torch::data::transforms::Stack<>());
	auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(val_dataset),
		torch::data::DataLoaderOptions().batch_size(sets_.batch_size).workers(1));

	FCN8s fcn8s(sets_.num_class);

#if 0
	auto criterion = torch::nn::BCEWithLogitsLoss();
	auto optimizer = torch::optim::RMSprop(fcn8s.parameters(),
		torch::optim::RMSpropOptions(double(sets_.lr_init)).momentum(0).weight_decay(10e-5));
#else
	auto criterion = torch::nn::CrossEntropyLoss();
	auto optimizer = torch::optim::Adam(fcn8s.parameters(),	torch::optim::AdamOptions(double(sets_.lr_init)).weight_decay(10e-5));
#endif
	criterion->to(device);
	double learning_rate = sets_.lr_init;
	float total_loss = 0.0f;
	int learning_times = 0;	

	for (int epoch = 0; epoch < sets_.epochs; epoch++)
	{
		if ((learning_times % sets_.auto_step) == 0 && epoch != 0)
		{
			std::cout << "change learn from rate: " << learning_rate << " to " << learning_rate * sets_.lr_gamma << std::endl;
			learning_rate *= sets_.lr_gamma;
			static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[0].options()).lr(learning_rate);
		}

		fcn8s.to(device);
		fcn8s.train();

		long long totaltime = 0;

		for (auto &batch : *train_loader)
		{
			// {0} get train sample data
			CHECK(batch.data.numel()) << "tensor is empty.";

			torch::Tensor input = batch.data.clone();
			torch::Tensor target = batch.target.clone();

			input = input.to(device).clone();
			target = target.to(device).clone();

			optimizer.zero_grad();

//			std::cout << "forward input size: " << input.sizes() << " " << input.type() << std::endl;
			auto time_start = std::chrono::system_clock::now();
			auto fcns_output = fcn8s.forward(input);
			auto time_end = std::chrono::system_clock::now();

//			std::cout << "fcns_outpu size:" << fcns_output.sizes() << " " << fcns_output.type() << " target: " << target.sizes() << " " << target.type() << std::endl;
			auto loss = criterion(fcns_output, target);
			loss.backward();
			optimizer.step();

			totaltime += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
			total_loss += loss.item().toFloat();
			float avg_loss = total_loss / std::max(1, learning_times);
			if (learning_times % sets_.check_step == 0 && learning_times != 0)
			{
				LOG(INFO) << "epoch:" << epoch << " learn times:" << learning_times << " avg loss:" << avg_loss << " lr:" << learning_rate << " cost:" << totaltime;
				totaltime = 0;
			}
			learning_times++;
		}
	}

	LOG(INFO) << "training is over...";
	torch::serialize::OutputArchive archive;
	fcn8s.save(archive);
	std::string filename = std::filesystem::path(sets_.out_root).append("fcn8s.pt").string();
	archive.save_to(filename);
	LOG(INFO) << "Save the training as: " << filename;
}
