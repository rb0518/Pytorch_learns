#include "fcntrain.h"
#include <filesystem>
#include <glog/logging.h>

FCN_Train::FCN_Train(const std::string& devicetype, const Settings& sets)
	: set_devicetype_(devicetype), sets_(sets)
{

}

void FCN_Train::Run()
{
	torch::Device device = (set_devicetype_ == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;

	auto train_dataset = VOCSegDataset(sets_.str_data_root, "train", 21).map(torch::data::transforms::Stack<>());
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset),
		torch::data::DataLoaderOptions().batch_size(sets_.batch_size).workers(5));

	auto val_dataset = VOCSegDataset(sets_.str_data_root, "val", 21).map(torch::data::transforms::Stack<>());
	auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset),
		torch::data::DataLoaderOptions().batch_size(sets_.batch_size).workers(5));

	FCN8s fcn8s(sets_.num_class);

	auto criterion = torch::nn::BCEWithLogitsLoss();
	auto optimizer = torch::optim::RMSprop(fcn8s.parameters(),
		torch::optim::RMSpropOptions(double(sets_.lr_init)).momentum(0).weight_decay(10e-5));
	criterion->to(device);
	double learning_rate = sets_.lr_init;
	for (int epoch = 0; epoch < sets_.epochs; epoch++)
	{
		if ((epoch % sets_.auto_step) == 0 && epoch != 0)
		{
			learning_rate *= sets_.lr_gamma;
			static_cast<torch::optim::RMSpropOptions&>(optimizer.param_groups()[0].options()).lr(learning_rate);
		}

		fcn8s.to(device);
		fcn8s.train();

		long long totaltime = 0;
		int learning_times = 0;
		for (auto &batch : *train_loader)
		{
			// {0} get train sample data
			CHECK(batch.data.numel()) << "tensor is empty.";

			torch::Tensor input = batch.data.unsqueeze(0);
			torch::Tensor target = batch.target.unsqueeze(0);
			input = input.to(device);
			target = target.to(device);

			optimizer.zero_grad();	
			auto time_start = std::chrono::system_clock::now();
			auto fcns_output = fcn8s.forward(input);
			auto time_end = std::chrono::system_clock::now();

			auto loss = criterion(fcns_output, target);
			loss.backward();
			optimizer.step();

			totaltime += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

			if ((learning_times++) % sets_.check_step == 0)
			{
				LOG(INFO) << "epoch:" << epoch << "learn times:" << learning_times << "loss:" << loss.item().toFloat() << "cost:" << totaltime;
				totaltime = 0;
			}
		}

	}

	LOG(INFO) << "training is over...";
	torch::serialize::OutputArchive archive;
	fcn8s.save(archive);
	std::string filename = std::filesystem::path(sets_.out_root).append("fcn8s.pt").string();
	archive.save_to(filename);
	LOG(INFO) << "Save the training as: " << filename;
}
