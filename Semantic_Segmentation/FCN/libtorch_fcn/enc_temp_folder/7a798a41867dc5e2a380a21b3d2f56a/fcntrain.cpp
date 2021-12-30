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
		torch::data::DataLoaderOptions().batch_size(1).workers(1));

	FCN8s fcn8s(sets_.num_class);

#if 1
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
			static_cast<torch::optim::RMSpropOptions&>(optimizer.param_groups()[0].options()).lr(learning_rate);
		}

		train(epoch, fcn8s, device, train_loader, optimizer, criterion, learning_times);
		val(epoch, fcn8s, device, val_loader);

		if ((epoch % 5) == 0 && epoch > 5)
		{
			torch::serialize::OutputArchive archive;
			fcn8s.save(archive);
			std::string filename = sets_.out_root + "\\fcn8s_tmp.pt";
			archive.save_to(filename);
			LOG(INFO) << "Save the training as: " << filename;
		}
	}

	LOG(INFO) << "training is over...";
	torch::serialize::OutputArchive archive;
	fcn8s.save(archive);
	std::string filename = std::filesystem::path(sets_.out_root).append("fcn8s.pt").string();
	archive.save_to(filename);
	LOG(INFO) << "Save the training as: " << filename;
}

template <typename DataLoader>
void FCN_Train::train(int epoch, FCN8s& fcn8s, torch::DeviceType devicetype, DataLoader& dataloader,
	torch::optim::Optimizer& optimizer, torch::nn::BCEWithLogitsLoss& criterion, int& learn_times)
{
	fcn8s.to(devicetype);
	fcn8s.train();

	long long totaltime = 0;
	float total_loss = 0.0f;
/*	int ntesttimes = 0;*/
	for (auto& batch : *dataloader)
	{
// 		ntesttimes++;
// 		if (ntesttimes > 5)
// 			break;
		// {0} get train sample data
		CHECK(batch.data.numel()) << "tensor is empty.";

		torch::Tensor input = batch.data.clone();
		torch::Tensor target = batch.target.clone();

		input = input.to(devicetype).clone();
		target = target.to(devicetype).clone();

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
		float avg_loss = total_loss / std::max(1, learn_times);
		if (learn_times % sets_.check_step == 0 && learn_times != 0)
		{
			LOG(INFO) << "epoch:" << epoch << " learn times:" << learn_times << " avg loss:" << avg_loss  << " cost:" << totaltime;
			totaltime = 0;
		}
		learn_times++;
	}

}

std::vector<float> iou(torch::Tensor pred, torch::Tensor target, int num_class)
{
	pred = pred.to(torch::kCPU);
	target = target.to(torch::kCPU);
	pred = pred.toType(torch::kInt32);
	target = target.toType(torch::kInt32);
	int h = target.sizes()[0];  // target {h , w} , at(x, y) value is class index
	int w = target.sizes()[1];
	std::cout << "h " << h << " w " << w << std::endl;
	torch::Tensor pred_inds = torch::zeros({ h, w }).toType(torch::kInt32);
	torch::Tensor target_inds = torch::zeros({ h, w }).toType(torch::kInt32);

	//��tensorת����int array
	pred = pred.squeeze(0);
	std::cout << "pred:" << pred.sizes() << std::endl;
	//��pred��targetȫ��ת����int ����,��Ϊtensor�ĵ�������ǳ�����
	int* pred_array = (int*)pred.data_ptr();
	int* target_array = (int*)target.data_ptr();
	int intersection = 0;	//����
	int union_ = 0;			//����

	std::vector<float> ious;	//����ͼƬ��n_class��ious
	//auto tm_start = std::chrono::system_clock::now();
	for (int cls = 0; cls < num_class; cls++)
	{
		intersection = 0;
		//��������Ϊ���µ�������Ŀ�ͷ,�������Ϊ++������ָ��������
		pred_array = (int*)pred.data_ptr();
		target_array = (int*)target.data_ptr();
		//��������Ϊ��0,ÿ��ѭ������һ���������
		pred_inds.zero_();
		target_inds.zero_();
		//ͬ����ת��pred_indsΪint��������,ÿ��ѭ����Ҫ��������ָ����ʼ
		int* pred_inds_array = (int*)pred_inds.data_ptr();
		int* target_inds_array = (int*)target_inds.data_ptr();
		//��ʼ����pred_arrayҲ������������tensor�е���cls����,����Ϊ1
		//ͬ����target_arrayҲ���Ǳ�ǩ,δ����one-hot�ı�ǩ,��CamVidUtils���Ѿ�������Labeled�ļ�����
		for (int j = 0; j < h; j++)
		{
			for (int k = 0; k < w; k++, pred_inds_array++, target_inds_array++)
			{
				if (*pred_array++ == cls)
					*pred_inds_array = 1;
				if (*target_array++ == cls)
					*target_inds_array = 1;
			}
		}
		//���°�ָ����ʼ���û���
		pred_inds_array = (int*)pred_inds.data_ptr();
		target_inds_array = (int*)target_inds.data_ptr();
		//�����ļ���,����ǩ�е���1�Ķ�Ӧ����ֵ���������еĶ�Ӧ����ֵ�Ƕ���,�����ֵ�����ۼ�,��Ϊ����
		//Ҳ�������������ͼƬ�ڵ�ǰ����ıȶ��ж������ص�ķ���ͱ�ǩ��ǰ���еķ�����һ����
		for (int k = 0; k < h * w; k++, target_inds_array++, pred_inds_array++)
		{
			if (*target_inds_array == 1)
			{
				intersection += *pred_inds_array;
			}
		}
		//printf("���� intersection = %d\n", intersection);
		//�󲢼�
		union_ = (pred_inds.sum().item<int>() + target_inds.sum().item<int>() - intersection);

		//�������Ϊ0,��ǰ�ಢû��ground truth
		if (union_ == 0)
			ious.push_back(std::nanf("nan"));
		else
			ious.push_back(float(intersection) / std::max(union_, 1));	//��iou,��ÿ�����iou���뵽ious����Ϊ��������
	}
	//auto tm_end = std::chrono::system_clock::now();
	//printf("cost:{%lld msec}\n", std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count());

	return ious;
}

float pixel_acc(torch::Tensor pred, torch::Tensor target)
{
	pred = pred.to(torch::kCPU);
	target = target.to(torch::kCPU);
	pred = pred.toType(torch::kInt32);
	target = target.toType(torch::kInt32);

	int correct = 0;
	int total = 0;

	pred = pred.squeeze(0);

	int h = pred.sizes()[0];
	int w = pred.sizes()[1];

	int* pred_array = (int*)pred.data_ptr();
	int* target_array = (int*)target.data_ptr();

	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			//printf("pred class : %d target class : %d\n", *pred_array, *target_array);
			if (*pred_array++ == *target_array++)
				correct++;
			total++;
		}
	}
	//printf("correct : %d total : %d\n", correct, total);
	return (float)correct / (float)total;
}


template <typename DataLoader>
void FCN_Train::val(int epoch, FCN8s& fcn8s, torch::DeviceType devicetype, DataLoader& dataloader)
{
	fcn8s.to(devicetype);
	fcn8s.train(false);
	fcn8s.eval();

	std::vector<float> ious;
	long long accumulationCost = 0;

	float totalMeanIoU = .0;
	float totalPixel_accs = .0;
	int N = 0;

	for (auto& batch : *dataloader)
	{
		N++;
		if (!batch.data.numel())
		{
			std::cout << "tensor is empty!" << std::endl;
			continue;
		}
		torch::Tensor input = batch.data.clone()/*data.unsqueeze(0)*/;
		input = input.to(devicetype);
		torch::Tensor target = batch.target[0].to(devicetype);

		auto tm_start = std::chrono::system_clock::now();
		auto fcns_output = fcn8s.forward(input);
		auto tm_end = std::chrono::system_clock::now();
		//std::cout << fcns_output.sizes() << " " << target.sizes() << std::endl;
		accumulationCost += std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
		int N = fcns_output.sizes()[0];
		int h = fcns_output.sizes()[2];
		int w = fcns_output.sizes()[3];

		//output���Ϊ{1, 32, h, w} => {-1, num_class} ������ÿ������һ��,������ص��32�����
		//Ȼ��argmax(1)��ÿ�����ֵ�±�,���������ǰ���ص�������һ��,ע�����ｫ���ص��ֵ����˷���ֵ
		//������µ�����{h,w}����
		torch::Tensor pred = fcns_output.permute({ 0, 2, 3, 1 }).reshape({ -1, sets_.num_class }).argmax(1).reshape({ N, h, w });
		//iou����������ÿ��ͼƬ(�������batch��)��n_class�еķ���iou
		//����,[��һ��ͼƬ[0���iou, 1���iou, .... n_class-1��iou],�ڶ���ͼƬ[0��iou, 1��iou, ....n_class-1��iou]....batch��]
		//std::cout << pred.sizes() << std::endl;
		ious = iou(pred, target, sets_.num_class);
		//��Ϊ�������еĽ����Լ�gpu�ڴ������,�ڽ�������ָ��ʱ�򾭳�ʹ��batch = 1,����������ֱ���ۼ�ious�е�ֵ(vector<float>)����һ��mean����
		//����,�����������ͷ����iou

		//ע�����ص�ľ�ȷaccs��IoU�ǲ�һ���ĺ����߶�,accs�ܴ�������,IoU����һ����,IoU��������ͼ����غϳ̶�,accs�����ص����ȳ̶�
		//���仰��,ͼƬ��������ڳ����͵�·,�������ص㶼һ������·�Ծɴ���������,�ͻ����accs�ܴ�,Ȼ������ͼƬ�غϳ̶��Ծɺܵ�
		float meanIoU = .0;
		float pixel_accs = .0;
		std::vector<float>::iterator it;
		for (it = ious.begin(); it != ious.end(); ++it)
		{
			if (std::isnan(*it))
				continue;
			else
				meanIoU += (*it);
		}
		meanIoU /= sets_.num_class;
		totalMeanIoU += meanIoU;
		pixel_accs = pixel_acc(pred, target);
		totalPixel_accs += pixel_accs;
		std::cout << "meanIoU: " << meanIoU << " pixel_accs: " << pixel_accs << std::endl;
	}
	totalMeanIoU /= N;
	totalPixel_accs /= N;
	std::cout << "epoch:" << epoch << " pix_acc:" << totalPixel_accs << " meanIoU:" << totalMeanIoU << std::endl;
}
