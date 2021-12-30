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

	//将tensor转换成int array
	pred = pred.squeeze(0);
	std::cout << "pred:" << pred.sizes() << std::endl;
	//将pred和target全部转换成int 数组,因为tensor的迭代计算非常缓慢
	int* pred_array = (int*)pred.data_ptr();
	int* target_array = (int*)target.data_ptr();
	int intersection = 0;	//并集
	int union_ = 0;			//交集

	std::vector<float> ious;	//单张图片的n_class的ious
	//auto tm_start = std::chrono::system_clock::now();
	for (int cls = 0; cls < num_class; cls++)
	{
		intersection = 0;
		//以下两行为重新调整数组的开头,否则会因为++而导致指针往后走
		pred_array = (int*)pred.data_ptr();
		target_array = (int*)target.data_ptr();
		//以下两行为清0,每次循环计算一个分类计数
		pred_inds.zero_();
		target_inds.zero_();
		//同样的转换pred_inds为int类型数组,每次循环需要重新设置指针起始
		int* pred_inds_array = (int*)pred_inds.data_ptr();
		int* target_inds_array = (int*)target_inds.data_ptr();
		//开始计算pred_array也就是推理结果的tensor中等于cls计数,设置为1
		//同样的target_array也就是标签,未经过one-hot的标签,在CamVidUtils中已经保存在Labeled文件夹中
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
		//重新把指针起始设置回来
		pred_inds_array = (int*)pred_inds.data_ptr();
		target_inds_array = (int*)target_inds.data_ptr();
		//交集的计算,即标签中等于1的对应像素值和推理结果中的对应像素值是多少,对这个值进行累加,即为并集
		//也就是推理出来的图片在当前分类的比对中多少像素点的分类和标签当前类中的分类是一样的
		for (int k = 0; k < h * w; k++, target_inds_array++, pred_inds_array++)
		{
			if (*target_inds_array == 1)
			{
				intersection += *pred_inds_array;
			}
		}
		//printf("交集 intersection = %d\n", intersection);
		//求并集
		union_ = (pred_inds.sum().item<int>() + target_inds.sum().item<int>() - intersection);

		//如果并集为0,当前类并没有ground truth
		if (union_ == 0)
			ious.push_back(std::nanf("nan"));
		else
			ious.push_back(float(intersection) / std::max(union_, 1));	//求iou,将每个类的iou推入到ious中作为函数返回
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

		//output结果为{1, 32, h, w} => {-1, num_class} 即拉成每个像素一行,这个像素点的32类概率
		//然后argmax(1)求每行最大值下标,即是求出当前像素点属于哪一类,注意这里将像素点的值变成了分类值
		//最后重新调整成{h,w}类型
		torch::Tensor pred = fcns_output.permute({ 0, 2, 3, 1 }).reshape({ -1, sets_.num_class }).argmax(1).reshape({ N, h, w });
		//iou函数将返回每张图片(如果是有batch的)在n_class中的分类iou
		//例如,[第一张图片[0类的iou, 1类的iou, .... n_class-1类iou],第二张图片[0类iou, 1类iou, ....n_class-1类iou]....batch张]
		//std::cout << pred.sizes() << std::endl;
		ious = iou(pred, target, sets_.num_class);
		//因为在论文中的建议以及gpu内存的限制,在进行语义分割的时候经常使用batch = 1,因此在这里就直接累加ious中的值(vector<float>)进行一次mean即可
		//至此,求完像素类型分类的iou

		//注意像素点的精确accs和IoU是不一样的衡量尺度,accs很大的情况下,IoU并不一定大,IoU衡量的是图像的重合程度,accs是像素点的相等程度
		//换句话讲,图片中如果存在车辆和道路,车辆像素点都一样而道路仍旧错误的情况下,就会造成accs很大,然而整张图片重合程度仍旧很低
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
