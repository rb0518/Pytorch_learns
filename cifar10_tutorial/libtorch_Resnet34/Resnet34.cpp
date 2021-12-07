#include "Resnet34.h"

void train_resnet_cifar_10(void);

void DataScale(cv::Mat src, cv::Mat& dst, float scale);

void DataShift(cv::Mat src, cv::Mat& dst, float x, float y);

void DataEhance(cv::Mat src, cv::Mat& dst);

void train_resnet_cifar10_test_one_epoch(void);

float test_resnet_cifar_10_after_one_epoch_batch(Resnet34 net, std::vector<cv::Mat> test_img,
	std::vector<uchar> test_label, int batch_size);

void train_resnet_cifar_10(void)
{
	Resnet34 net1(3, 10);
	net1.train();

}

void DataScale(cv::Mat src, cv::Mat& dst, float scale)
{
	int row = src.rows * scale;
	int col = src.cols * scale;

	cv::Mat out;
	cv::resize(src, out, cv::Size(col, row), cv::INTER_CUBIC);

	if (scale < 1.0)
	{
		int row_d = src.rows - row;
		int col_d = src.cols - col;
		int top, bottom, left, right;

		top = row_d / 2;
		bottom = (row_d % 2) ? top + 1 : top;

		left = col_d / 2;
		right = (col_d % 2) ? left + 1 : left;

		cv::copyMakeBorder(out, dst, top, bottom, left, right, cv::BORDER_REPLICATE);
	}
	else
	{
		int x = (rand() % (col - src.cols + 1));
		int y = (rand() % (row - src.rows + 1));
		out(cv::Rect(x, y, src.cols, src.rows)).copyTo(dst);
	}
}

void DataShift(cv::Mat src, cv::Mat& dst, float x, float y)
{
	cv::Size dst_sz = src.size();
	cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);

	t_mat.at<float>(0, 1) = 1;
	t_mat.at<float>(0, 2) = x;
	t_mat.at<float>(1, 1) = 1;
	t_mat.at<float>(1, 2) = y;

	cv::warpAffine(src, dst, t_mat, dst_sz, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
}

void DataEhance(cv::Mat src, cv::Mat& dst)
{
	if (rand() % 2)
	{
		flip(src, dst, 1);
	}
	else
	{
		src.copyTo(dst);
	}

	if (rand() % 2)
	{
		float x = 10 * (rand() / double(RAND_MAX)) - 5;
		float y = 10 * (rand() / double(RAND_MAX)) - 5;

		DataShift(dst, dst, x, y);
	}

	if (rand() % 2)
	{
		float scale = 0.4 * (rand() / double(RAND_MAX)) + 0.8;
		DataScale(dst, dst, scale);
	}
}


// gcn对比度全局归一化
void cal_gcn(cv::Mat& src)
{
	float m = 0;
	float m_2 = 0;

	for (int i = 0; i < src.rows; i++)
	{
		cv::Vec3f* p = src.ptr<cv::Vec3f>(i);
		for (int j = 0; j < src.cols; j++)
		{
			m += (p[j][0] + p[j][1] + p[j][2]);
			m_2 += (p[j][0] * p[j][0] + p[j][1] * p[j][1] + p[j][2] * p[j][2]);
		}
	}

	float total_cnt = src.rows * src.cols * 3.0;
	m /= total_cnt;
	m_2 /= total_cnt;

	float std = sqrt(m_2 - m * m);   //标准差

	for (int i = 0; i < src.rows; i++)
	{
		cv::Vec3f* p = src.ptr<cv::Vec3f>(i);
		for (int j = 0; j < src.cols; j++)
		{
			p[j][0] = (p[j][0] - m) / std::max(float(1e-8), std);
			p[j][1] = (p[j][1] - m) / std::max(float(1e-8), std);
			p[j][2] = (p[j][2] - m) / std::max(float(1e-8), std);
		}
	}
}


#define CIFAR_10_OENFILE_DATANUM 10000
#define CIFAR_10_FILENUM 5
#define CIFAR_10_TOTAL_DATANUM (CIFAR_10_OENFILE_DATANUM*CIFAR_10_FILENUM)
#include <random>
#include <chrono>

#include "CCifar10Loader.h"
#include <vector>


void read_cifar_batch(CCifar10Loader& dataload, std::vector<int> shuffle_idx, int start_idx, int batchsize,
	std::vector<cv::Mat>& img_list, std::vector<uchar>& label_list)
{
	img_list.clear();
	label_list.clear();

	for (int i = 0; i < batchsize; i++)
	{

		cv::Mat img;
		cv::cvtColor(dataload.GetImages().at(shuffle_idx.at(start_idx + i)), img, cv::COLOR_BGR2RGB);

		DataEhance(img, img);
		img.convertTo(img, CV_32F, 1.0 / 255.0);
		cal_gcn(img);

		img_list.push_back(img.clone());
		label_list.push_back(dataload.GetLabels().at(shuffle_idx.at(start_idx + i)));
	}
}

#define DATA_ROOT "D:\\data\\cifar-10-batches-bin"


float test_resnet_cifar_10_after_one_epoch_batch(Resnet34 net, std::vector<cv::Mat> test_img, 
	std::vector<uchar> test_label, int batch_size)
{
	int total_test_items = 0, passed_test_items = 0;
	double total_time = 0.0;

	for (int i = 0; i < test_img.size() / batch_size; i++)
	{

		std::vector<long long> label_list;
		//[batch_size, Height, Width, Channels]
		auto inputs = torch::ones({ batch_size, 32, 32, 3 });
		for (int j = 0; j < batch_size; j++)
		{
			int idx = i * batch_size + j;
			inputs[j] = torch::from_blob(test_img[idx].data, { test_img[idx].rows, test_img[idx].cols, test_img[idx].channels() }, torch::kFloat).clone();
			label_list.push_back((long long)test_label[idx]);
		}
		//将[batch_size, Height, Width, Channels]调整为[batch_size, Channels， Height, Width]
		inputs = inputs.permute({ 0, 3, 1, 2 });

		torch::Tensor labels = torch::tensor(label_list);

//  		inputs = inputs.to(device_type);
//  		labels = labels.to(device_type);

		// 用训练好的网络处理测试数据
		auto outputs = net.forward(inputs);

		// 得到预测值，0 ~ 9
		auto predicted = (torch::max)(outputs, 1);

		// 比较预测结果和实际结果，并更新统计结果
		for (int k = 0; k < batch_size; k++)  //分别统计一个batch中所有样本的预测是否准确
		{
			if (labels[k].item<int>() == std::get<1>(predicted)[k].item<int>())
				passed_test_items++;
		}
		total_test_items += batch_size;

	}

	float acc = passed_test_items * 1.0 / total_test_items;
	std::cout << "total_test_items=" << total_test_items <<" passed_test_items =" << passed_test_items 
		<< " pass rate = " << acc << std::endl;

	return acc;

}


//注意调用该函数之前需要先net1.eval()将网络切换到测试状态
float test_resnet_cifar_10_after_one_epoch(Resnet34 net, std::vector<cv::Mat> test_img, std::vector<uchar> test_label)
{
	int total_test_items = 0, passed_test_items = 0;
	double total_time = 0.0;

	for (int i = 0; i < test_img.size(); i++)
	{
		//将样本、标签转换为Tensor张量
		torch::Tensor inputs = torch::from_blob(test_img[i].data, { 1, test_img[i].channels(), test_img[i].rows, test_img[i].cols }, torch::kFloat);  //1*1*32*32
		torch::Tensor labels = torch::tensor({ (long long)test_label[i] });
		//将样本、标签张量由CPU类型切换到GPU类型，对应于GPU类型的网络
//  		inputs = inputs.to(device_type);
//  		labels = labels.to(device_type);

		// 用训练好的网络处理测试数据
		auto outputs = net.forward(inputs);

		// 得到预测值，0 ~ 9
		auto predicted = (torch::max)(outputs, 1);

		// 比较预测结果和实际结果，并更新统计结果
		if (labels[0].item<int>() == std::get<1>(predicted).item<int>())
			passed_test_items++;

		total_test_items++;
	}

	float acc = passed_test_items * 1.0 / total_test_items;
	printf("total_test_items=%d, passed_test_items=%d, pass rate=%f\n", total_test_items, passed_test_items, acc);

	return acc;   //返回准确率

}

torch::Device Set_Device() 
{
	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available() ) 
	{
	std::cout << "cuda is available: " << torch::cuda::is_available() << "device count: " << torch::cuda::device_count() << std::endl;
		torch::Device device(torch::kCUDA, 0);
		return device;
	}

	return device;
}

void train_resnet_cifar10_test_one_epoch(void)
{
	CCifar10Loader dataloader(DATA_ROOT, CCifar10Loader::_TRAIN_);
	
	torch::Device device_type = Set_Device();

	Resnet34 net1(3, 10);	// 定义Resnet34网络
	net1.train();			// 切换到训练模式
	try
	{
		net1.to(device_type);
	}
	catch (const c10::Error& e)
	{
		std::cerr << "Error\n";
	}


	std::cout << "Start train net...." << std::endl;

	int kNumberOfEpochs = 500;	// 训练epoch为500
	double alpha = 0.01;		// 初始学习率
	int batch_size = 32;

	// 读取标签值 的100*500
	std::vector<cv::Mat> test_img;
	std::vector<uchar> test_label;

	// 读取测试数据 的图像和标签

	// 定义交叉熵误差函数
	auto criterion = torch::nn::CrossEntropyLoss();

	// 定义SGD优化器
	auto optimizer = torch::optim::SGD(net1.parameters(),
		torch::optim::SGDOptions(alpha).momentum(0.9));// .weight_decay(1e-6));  //weight_decay表示L2正则化
	float l1 = 50.0;	 //训练完50个batch之后所得损失函数的均值

	std::vector<int> train_image_shuffle_set;

	for (int epoch = 0; epoch < kNumberOfEpochs; epoch++)
	{
		train_image_shuffle_set.clear();
		for (size_t i = 0; i < CIFAR_10_TOTAL_DATANUM; i++)
		{
			train_image_shuffle_set.push_back(i);
		}
		unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(train_image_shuffle_set.begin(), train_image_shuffle_set.end(), 
			std::default_random_engine(seed1));   //打乱顺序

		auto running_loss = 0.0;
		
		for (int k = 0; k < CIFAR_10_TOTAL_DATANUM / batch_size; k++)	// 每32组数据为一组
		{
			read_cifar_batch(dataloader, train_image_shuffle_set, k * batch_size, batch_size, test_img, test_label);

			auto inputs = torch::ones({ batch_size, 32, 32, 3 });

			for (int b = 0; b < batch_size; b++)
			{
				inputs[b] = torch::from_blob(test_img[b].data, 
					{ test_img[b].rows, test_img[b].cols, test_img[b].channels() }, torch::kFloat).clone();
			}

			//将[batch_size, Height, Width, Channels]调整为[batch_size, Channels， Height, Width]
			inputs = inputs.permute({ 0, 3, 1, 2 });

			torch::Tensor labels = torch::tensor(test_label);

			inputs = inputs.to(device_type);
			labels = labels.to(device_type);

			auto outputs = net1.forward(inputs);
			auto loss = criterion(outputs, labels);

			optimizer.zero_grad();  //清除参数
			loss.backward();   //反向传播
			optimizer.step();  //更新参数	

			running_loss += loss.item().toFloat();
			
			if ((k + 1) % 50 == 0)
			{
				srand((unsigned int)(time(NULL)));

				l1 = running_loss / 50;
				std::cout << "alpha = " << alpha << " loss: " << l1 << std::endl;
				running_loss = 0.0;
			}
		}

		std::cout << "epoch: " << epoch + 1 << " batch_size: " << batch_size << std::endl;
		alpha *= 0.9999;
	}
	
	net1.eval();   //切换测试状态
	//使用测试数据集对训练得到的模型进行验证，并返回准确率
	float acc = test_resnet_cifar_10_after_one_epoch(net1, test_img, test_label);
	std::cout << "acc = " << acc << std::endl;

	remove("D:\\data\\mnist_cifar_10_resnet.pt");
	std::cout <<"Finish training! "  << std::endl;
	torch::serialize::OutputArchive archive;
	net1.save(archive);
	archive.save_to("D:\\data\\mnist_cifar_10_resnet.pt");  //保存训练模型到文件
	std::cout << "Save the training result to mnist_cifar_10_resnet.pt." << std::endl;
}

int main()
{
	train_resnet_cifar10_test_one_epoch();

	system("PAUSE");
	return 0;
}