#include <string>
#include <vector>
#include <random>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>


#include "myutils.h"
#include "VOCSegDataset.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;



po::options_description parse_argument() 
{
	po::options_description args("Options", 200, 30);

	args.add_options()

		// (1)  Define for General Parameter
		("help", "LibTorch FCN project.")
		("dataset_root", po::value<std::string>()->default_value("d:\\data\\VOCdevkit\\VOC2012"), "the data store folder name, only use for VOC2012")
		("class_num", po::value<int>()->default_value(21), "number for classification")
		
		// (2) Define for Training
		("batch_size", po::value<int>()->default_value(5), "training batch size")
		("num_epochs", po::value<int>()->default_value(30), "total training times")
		("lr_init", po::value<float>()->default_value(0.01), "the init value for learning rate")
		("lr_gamma", po::value<float>()->default_value(0.5), "decay learning rate by a factor of gamma")




	// End Processing
	;

	return args;
}

#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{

	// (1) Extract Arguments
	po::options_description args = parse_argument();
	po::variables_map vm{};
	po::store(po::parse_command_line(argc, argv, args), vm);
	po::notify(vm);
	if (vm.empty() || vm.count("help")) {
		std::cout << args << std::endl;
		return 1;
	}
 	::google::InitGoogleLogging(argv[0]);
 	FLAGS_alsologtostderr = true;

// 	cv::Mat img = cv::imread("D:\\data\\VOCdevkit\\VOC2012\\SegmentationClass\\2007_000032.png");
// 	std::cout << img.channels() << std::endl;
// 	cv::imshow("test", img);
// 	cv::waitKey();
// 	cv::destroyAllWindows();

	auto train_dataset = VOCSegDataset(vm["dataset_root"].as<std::string>(), "train", 21).map(torch::data::transforms::Stack<>());
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), 
		torch::data::DataLoaderOptions().batch_size(vm["batch_size"].as<int>()).workers(5));

	::google::ShutdownGoogleLogging();
	system("PAUSE");
}