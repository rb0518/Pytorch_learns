#include <string>
#include <vector>
#include <random>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>


#include "myutils.h"
#include "fcntrain.h"

namespace po = boost::program_options;

po::options_description parse_argument() 
{
	po::options_description args("Options", 200, 30);

	args.add_options()

		// (1)  Define for General Parameter
		("help", "LibTorch FCN project.")
		("dataset_root", po::value<std::string>()->default_value("d:\\data\\VOCdevkit\\VOC2012"), "the data store folder name, only use for VOC2012")
		("output_root", po::value<std::string>()->default_value("d:\\data"), "the data store folder name, only use for VOC2012")
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

	FCN_Train::Settings sets;
	sets.str_data_root = vm["dataset_root"].as<std::string>();
	sets.batch_size = vm["batch_size"].as<int>();
	sets.lr_init = vm["lr_init"].as<float>();
	sets.lr_gamma = vm["lr_gamma"].as<float>();
	sets.out_root = vm["outpuy_root"].as<std::string>();
	sets.auto_step = 10;
	sets.check_step = 5;
	sets.epochs = 50;
	
	FCN_Train fcn_train("cuda", sets);
	fcn_train.Run();

	::google::ShutdownGoogleLogging();
	system("PAUSE");
}

