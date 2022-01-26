#include <string>
#include <vector>
#include <random>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>


#include "myutils.h"

#include "UNet.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;

using Optimizer = torch::optim::SGD;


po::options_description parse_argument() {
	po::options_description args("Options", 200, 30);

	args.add_options()

		// (1)  Define for General Parameter
		("help", "this project collect common semantic segmentation codes")
		("use_gpu", po::value<bool>()->default_value(true), "use GPU to run this program, default is true.")
		("run_mode", po::value<std::string>()->default_value("train"), "run mode: train detect")
		("module", po::value<std::string>()->default_value("UNet"), "select which architecture: FCN, PSPNet")



	// End Processing
	;

	return args;
}

#include "util.h"
#include "Segmentor.h"
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

	Segmentor<UNet> segmentor;
	segmentor.Initialize(0, 512, 512, { "backgroud", "persion" },
		"resnet34", "D:\\Pytorch_learns\\Semantic_Segmentation\\utils\\resetnet34.pt");

	trainTricks tricks;
	tricks.horizontal_flip_prob = 0.5f;
	tricks.vertical_flip_prob = 0.5f;
	tricks.scale_rotate_prob = 0.3f;
	tricks.decay_epochs = { 40, 80 };
	tricks.freeze_epochs = 8;

	segmentor.SetTrainTricks(tricks);
	segmentor.Train(0.0003f, 300, 4, "D:\\LibtorchSegmentation-main\\voc_person_seg",
		".jpg", "D:\\LibtorchSegmentation-main\\weights\\segmentor.pt");


	::google::ShutdownGoogleLogging();
	system("PAUSE");
}