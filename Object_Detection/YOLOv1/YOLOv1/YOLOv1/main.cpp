#include <string>
#include <vector>
#include <random>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>


#include "utils.h"
#include "augmentation.hpp"
#include "transforms.hpp"
#include "networks.hpp"

#include "vocdataloader.hpp"
#include "loss.hpp"

#include "progress.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;

using Optimizer = torch::optim::SGD;
using OptimizerOptions = torch::optim::SGDOptions;


void Set_Options(po::variables_map& vm, int argc, const char* argv[], po::options_description& args, const std::string mode);
template <typename T> void Set_Model_Params(po::variables_map& vm, T& model, const std::string name);

void train(po::variables_map& vm, torch::Device& device, YOLOv1& model, std::vector<transforms_Compose>& transformBB,
	std::vector<transforms_Compose>& transformI, const std::vector<std::string> class_names);
template <typename Optimizer, typename OptimizerOptions> void Update_LR(Optimizer& optimizer, const float lr_init, const float lr_base, 
		const float lr_decay1, const float lr_decay2, const size_t epoch, const float burnin_base, const float burnin_exp = 4.0);


po::options_description parse_argument() {
	po::options_description args("Options", 200, 30);

	args.add_options()

		// (1)  Define for General Parameter
		("help", "produce help message")
		("data_root", po::value<std::string>()->default_value("D:\\data\\VOCdevkit"))
		("dataset", po::value<std::string>()->default_value("VOC2012"), "dataset name")
		("class_list", po::value<std::string>()->default_value("VOC2012.txt"), "file name in which class names are listed")
		("class_num", po::value<size_t>()->default_value(20), "total classes")
		("size", po::value<size_t>()->default_value(448), "image width and height")
		("prob_thresh", po::value<float>()->default_value(0.1f), "threshold of simultaneous probability with confidence and class score")
		("nms_thresh", po::value<float>()->default_value(0.5f), "threshold of IoU between bounding boxes in Non-Maximum Suppression")
		("num_channels", po::value<size_t>()->default_value(3), "input image channel: RGB=3, mono=1")
		("num_grid", po::value<size_t>()->default_value(7), "the number of grid")
		("num_bbox", po::value<size_t>()->default_value(2), "the number of bounding box in each grid")
		("gpu_id", po::value<int>()->default_value(0), "cuda device : 'x=-1' is CPU device")
		("seed_random", po::value<bool>()->default_value(false), "whether to make the seed of random number in a random")
		("seed", po::value<int>()->default_value(0), "seed of random number")
	
		// (2) Define for Training
		("train", po::value<bool>()->default_value(false), "training mode on/off")
		("train_in_dir", po::value<std::string>()->default_value("trainI"), "training input image directory : ./datasets/<dataset>/<train_in_dir>/<image files>")
		("train_out_dir", po::value<std::string>()->default_value("trainO"), "training output image directory : ./datasets/<dataset>/<train_out_dir>/<annotation files>")
		("epochs", po::value<size_t>()->default_value(200), "training total epoch")
		("batch_size", po::value<size_t>()->default_value(2), "training batch size")
		("train_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning")
		("save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer")
		/*************************** Data Augmentation ***************************/
		("augmentation", po::value<bool>()->default_value(true), "data augmentation mode on/off")
		("jitter", po::value<double>()->default_value(0.2), "the distortion of image shifting")
		("flip_rate", po::value<double>()->default_value(0.5), "frequency to flip")
		("scale_rate", po::value<double>()->default_value(0.5), "frequency to scale")
		("blur_rate", po::value<double>()->default_value(0.5), "frequency to blur")
		("brightness_rate", po::value<double>()->default_value(0.5), "frequency to change brightness")
		("hue_rate", po::value<double>()->default_value(0.5), "frequency to change hue")
		("saturation_rate", po::value<double>()->default_value(0.5), "frequency to change saturation")
		("shift_rate", po::value<double>()->default_value(0.5), "frequency to shift")
		("crop_rate", po::value<double>()->default_value(0.5), "frequency to crop")

		// (7) Define for Network Parameter
		("lr_init", po::value<float>()->default_value(1e-3), "learning rate in the initial iteration")
		("lr_base", po::value<float>()->default_value(1e-2), "learning rate in the base iteration")
		("lr_decay1", po::value<float>()->default_value(1e-3), "learning rate in the decay 1 iteration")
		("lr_decay2", po::value<float>()->default_value(1e-4), "learning rate in the decay 2 iteration")
		("momentum", po::value<float>()->default_value(0.9), "momentum in SGD of optimizer method")
		("weight_decay", po::value<float>()->default_value(5e-4), "weight decay in SGD of optimizer method")
		("Lambda_coord", po::value<float>()->default_value(5.0), "the multiple of coordinate term")
		("Lambda_object", po::value<float>()->default_value(1.0), "the multiple of object confidence term")
		("Lambda_noobject", po::value<float>()->default_value(0.5), "the multiple of no object confidence term")
		("Lambda_class", po::value<float>()->default_value(1.0), "the multiple of class term")


	// End Processing
	;

	return args;
}


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

	// (2) Set device
	int gpu_id = vm["gpu_id"].as<int>();
	torch::Device device = torch::Device(torch::kCPU);
	if (gpu_id >= 0 && torch::cuda::is_available()) 
	{
		device = torch::Device(torch::kCUDA, gpu_id);
		LOG(INFO) << "Will run with GPU: " << gpu_id;
	}
	else
	{
		LOG(INFO) << "Will run with CPU";
	}

	// (3) Set Seed
	if (vm["seed_random"].as<bool>()) {
		std::random_device rd;
		std::srand(rd());
		torch::manual_seed(std::rand());
		torch::globalContext().setDeterministicCuDNN(false);
		torch::globalContext().setBenchmarkCuDNN(true);
	}
	else {
		std::srand(vm["seed"].as<int>());
		torch::manual_seed(std::rand());
		torch::globalContext().setDeterministicCuDNN(true);
		torch::globalContext().setBenchmarkCuDNN(false);
	}

	// (4) Set Transforms
	std::vector<transforms_Compose> transformBB;
	if (vm["augmentation"].as<bool>()) {
		transformBB.push_back(
			YOLOAugmentation(  // apply "flip", "scale", "blur", "brightness", "hue", "saturation", "shift", "crop"
				vm["jitter"].as<double>(),
				vm["flip_rate"].as<double>(),
				vm["scale_rate"].as<double>(),
				vm["blur_rate"].as<double>(),
				vm["brightness_rate"].as<double>(),
				vm["hue_rate"].as<double>(),
				vm["saturation_rate"].as<double>(),
				vm["shift_rate"].as<double>(),
				vm["crop_rate"].as<double>()
			)
		);
	}
	std::vector<transforms_Compose> transformI{
		transforms_Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
		transforms_ToTensor()                                                                             // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
	};
	if (vm["num_channels"].as<size_t>() == 1) {
		transformI.insert(transformI.begin(), transforms_Grayscale(1));
	}
	std::vector<transforms_Compose> transformD{
		transforms_ToTensor()  // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
	};
	if (vm["num_channels"].as<size_t>() == 1) {
		transformD.insert(transformD.begin(), transforms_Grayscale(1));
	}

	// (5) Define Network
	YOLOv1 model(vm);
	model->to(device);

	// (6) Make Directories
	boost::filesystem::path dir = boost::filesystem::path("../../checkpoints/" + vm["dataset"].as<std::string>());
	LOG(INFO) << "create dir: " << dir.string();
	boost::filesystem::create_directories(boost::filesystem::path(dir));

	// (7) Save Model Parameters
	Set_Model_Params(vm, model, "YOLOv1");

	std::vector<std::string> class_names;
	read_lines_from_file(vm["data_root"].as<std::string>() + "\\" + vm["class_list"].as <std::string>(), class_names);

	if (vm["train"].as<bool>())
	{
		LOG(INFO) << "Set Train options....";
		Set_Options(vm, argc, argv, args, "train");

		train(vm, device, model, transformBB, transformI, class_names);
	}



	::google::ShutdownGoogleLogging();
	system("PAUSE");
}

// -----------------------------------
// 3. Model Parameters Setting Function
// -----------------------------------
template <typename T>
void Set_Model_Params(po::variables_map& vm, T& model, const std::string name) 
{

	// (1) Make Directory
	boost::filesystem::path dir = boost::filesystem::path("../../checkpoints/" + vm["dataset"].as<std::string>() + "/model_params/");
	boost::filesystem::create_directories(dir);

	// (2.1) File Open
	std::string fname = dir.string() + name + ".txt";
	std::ofstream ofs(fname);

	// (2.2) Calculation of Parameters
	size_t num_params = 0;
	for (auto param : model->parameters()) 
	{
		num_params += param.numel();
	}
	ofs << "Total number of parameters : " << (float)num_params / 1e6f << "M" << std::endl << std::endl;
	ofs << model << std::endl;

	// (2.3) File Close
	ofs.close();

	// End Processing
	return;

}

// -----------------------------------
// 5. Options Setting Function
// -----------------------------------
void Set_Options(po::variables_map& vm, int argc, const char* argv[], po::options_description& args, const std::string mode) 
{

	// (1) Make Directory
	boost::filesystem::path dir = boost::filesystem::path("../../checkpoints/" + vm["dataset"].as<std::string>() + "/options/");
	boost::filesystem::create_directories(dir);

	// (2) Terminal Output
	std::cout << "--------------------------------------------" << std::endl;
	std::cout << args << std::endl;
	std::cout << "--------------------------------------------" << std::endl;

	// (3.1) File Open
	std::string fname = dir.string() + mode + ".txt";
	std::ofstream ofs(fname);

	// (3.2) Arguments Output
	ofs << "--------------------------------------------" << std::endl;
	ofs << "Command Line Arguments:" << std::endl;
	for (int i = 1; i < argc; i++) {
		if (i % 2 == 1) {
			ofs << "  " << argv[i] << '\t' << std::flush;
		}
		else {
			ofs << argv[i] << std::endl;
		}
	}
	ofs << "--------------------------------------------" << std::endl;
	ofs << args << std::endl;
	ofs << "--------------------------------------------" << std::endl << std::endl;

	// (3.3) File Close
	ofs.close();

	// End Processing
	return;

}

void train(po::variables_map& vm, torch::Device& device, YOLOv1& model, std::vector<transforms_Compose>& transformBB,
	std::vector<transforms_Compose>& transformI, const std::vector<std::string> class_names)
{
	constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
	constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset
	constexpr size_t save_sample_iter = 50;  // the frequency of iteration to save sample images
	constexpr std::string_view extension = "jpg";  // the extension of file name to save sample images
	constexpr std::pair<float, float> output_range = { 0.0, 1.0 };  // range of the value in output images

	//-------------------------------------------
	//  (1) Initialization and Declaration
	//-------------------------------------------
	size_t epoch, iter;
	size_t total_iter;
	size_t start_epoch, total_epoch;
	size_t batch_size;

	float loss_f, loss_coord_xy_f, loss_coord_wh_f, loss_obj_f, loss_noobj_f, loss_class_f;
	float lr_init, lr_base, lr_decay1, lr_decay2;		// learn ratio
	std::string data, data_out;
	std::string buff, latest;
	std::string checkpoint_dir, save_image_dir, path;
	std::string input_dir, output_dir;
	std::string valid_input_dir, valid_output_dir;
	std::stringstream ss;
	std::ifstream infoi;
	std::ofstream ofs, init, infoo;
	std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>,
		std::vector<std::string>> mini_batch;

	torch::Tensor loss, image, output;
	torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
	cv::Mat sample;

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> losses;
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;

	std::vector<transforms_Compose> null;
	

	progress::display* show_progress;
	progress::irregular irreg_progress;

	
	//-----------------------------------------
	// Preparation
	//-----------------------------------------
	DataLoader::VOCDataLoader dataloader(vm["data_root"].as<std::string>(), DataLoader::VOCDataLoader::VOC_2012, 
										DataLoader::VOCDataLoader::_TRAIN_,
										transformBB, transformI);

	// Set Optimizer Method
	auto optimzer = torch::optim::SGD(model->parameters(), 
		torch::optim::SGDOptions(vm["lr_init"].as<float>()).momentum(vm["momentum"].as<float>()).weight_decay(vm["weight_decay"].as<float>()));

	// Set Loss Function
	auto criterion = Loss((int64_t)vm["class_num"].as<size_t>(), (int64_t)vm["num_grid"].as<size_t>(), (int64_t)vm["num_bbox"].as<size_t>());

	total_iter = dataloader.get_samples_count();
	total_epoch = vm["epochs"].as<size_t>();
	lr_init = vm["lr_init"].as<float>();		// learn ratio init parameter.
	lr_base = vm["lr_base"].as<float>();
	lr_decay1 = vm["lr_decay1"].as<float>();
	lr_decay2 = vm["lr_decay2"].as<float>();

	batch_size = vm["batch_size"].as<size_t>();
	start_epoch = 0;

	irreg_progress.restart(start_epoch, total_epoch);
	for (epoch = 0; epoch < total_epoch; epoch++)
	{
		model->train();
		LOG(INFO) << "epoch: " << epoch << " / " << total_epoch;
		size_t samplecount = dataloader.get_samples_count();
		size_t startindex = 0;
		show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{ epoch, total_epoch }, /*loss_=*/{ "coord_xy", "coord_wh", "conf_o", "conf_x", "class" });


		while (dataloader.loadbatch(startindex, batch_size, mini_batch))
		{
//			LOG(INFO) << "startindex: " << startindex << "  batch size: " << batch_size << " count: " << samplecount;
			// Update Learning Rate
			Update_LR<Optimizer, OptimizerOptions>(optimzer, lr_init, lr_base, lr_decay1, lr_decay2, epoch, 1.0, 1.0);
		

			// YOLOv1 Training Phase
			image = std::get<0>(mini_batch).to(device);		// N,C,H,W
			label = std::get<1>(mini_batch);
			output = model->forward(image);

//			losses = criterion(output, label);
// 			loss_coord_xy = std::get<0>(losses) * vm["Lambda_coord"].as<float>();
// 			loss_coord_wh = std::get<1>(losses) * vm["Lambda_coord"].as<float>();
// 			loss_obj = std::get<2>(losses) * vm["Lambda_object"].as<float>();
// 			loss_noobj = std::get<3>(losses) * vm["Lambda_noobject"].as<float>();
// 			loss_class = std::get<4>(losses) * vm["Lambda_class"].as<float>();

			loss = loss_coord_xy + loss_coord_wh + loss_obj + loss_noobj + loss_class;
			optimzer.zero_grad();
/*			loss.backward();*/
			optimzer.step();

			// -----------------------------------
			// c3. Record Loss (iteration)
			// -----------------------------------
			show_progress->increment(/*loss_value=*/{ loss_coord_xy.item<float>(), loss_coord_wh.item<float>(), 
											loss_obj.item<float>(), loss_noobj.item<float>(), loss_class.item<float>() });
		
// 			ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
// 			ofs << "coord_xy:" << loss_coord_xy.item<float>() << "(ave:" << show_progress->get_ave(0) << ") " << std::flush;
// 			ofs << "coord_wh:" << loss_coord_wh.item<float>() << "(ave:" << show_progress->get_ave(1) << ") " << std::flush;
// 			ofs << "conf_o:" << loss_obj.item<float>() << "(ave:" << show_progress->get_ave(2) << ") " << std::flush;
// 			ofs << "conf_x:" << loss_noobj.item<float>() << "(ave:" << show_progress->get_ave(3) << ") " << std::flush;
// 			ofs << "class:" << loss_class.item<float>() << "(ave:" << show_progress->get_ave(4) << ')' << std::endl;

			// -----------------------------------
			// c4. Save Sample Images
			// -----------------------------------
			iter = show_progress->get_iters();
			if (iter % save_sample_iter == 1) 
			{
				ss.str(""); ss.clear(std::stringstream::goodbit);
// 				ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
// 				detect_result = detector(output[0]);
// 				sample = visualizer::draw_detections_des(image[0].detach(), { std::get<0>(detect_result), std::get<1>(detect_result) }, std::get<2>(detect_result), class_names, label_palette, /*range=*/output_range);
// 				cv::imwrite(ss.str(), sample);
			}
		}

		// -----------------------------------
		// b2. Record Loss (epoch)
		// -----------------------------------
		loss_f = show_progress->get_ave(0) + show_progress->get_ave(1) + show_progress->get_ave(2) + show_progress->get_ave(3) + show_progress->get_ave(4);
		loss_coord_xy_f = show_progress->get_ave(0);
		loss_coord_wh_f = show_progress->get_ave(1);
		loss_obj_f = show_progress->get_ave(2);
		loss_noobj_f = show_progress->get_ave(3);
		loss_class_f = show_progress->get_ave(4);

		// -----------------------------------
		// b6. Show Elapsed Time
		// -----------------------------------
		if (epoch % 10 == 0) {

			// -----------------------------------
			// c1. Get Output String
			// -----------------------------------
			ss.str(""); ss.clear(std::stringstream::goodbit);
			irreg_progress.nab(epoch);
			ss << "elapsed = " << irreg_progress.get_elap() << '(' << irreg_progress.get_sec_per() << "sec/epoch)   ";
			ss << "remaining = " << irreg_progress.get_rem() << "   ";
			ss << "now = " << irreg_progress.get_date() << "   ";
			ss << "finish = " << irreg_progress.get_date_fin();
			std::cout << ss.str() << std::endl;
			//date_out = ss.str();
		}
	}
}

template <typename Optimizer, typename OptimizerOptions>
void Update_LR(Optimizer& optimizer, const float lr_init, const float lr_base, const float lr_decay1, const float lr_decay2,
	const size_t epoch, const float burnin_base, const float burnin_exp/* = 4.0*/)
{
	float lr;
	if (epoch == 1) {
		lr = lr_init/* + (lr_base - lr_init) * std::pow(burnin_base, burnin_exp)*/;
	}
	else if (epoch == 2) {
		lr = lr_base;
	}
	else if (epoch == 76) {
		lr = lr_decay1;
	}
	else if (epoch == 106) {
		lr = lr_decay2;
	}
	else
	{
		return;
	}

	for (auto& param_group : optimizer.param_groups()) {
		if (param_group.has_options()) {
			auto& options = (torch::optim::SGDOptions&)(param_group.options());
			options.lr(lr);
		}
	}

	return;
}
