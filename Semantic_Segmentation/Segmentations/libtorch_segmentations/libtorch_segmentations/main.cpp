#include <string>
#include <vector>
#include <random>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>


#include "myutils.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;



po::options_description parse_argument() {
	po::options_description args("Options", 200, 30);

	args.add_options()

		// (1)  Define for General Parameter
		("help", "give your own message")


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



	::google::ShutdownGoogleLogging();
	system("PAUSE");
}