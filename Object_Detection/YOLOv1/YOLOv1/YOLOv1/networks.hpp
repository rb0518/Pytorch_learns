#pragma once

#include <torch/torch.h>
#include <boost/program_options.hpp>


namespace nn = torch::nn;
namespace po = boost::program_options;

void weights_init(nn::Module& m);
void Convolution(nn::Sequential& sq, const size_t in_num_channels, const size_t out_num_channels,
	const size_t ksize, const size_t stride, const size_t pad, 
	const bool BN, const bool LReLU, const bool bias = false);

// -------------------------------------------------
// struct{YOLOv1Impl}(nn::Module)
// -------------------------------------------------
struct YOLOv1Impl : nn::Module {
private:
	long int grid, final_features;
	nn::Sequential features, avgpool, classifier;
public:
	YOLOv1Impl() {}
	YOLOv1Impl(po::variables_map& vm);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(YOLOv1);

