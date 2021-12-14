#pragma once

#include <tuple>
#include <vector>
#include <torch/torch.h>

#if 0
// -------------------
// namespace{Losses}
// -------------------
namespace Losses {

	// -------------------------------------
	// namespace{Losses} -> class{SSIMLoss}
	// -------------------------------------
	class SSIMLoss {
	private:
		size_t nc;
		size_t window_size;
		float gauss_std;
		float c1_base;
		float c2_base;
		torch::Tensor window;
	public:
		SSIMLoss(){ ; }
		SSIMLoss(const size_t nc_, const torch::Device device, const size_t window_size_ = 11, const float gauss_std_ = 1.5, const float c1_base_ = 0.01, const float c2_base_ = 0.03);
		torch::Tensor Structural_Similarity(torch::Tensor& image1, torch::Tensor& image2);
		torch::Tensor operator()(torch::Tensor& input, torch::Tensor& target);
	};

}
#endif

// -------------------
// class{Loss}
// -------------------
class Loss {
private:
	long int class_num, ng, nb;
	std::tuple<torch::Tensor, torch::Tensor> build_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>>& target);
	torch::Tensor rescale(torch::Tensor& BBs);
	torch::Tensor compute_iou(torch::Tensor& BBs1, torch::Tensor& BBs2);
public:
	Loss() {}
	Loss(const long int class_num_, const long int ng_, const long int nb_);
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> operator()(torch::Tensor& input, std::vector<std::tuple<torch::Tensor, torch::Tensor>>& target);
};


