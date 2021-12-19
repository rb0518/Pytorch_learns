#pragma once

#include <tuple>
#include <vector>
#include <torch/torch.h>

// -------------------
// class{Loss}
// -------------------
class Loss {
private:
	int64_t class_num, ng, nb;
	std::tuple<torch::Tensor, torch::Tensor> build_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>>& target);
	torch::Tensor rescale(torch::Tensor& BBs);
	torch::Tensor compute_iou(torch::Tensor& BBs1, torch::Tensor& BBs2);
public:
	Loss() {}
	Loss(const int64_t class_num_, const int64_t ng_, const int64_t nb_);
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> operator()(torch::Tensor& input, std::vector<std::tuple<torch::Tensor, torch::Tensor>>& target);
};


