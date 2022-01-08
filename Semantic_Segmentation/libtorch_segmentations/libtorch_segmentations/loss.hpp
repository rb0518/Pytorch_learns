#ifndef _LOSS_H_
#define _LOSS_H_
#include <torch/torch.h>

//prediction [NCHW], a tensor after softmax activation at C dim
//target [N1HW], a tensor refer to label
//num_class: int, equal to C, refer to class numbers, including background
torch::Tensor DiceLoss(torch::Tensor prediction, torch::Tensor target, int num_class);

//prediction [NCHW], target [NHW]
torch::Tensor CELoss(torch::Tensor prediction, torch::Tensor target);

#endif