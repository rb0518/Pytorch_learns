import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):
    '''
    这个实现有问题, mmp 误事
    loss doesn't change, loss can not be backward?

    why need change? only net weight need to be change.
    '''
    def __init__(self):
        super(CrossEntropy2d, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=None, size_average=False)  # should size_average=False?  True for average,false for tatal

    def forward(self, out, target):
        n, c, h, w = out.size()         # n:batch_size, c:class
        out = out.view(-1, c)           # (n*h*w, c)
        target = target.view(-1)        # (n*h*w)
        # print('out', out.size(), 'target', target.size())
        loss = self.criterion(out, target)

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=None, size_average=False)

    def forward(self, inputs, targets):
        return self.cross_entropy_loss(inputs, targets)

class CrossEntropyLoss2d(nn.Module):
    """
    亲测有效
    """

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def smooth_l1(deltas, targets, sigma=3.0):
    """
    :param deltas: (tensor) predictions, sized [N,D].
    :param targets: (tensor) targets, sized [N,].
    :param sigma: 3.0
    :return:
    """

    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.min(torch.abs(diffs), 1.0 / sigma2).detach().float()

    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + \
                    torch.mul(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, float)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False, weight=None)
    input = Variable(torch.randn(2, 3, 5))  # (batch_size, C)
    target = Variable(torch.LongTensor(2, 5).random_(3))
    loss = loss_fn(input, target)
    print(input)
    print(target)
    print(loss)











