## 参考代码
原始代码见[overfitover
](https://github.com/overfitover/fcn_pytorch)的代码.

## 数据集
数据集还是VOC Pascal，存放目录为``d:\data\VOCdevtik``

## 代码调试
### 1. loss.py 命名冲突，修改为myloss.py

### 2. 运行报错：
```
total_loss += loss.data[0] 
IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
```
pytorch版本更新，原来只出警告的语句现在报错了，按提示替换代码
```python
#train.py
total_loss += loss.item()

```

## train 代码输出
```
Python FCN program select device: cpu
Dataset root:d:/data/VOCdevkit/VOC2012
load data....
D:\Python39\lib\site-packages\torch\nn\modules\loss.py:217: UserWarning: NLLLoss2d has been deprecated. Please use NLLLoss instead as a drop-in replacement and see https://pytorch.org/docs/master/nn.html#torch.nn.NLLLoss for more details.
  warnings.warn("NLLLoss2d has been deprecated. "
D:\Python39\lib\site-packages\torch\nn\_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
  warnings.warn(warning.format(ret))
D:\Pytorch_learns\Semantic_Segmentation\FCN\pycodes\myloss.py:45: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return self.nll_loss(F.log_softmax(inputs), targets)
train epoch [0/300], iter[0/732], lr 0.0010000, aver_loss 1.75955
train epoch [0/300], iter[20/732], lr 0.0010000, aver_loss 1.57603
train epoch [0/300], iter[40/732], lr 0.0010000, aver_loss 1.51937
train epoch [0/300], iter[60/732], lr 0.0010000, aver_loss 1.48019
train epoch [0/300], iter[80/732], lr 0.0010000, aver_loss 1.44557
train epoch [0/300], iter[100/732], lr 0.0010000, aver_loss 1.41061
train epoch [0/300], iter[120/732], lr 0.0010000, aver_loss 1.37744
```
