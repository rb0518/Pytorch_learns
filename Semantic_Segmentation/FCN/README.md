# FCN

In 2015, CVPR published a paper entitled ``Full Convolutional Networks for Semantic Segmentation``,written by Jonathan Long, Evan Shelhamer and Trevor Darrell, which from UC Berkeley. This paper proposed the concept of full convolution, and extended the end-to-end convolution network to the task of semantic segmentation for the first time. Then there were many network structures based on FCN, such as u-net.
[The Paper](docs/1411.4038.pdf)



# CUDA出错的验证

[Windows下PyTorch(LibTorch)配置cuda加速](https://blog.csdn.net/my544903633/article/details/107111861/)中提到在libtorch 1.9以上版本设置是`-INCLUDE:?warp_size@cuda@at@@YAHXZ `，但在`1.10`和`1.10.1`版本中还是要设置为`/INCLUDE:?searchsorted_cuda@native@at@@YA?AVTensor@2@AEBV32@0_N1@Z`才不会出错。

# 程序说明
因为只有一张980ti的老卡，所以没有完全跑完训练和测试。有关验证的代码今后有机会再添加与调试，包括对应Python代码中的`myloss.CrossEntropyLoss2d`的`C++`代码也未完成 

将网上下载的VOC数据包直接解压到`d:\data`目录下，完整为`d:\data\VOCdevtik`,程序默认为`CPU`运行，可以代参数运行
```
libtorch_fcn.exe --device=cuda
```
## 程序运行输出
```
You settings:
   device: cuda
   Dataset root: d:\data\VOCdevkit\VOC2012
   Output root: d:\data
   Batch size: 5 total epochs: 50
   learn rate init: 0.01 learn rate adjust value: 0.5
I1228 22:25:19.502473 18744 fcntrain.cpp:81] epoch:0 learn times:5 avg loss:2.76732 lr:0.01 cost:950
I1228 22:25:21.924997 18744 fcntrain.cpp:81] epoch:0 learn times:10 avg loss:3.53316 lr:0.01 cost:16
I1228 22:25:24.320307 18744 fcntrain.cpp:81] epoch:0 learn times:15 avg loss:2.97064 lr:0.01 cost:19
I1228 22:25:26.750819 18744 fcntrain.cpp:81] epoch:0 learn times:20 avg loss:2.97263 lr:0.01 cost:17
I1228 22:25:29.135890 18744 fcntrain.cpp:81] epoch:0 learn times:25 avg loss:2.75852 lr:0.01 cost:18
I1228 22:25:31.535954 18744 fcntrain.cpp:81] epoch:0 learn times:30 avg loss:2.74817 lr:0.01 cost:16
I1228 22:25:33.932533 18744 fcntrain.cpp:81] epoch:0 learn times:35 avg loss:2.55081 lr:0.01 cost:17
I1228 22:25:36.311609 18744 fcntrain.cpp:81] epoch:0 learn times:40 avg loss:2.40762 lr:0.01 cost:19
I1228 22:25:38.708510 18744 fcntrain.cpp:81] epoch:0 learn times:45 avg loss:2.71759 lr:0.01 cost:16
I1228 22:25:41.106178 18744 fcntrain.cpp:81] epoch:0 learn times:50 avg loss:2.55648 lr:0.01 cost:16
I1228 22:25:43.504539 18744 fcntrain.cpp:81] epoch:0 learn times:55 avg loss:2.53127 lr:0.01 cost:18
I1228 22:25:45.878552 18744 fcntrain.cpp:81] epoch:0 learn times:60 avg loss:2.50167 lr:0.01 cost:18
I1228 22:25:48.281028 18744 fcntrain.cpp:81] epoch:0 learn times:65 avg loss:2.40137 lr:0.01 cost:15
I1228 22:25:50.681831 18744 fcntrain.cpp:81] epoch:0 learn times:70 avg loss:2.39125 lr:0.01 cost:19
I1228 22:25:53.079835 18744 fcntrain.cpp:81] epoch:0 learn times:75 avg loss:2.31588 lr:0.01 cost:18
......
```