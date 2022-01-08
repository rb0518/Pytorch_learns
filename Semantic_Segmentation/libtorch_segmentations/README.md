# **语义分割（Semantic Segmentation)**

语义分割(Semantic Segmentation)主要是用来对图像上的每个像素都要进行分类（返回每个像素是那个类的概率），主要的算法有FCN，SegNet，Dilated Convolutions, DeepLabs(V1，V2 V3), PSPNet, Larg Kernel Matters...

## 2014 **FCN ：Fully Convolutional Networks for Semantic Segmentation**
引入全卷积网络，可以接受任意大小的输入图像。但结果细节不敏感，训练耗时，没有充分考虑像素间的关系

## 2015 **U-Net: Convolutional Networks for Biomedical Image Segmentation**
可以对任意形状大小的图像进行卷积操作，将Encoder的每层卷积结果拼接到Decoder中，得到更好的结果，训练比FCN效率高

## 2015 **SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**
通过译码过程中引入池化层，提升分割分辨率。

## 2015 **DeconvNet: Learning Deconvolution Network for Semantic Segmentation**
比SegNet增加了两个全连接层。

## **Dilated Convolutions: Multi-Scale Context Aggregation by Dilated Convolutions**
通过Dilate扩大卷积观察视野。如`3*3`的卷积核，当`Dilate=2`时，观察视野变为了`7*7`

# 代码

参考和引用了***Github***上[LibtorchSegmentation](https://github.com/AllentDan/LibtorchSegmentation)的代码
## 编译环境
- Visual Studio 2019
- opencv 4.5.1
- boot 1.7.0
- libtorch 1.10.1_cu11.3
  