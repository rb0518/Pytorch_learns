# PyTorch Learns
These are Deep Learning sample program of PyTorch written in Python and C++.

## Description
PyTorch is famous as a kind of Deep Learning Frameworks.<br>
Because on the web, we can easily found Python source codes. But I usually write code in C++. I will collected c++ codes or converted Python codes to C++ in this repository. I hope this repository will help many programmer who written in C++.<br> 

选择PyTorch作为深度学习的主力，这里用来记录自己的深度学习过程，主要是收集整理网络上有关深度学习的Python和C++代码。Python代码主要为学习，实现尽可能用C++来实现。因为工作环境为Windows，所以全部的代码将以Windows11下Visual Studio 2019下实现，有可能部分网上代码测试会用到WSL2。



## Requirement

### 1. OS
Many codes is run in linux, but I'm working on windows every day. So this repository demo is run in windows. The IDE is Visual Studio Code and Visual Studio 2019.
All code run in Windows 10 and Windows 11, WSL2.

### 2. Python

**Python 3.9**

You can download from [Python official](https://www.python.org/), and add a file which name `pip.ini`  under the `c:\user\current user\.pip\` folder.
~~~
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
~~~

**OpenCV**

I am git clone the OpenCV latest version codes from [Github](https://github/opencv/opencv.git) and build with CMake and Visual Studio 2019. You can install OpenCV use the command.
~~~
pip install opencv-python
~~~

**PyTorch**
  
Please select the environment to use as follows on [PyTorch official](https://pytorch.org/), I am install this version `Stable (1.10.0)`

```
PS C:\Users\Michael> python
Python 3.9.9 (tags/v3.9.9:ccb0e6a, Nov 15 2021, 18:08:50) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'1.10.0+cu113'
>>>
```

### 3. C++

**libtorch**

Please select the environment to use as follows on PyTorch official. <br>
PyTorch official : https://pytorch.org/ <br>
***
PyTorch Build : Stable (1.10.0) + CUDA11.3 <br>
CUDA : 11.5 <br>

**Other depended library**

I like use this libraries, store in one folder like `D:\thirdparty`, because when  use caffe, I learned from web.
- boost 1.7.0 
- glog 
- gflags
- protobuf 3.8
  
## First project

First step, use PyTorch tutorials demo [TRAINING A CLASSIFIER](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). I will start learning from Python codes. Next step will directly implement the same training and testing with C + + in Visual Studio 2019 environment.The working folder name is `cifar10_tutorial`.

因为大多教程都有现成的例子，所以第一个程序也将以PyTorch官方新手教程中的例子入手，首先是Python代码阅读，然后会用libtorch库，在Visual Studio 2019中实现Python代码同样的训练与测试。