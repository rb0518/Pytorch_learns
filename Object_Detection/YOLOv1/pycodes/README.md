# Pytorch YOLOv1代码精读

在利用libtorch实现YOLOv1之前，我们需要通过学习YOLOv1的相关知识，同时网上Python实现代码较多，C++的例子主要用来测试，参考意义不大。所以先需要根据Python的代码精读，理解整个模型架构和训练数据准备与过程。

主要参考网络上文章[zhangshen12356：pytorch简单实现yolo v1](https://blog.csdn.net/weixin_41009689/article/details/106036687)，以及他文章中参考的两个代码[DuanYiqun/pytorch_implementation_of_Yolov1](https://github.com/DuanYiqun/pytorch_implementation_of_Yolov1)和[abeardear/pytorch-YOLO-v1](https://github.com/abeardear/pytorch-YOLO-v1)。主要以读`DuanYiqun`的代码为主

## 1. 数据集相关操作

Pascal voc 2012数据集可以用于分类、检测和分割。原始的数据定义和下载可到[官网 -The PASCAL Visual Object Classes Homepage](http://host.robots.ox.ac.uk/pascal/VOC/index.html)，主要有20个类别。以`PASCAL VOC 2012`文件目录
```
+ VOCdevkit
  + VOC2012
    + Annotations
    + ImageSets
    + JPEGImages
    + SegmentationClass
    + SegmentationObject
```
### 1.1 文件目录说明：
- Annotations: 这个文件夹内主要存放了数据的标签，里面包含了每张图片的bounding box信息，主要用于目标检测，格式为`XML`。
- ImageSets: ImageSets中的四个子目录下存放了对应数据集应用的定义与索引文件，我们可利用`train`, `val`, `trainval`这几个`txt`文件
- JPEGImages: 这里存放的就是JPG格式的原图，包含17125张彩色图片。
SegmentationClass: 语义分割任务中用到的label图片，PNG格式，共2913张，与原图的每一张图片相对应。
- SegmentationObject: 实例分割任务用到的label图片。

其中的`train.txt`中的文件作为训练集索引，`val.txt`中文件名作为验证集索引

### 1.2 提取标注信息
从xml文件中提取信息如下
```python
#file: xml2txt.py
import xml.etree.ElementTree as ET

def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text) #抛弃标注为困难的目标，所以后面要判定返回对象是否为空
        if difficult == 1:
            continue

        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
                            int(float(bbox.find('xmin').text)),
                            int(float(bbox.find('ymin').text)),
                            int(float(bbox.find('xmax').text)),
                            int(float(bbox.find('ymax').text))
                            ]
        objects.append(obj_struct)
    return objects
```
根据索引文件中文件名逐一匹配，提取对应xml中的目标标注信息，并写入到txt文件中待用。其实这里根本不需要存到文件中，因为整体数据集的信息
很有限，完全可以一次存放在内存中，只是因为训练集可能文件很多，这样每次都要一个个从xml文件中操作也是很繁琐的。
 ```python
# file: xml2txt.py
VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


    xml_files = os.listdir(self.annotations_root)
    #print("found xml files:{}".format(len(xml_files)))

    for xml_file in xml_files:
        image_path = xml_file.split('.')[0] + '.jpg'
        if(xml_file.split('.')[0] in trainlist):
            results = parse_rec(self.annotations_root+xml_file)
            if len(results) == 0:
                print('{} read labels is 0'.format(xml_file))
            else:
                for result in  results:
                    class_name = result['name']
                    bbox = result['bbox']
                    class_idx = VOC_CLASSES.index(class_name)
                    out_train_txt_file.write(image_path + ' ' + str(bbox[0]) + \
                                    ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + \
                                    ' ' + str(bbox[3]) + ' ' + str(class_idx))
```
输出文件中每一行都记录了文件名和对应每个Object的标注信息: `文件名 目标1 目标2 ... 目标n`，如下所示
```
2008_000008.jpg 53 87 471 420 12 158 44 289 167 14
```
可以看到这里用到的是XML文件中的原始坐标信息，即每个标注对像为实际图中的坐标信息[`xmin ymin xmax ymax classid`]，有些例子则是用了中心坐标加宽高，并用了图像宽高来规一化坐标代码可以如下所示
```python
    for result in  results:
        class_name = result['name']
        bbox = result['bbox']
        class_idx = VOC_CLASSES.index(class_name)
        #image_width image_height可以在parse_rec函数中通过查找width和height节点获得
        center_x = ((bbox[0] + bbox[2]) / 2) / image_width  
        center_y = ((bbox[1] + bbox[3]) / 2) / image_height
        width = (bbox[2] - bbox[0]) / image_width
        height = (bbox[3] - bbox[1]) / image_height

        out_train_txt_file.write(image_path + ' ' + str(center_x) + \
                        ' ' + str(center_y) + ' ' + str(width) + \
                        ' ' + str(height) + ' ' + str(class_idx))
```

## 2. 数据集相关操作
继承`torch.utils.data.Dataset`构建自己的`YOLODataset`类。只是参考文中提到在进行数据增强时，有一些比如旋转、随即裁剪等会改变图片中物体的bbox的坐标，因此不能直接应用torchvision里面的transform包来进行数据增强，通过`opencv`自己实现，其中有几个函数需要同时修改`boxes`，代码见`dataset.py`

## 3. 损失函数
以下几点对理解`loss.py`代码比较有用
- 更重视8维的坐标预测，给这些损失前面赋予更大的Loss weight，记为l_coord，在pascaal VOC训练中取5
- 对于没有物体的box的confidense loss,赋予小的loss weight,记为l_noobj，在pascaal VOC训练中取0.5
- 有object的box的confidence loss和类别的loss的loss weight正常取1。
- 运行的时候会有警告：
  

  ```
  Warning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead
  ```
  `yololoss.py`中搜索`torch.ByteTenser`，用`torch.BoolTensor`替换，网上有人提到下面的代码，经测试没有效果。
    ```
    import warnings
    warnings.filterwarnings('ignore')  
    ```

## 运行
pycodes同级目录下建data目录，选运行python main.py --run_mode=other, 生成train.txt和val.txt，然后运行运行python main.py，默认参数可不修改 VOC数据集解压到D:\data\VOCdevtik

```
Epoch: 0
batch 0 of total batch 572 Loss: 271.576 
batch 1 of total batch 572 Loss: 228.606 
batch 2 of total batch 572 Loss: 193.480 
batch 3 of total batch 572 Loss: 161.341 
batch 4 of total batch 572 Loss: 141.175 
batch 5 of total batch 572 Loss: 128.351 
batch 6 of total batch 572 Loss: 116.464 
batch 7 of total batch 572 Loss: 108.521 
batch 8 of total batch 572 Loss: 100.967 
batch 9 of total batch 572 Loss: 94.359 
batch 10 of total batch 572 Loss: 88.772 
batch 11 of total batch 572 Loss: 84.791 
batch 12 of total batch 572 Loss: 81.212 
batch 13 of total batch 572 Loss: 78.943 

```