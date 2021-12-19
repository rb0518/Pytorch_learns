import os
import xml.etree.ElementTree as ET

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    #print("file name:{}".format(filename))
    tree = ET.parse(filename)
    objects = []
    allobjs = tree.findall('object')
    for obj in allobjs:
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
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



class ExtractVOCLabels():
    """
        data_root: 指向数据集目录路径，在Windows下如D:/VOCdetkit/VOC2007，Linux下如/home/VOCdetkit/VOC2012
        out_root: 存放提取的目标标注信息与文件名一块存放到train.tx 和 val.txt中
    """
    def __init__(self, data_root, out_root):
        print(data_root)
        self.data_root = data_root
        self.out_root = out_root

        """
            为了保持原始数据目录今后使用，检测是否输入与输出目录一致
        """
        if data_root == out_root:
            print('output txt file must not in same directory')
            exit()

        self.annotations_root = os.path.join(data_root, 'Annotations')
        self.trainlist_filename = os.path.join(data_root, "ImageSets/Main/train.txt")
        self.vallist_filename = os.path.join(data_root, "ImageSets/Main/val.txt")
        self.images_root = os.path.join(data_root, 'JPEGImages')

        print('train list file name: {}'.format(self.trainlist_filename))
        print('val list file name: {}'.format(self.vallist_filename))

        print(out_root)
        os.makedirs(out_root, exist_ok=True)   # exist_ok = False是为了保证第二次运行时不会报目录已经存在的错误 
        out_train_txt_file = open(os.path.join(out_root,'train.txt'), 'w')
        out_val_txt_file = open(os.path.join(out_root, 'val.txt'), 'w')

        print("out file name:{}, {}".format(os.path.join(out_root,'train.txt'), os.path.join(out_root, 'val.txt')))

        train_txt_file = open(self.trainlist_filename, 'rt')
        trainlist = train_txt_file.readlines()
        train_txt_file.close()

        val_txt_file = open(self.vallist_filename, 'rt')
        vallist = val_txt_file.readlines()
        val_txt_file.close()

        print("train count: {}, val count: {}".format(len(trainlist), len(vallist)))
        trainlist = [x[:-1] for x in trainlist]
        vallist = [x[:-1] for x in vallist]

        count = 0
        for fname in trainlist:
            count += 1
            
            image_path = fname.strip() + '.jpg'
            results = parse_rec(os.path.join(self.annotations_root,fname.strip()+'.xml'))
            if len(results) == 0:
                print('{} read labels is 0'.format(os.path.join(self.annotations_root,fname)))
            else:
                out_train_txt_file.write(image_path)
                for result in  results:
                    class_name = result['name']
                    bbox = result['bbox']
                    class_idx = VOC_CLASSES.index(class_name)
                    """
                        out line like: jpgfile obj1_bbox_minx obj1_bbox_miny obj1_bbox_maxx obj1_bbox1_maxy obj1_classidx obj2_bbox_minx ... obj2_classidx
                    """
                    out_train_txt_file.write(' ' + str(bbox[0]) + \
                                    ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + \
                                    ' ' + str(bbox[3]) + ' ' + str(class_idx))
                out_train_txt_file.write('\n')
        print('total {} line had write into train.txt'.format(count))

        count = 0
        for fname in vallist:
            count += 1
            
            image_path = fname.strip() + '.jpg'
            results = parse_rec(os.path.join(self.annotations_root,fname.strip()+'.xml'))
            if len(results) == 0:
                print('{} read labels is 0'.format(os.path.join(self.annotations_root,fname)))
            else:
                out_val_txt_file.write(image_path)
                for result in  results:
                    class_name = result['name']
                    bbox = result['bbox']
                    class_idx = VOC_CLASSES.index(class_name)
                    """
                        out line like: jpgfile obj1_bbox_minx obj1_bbox_miny obj1_bbox_maxx obj1_bbox1_maxy obj1_classidx obj2_bbox_minx ... obj2_classidx
                    """
                    out_val_txt_file.write(' ' + str(bbox[0]) + \
                                    ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + \
                                    ' ' + str(bbox[3]) + ' ' + str(class_idx))
                out_val_txt_file.write('\n')
        print('total {} line had write into val.txt'.format(count))

        out_train_txt_file.close()
        out_val_txt_file.close()
        print('extract labels info from xml files over.')



        