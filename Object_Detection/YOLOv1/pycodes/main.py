import os
import argparse

import torch
from torch.nn import parameter
#from dataset import VOCDataset
#from YOLOv1 import resnet50
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from xml2txt import ExtractVOCLabels
import train

parser = argparse.ArgumentParser(description='Pytorch YOLOv1')

parser.add_argument('--gpu_or_cpu', default='cuda', type=str, help='select run on GPU or CPU, default is run on GPU')
parser.add_argument('--run_mode', default='train', type=str, help='program run mode: train, detect, test')
parser.add_argument('--data_root',default='../data', type=str, help='store for data, log, weghts')
parser.add_argument('--train_file', default='../data/train.txt', type=str, help='train list file')
parser.add_argument('--test_file', default='../data/val.txt', type=str, help='test list file')
parser.add_argument('--model_name',default='train50', type=str, help='experimentname')

"""parameters for model"""
parser.add_argument('--learn_rate', default=0.01, type=float, help='learning rate')     
parser.add_argument('--num_grid', default=7, type=int, help='grid numbers, 7*7 is default set')
parser.add_argument('--num_bbox', default=2, type=int, help='bounding box number, default is 2')
parser.add_argument('--num_class', default=20, type=int, help='class number, On VOC dataset, it is 20')
parser.add_argument('--num_epochs', default=30, type=int, help='training length')
parser.add_argument('--batch_size', default=10, type=int, help='batch size for dataload')

"""VOC dataset"""
parser.add_argument('--dataset_root', default='d:/data/VOCdevkit/', type=str, help='root for VOC dataset')
parser.add_argument('--dataset_year', default='2012', type=str, help='select the dataset year: 2007 or 2012')
parser.add_argument('--xml_to_txt', default=True, help='extract labels from xml files and save to data_root')

def main():
    args = parser.parse_args()
    show_settings(args)
    env = envsettings(args)

    if args.run_mode == 'train':
        run_mode_train(args, env)
    elif args.run_mode == 'test':
        print('start test...')
    elif args.run_mode == 'detect':
        print('start detect')
    else:
        print('start extract labels from xml files')
        xml2txt = ExtractVOCLabels(env.dataset_root, args.data_root)

class envsettings():
    def __init__(self, parser):
        self.parse_args = parser
        self.device = 'cuda' if torch.cuda.is_available() and parser.gpu_or_cpu == 'cuda' else 'cpu'
#        print('test out:'+self.device)
        self.dataset_root = os.path.join(parser.dataset_root, "VOC"+parser.dataset_year)
        if not os.path.exists(self.dataset_root):
            print("***Error {} not exists.".format(self.dataset_root))
            exit()


def show_settings(parser):
    print('run mode:{}'.format(parser.run_mode))
    print('device: {}'.format(parser.gpu_or_cpu))

def run_mode_train(parser, env):
    imagefiles_root = env.dataset_root + '/JPEGImages/'
    train.run_train(device = env.device,image_root = imagefiles_root, trainlist_file = parser.train_file,
        testlist_file = parser.test_file,num_epochs = parser.num_epochs,
        batch_size = parser.batch_size,num_grid = parser.num_grid,
        num_bbox = parser.num_bbox,
        num_class = parser.num_class,
        learn_rate = parser.learn_rate,
        data_root = parser.data_root, model_name = parser.model_name)
    print('train over...')

if __name__ == '__main__':
    main()