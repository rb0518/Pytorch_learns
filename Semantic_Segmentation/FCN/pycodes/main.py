import os
import argparse
import torch
import train
import evaluate

parser = argparse.ArgumentParser(description='Python FCN')

parser.add_argument('--run_mode', type = str, choices=('train', 'evaluate'), default='train', help='set program run mode')
parser.add_argument('--device', type = str, choices=('cuda', 'cpu'), default='cuda', help='set gpu or cpu')

"""VOC dataset"""
parser.add_argument('--dataset_root', default='d:/data/VOCdevkit/', type=str, help='root for VOC dataset')
parser.add_argument('--dataset_year', default='2012', type=str, help='select the dataset year: 2007 or 2012')
parser.add_argument('--data_root',default='../data', type=str, help='store for data, log, weghts')

"""train parameters"""
parser.add_argument('--batch_size', type=int, default=5, help="batch size of the data")
parser.add_argument('--epochs', type=int, default=30, help='epoch of the train')
parser.add_argument('--n_class', type=int, default=21, help='the classes of the dataset')   # add a class: background
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')


def main():
    args = parser.parse_args()

    dataset_root = os.path.join(args.dataset_root, "VOC"+args.dataset_year)
    device = 'cpu' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print("Python FCN program select device: {}".format(device))
    print("Dataset root:{}".format(dataset_root))

    if args.run_mode == 'train':
        train.train(args)
    elif args.run_mode =='evaluate':
        evaluate(args)

    os.makedirs(args.data_root, exist_ok=True)

if __name__ == '__main__':
    main()