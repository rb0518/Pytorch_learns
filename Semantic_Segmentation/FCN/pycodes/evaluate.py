import os
import numpy as np

import torch
from torch.autograd import Variable

import models
import voc_loader
import tools

def evaluate(args):
    batch_size = args.batch_size            
    n_class = args.n_class
    use_cuda = True if args.device == 'cuda' and torch.cuda.is_available() else False

    data_path = os.path.join(args.dataset_root, "VOC"+args.dataset_year)

    val_data = voc_loader.VOC2012ClassSeg(root = data_path, split='val', transform=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers = 5)

    print('load model....')

    vgg_model = models.VGGNet(requires_grad=True, dataroot=args.data_root)
    fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

    if use_cuda:
        fcn_model.cuda()
    
    fcn_model.eval()

    label_trues, label_preds = [], []

    for idx in range(len(val_data)):
        img, label = val_data[idx]
        img = img.unsqueeze(0)
        if use_cuda:
            img = img.cuda()
        
        img = Variable(img)
        out = fcn_model(img)

        pred = out.data.max(1)[1].squeeze_(1).squeeze_(0)   # 320, 320

        if use_cuda:
            pred = pred.cpu()
        label_trues.append(label.numpy())
        label_preds.append(pred.numpy())

        if idx % 30 == 0:
            print('evaluate [%d/%d]' % (idx, len(val_loader)))

    metrics = tools.accuracy_score(label_trues, label_preds)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
            Accuracy: {0}
            Accuracy Class: {1}
            Mean IU: {2}
            FWAV Accuracy: {3}'''.format(*metrics))
