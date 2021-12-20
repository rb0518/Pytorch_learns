import os
import torch
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import torchvision
from torch.optim import Adam, SGD, optimizer

import models
import voc_loader
import myloss

def train(args):
    batch_size = args.batch_size            
    learning_rate = args.learning_rate
    epoch_num = args.epochs
    n_class = args.n_class

    best_test_loss = np.inf
    pretrained = 'reload'       #resume learn

    use_cuda = True if args.device == 'cuda' and torch.cuda.is_available() else False

    data_path = os.path.join(args.dataset_root, "VOC"+args.dataset_year)
    print('load data....')

    # create the dataset and dataloader
    train_data = voc_loader.VOC2012ClassSeg(root = data_path, split='train', transform=True)
    train_loader = torch.utils.data.dataloader.DataLoader(train_data,
                                            batch_size = batch_size,
                                            shuffle=True,
                                            num_workers = 5)
    
    val_data = voc_loader.VOC2012ClassSeg(root = data_path, split='val', transform=True)
    val_loader = torch.utils.data.dataloader.DataLoader(val_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=5)

    vgg_model = models.VGGNet(requires_grad=True, dataroot=args.data_root)
    fcn_model = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

    if use_cuda:
        fcn_model.cuda()
    
    criterion = myloss.CrossEntropyLoss2d()           
    optimizer = Adam(fcn_model.parameters())

    for epoch in range(epoch_num):
        # tran mode
        total_loss = 0.
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            N = imgs.size(0)
            if use_cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            imgs_tensor = Variable(imgs)       # torch.Size([2, 3, 320, 320])
            labels_tensor = Variable(labels)   # torch.Size([2, 320, 320])
            out = fcn_model(imgs_tensor)       # torch.Size([2, 21, 320, 320])

            # with open('./result.txt', 'r+') as f:
            #     f.write(str(out.detach().numpy()))
            #     f.write("\n")

            loss = criterion(out, labels_tensor)
            loss /= N
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            # update all arguments
            total_loss += loss.item()  # return float

            # if batch_idx == 2:
            #     break

            if (batch_idx) % 20 == 0:
                print('train epoch [%d/%d], iter[%d/%d], lr %.7f, aver_loss %.5f' % (epoch,
                                                                                    epoch_num, batch_idx,
                                                                                    len(train_loader), learning_rate,
                                                                                    total_loss / (batch_idx + 1)))

            # # visiualize scalar
            # if epoch % 10 == 0:
            #     label_img = tools.labelToimg(labels[0])
            #     net_out = out[0].data.max(1)[1].squeeze_(0)
            #     out_img = tools.labelToimg(net_out)
            #     writer.add_scalar("loss", loss, epoch)
            #     writer.add_scalar("total_loss", total_loss, epoch)
            #     writer.add_scalars('loss/scalar_group', {"loss": epoch * loss,
            #                                              "total_loss": epoch * total_loss})
            #     writer.add_image('Image', imgs[0], epoch)
            #     writer.add_image('label', label_img, epoch)
            #     writer.add_image("out", out_img, epoch)

            assert total_loss is not np.nan
            assert total_loss is not np.inf

        # model save
        if (epoch) % 5 == 0:
            pretrained_path = os.path.join(args.data_root, 'pretrained_models')
            os.makedirs(pretrained_path, exist_ok=True)
            tempfilename = 'model'+str(epoch)+'.pth'
            torch.save(fcn_model.state_dict(), os.path.join(pretrained_path, tempfilename))  # save for 5 epochs
        total_loss /= len(train_loader)
        print('train epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))

        if epoch % 5 == 0:         # 每5次调整一次learning_rate
            learning_rate *= 0.01
            optimizer.param_groups[0]['lr'] = learning_rate
    

 
