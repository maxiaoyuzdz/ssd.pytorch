import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VGGDetection, detection_collate, VOCroot, VOC_CLASSES, vgg_detection_collate
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
import cnn
from averagemeter import AverageMeter

import torchvision.transforms as transforms

def getTransformsForVGGTraining():
    return transforms.Compose([
        #transforms.Lambda(lambda x: hsvTransform(x)),
        #transforms.Lambda(lambda x: verticalFlipTransform(x)),
        #transforms.Lambda(lambda x: rotateTransform(x)),
        #transforms.Lambda(lambda x: noiseTransform(x)),
        transforms.Scale(224),
        transforms.RandomCrop(224, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v1', help='still be in edit')
parser.add_argument('--basenet', default='/media/maxiaoyu/data/pretrainedmodel/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='/media/maxiaoyu/data/checkpoint/vgg/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]

ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = len(VOC_CLASSES) + 1
batch_size = args.batch_size

max_iter = 80
weight_decay = 0.0005
stepvalues = (30, 50)
gamma = 0.1
momentum = 0.9




"""
if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
"""
    #cudnn.benchmark = True






def runTraining():
    vgg_net = cnn.__dict__['vgg16_bn'](20)
    print('====== net =====')
    print(vgg_net)
    print('================')
    """
    net = vgg_net
    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()


    net.train()

    epoch = 0
    print('Loading Dataset...')
    """



    dataset = VGGDetection(args.voc_root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())

    epoch_size = len(dataset) // args.batch_size
    print('Training vgg on', dataset.name)
    step_index = 0


    #batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True)
    batch_count = 0
    # epoch iteration
    for iteration in range(args.start_iter, max_iter):
        #if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            #batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            #adjust_learning_rate(optimizer, args.gamma, step_index)
            #epoch += 1

        # load train data
        #images, targets = next(batch_iterator)

        for i, (images, targets) in enumerate(data_loader):
            batch_count += 1

            print(images.size())
            print(len(targets))
            # process image data


            """
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            
            t0 = time.time()
            out = net(images)
            
            optimizer.zero_grad()

            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            t1 = time.time()

            if batch_count % 10 == 0:
                print('Timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(batch_count) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            """



        """
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(vgg_net.state_dict(), '/media/maxiaoyu/data/checkpoint/vgg/vgg_' + repr(iteration) + '.pth')
        """
    # used for eval
    #torch.save(vgg_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    runTraining()
