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
from utils.augmentations import SSDAugmentation, VGGAugmentation, VGGValAugmentation
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
parser.add_argument('--version', default='v2', help='still be in edit')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
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
val_sets = [('2007', 'test'), ('2012', 'val')]

ssd_dim = 224  # only support 300 now
means = (104, 117, 123)  # only support voc now
batch_size = args.batch_size

max_iter = 320
weight_decay = 0.0005
stepvalues = (120, 220)
gamma = 0.1
momentum = 0.9




"""
if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
"""
    #cudnn.benchmark = True



def accuracy(output, target, topk=(1,)):
    """use cuda tensor, parameters output and target are cuda tensor"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred).long())

    res = []
    correct_k = correct.sum()
    accuracy_value = correct_k * 100.0 / batch_size
    #res.append(correct_k * 100.0 / batch_size)

    return accuracy_value


def train(training_set_loader, model, criterion, optimizer, epoch):
    """train a epoch, return average loss and accuracy"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.train()

    for i, (images, targets) in enumerate(training_set_loader):

        # process image data
        images = Variable(images.cuda())
        targets = Variable(targets.cuda())

        t0 = time.time()

        out = model(images)
        optimizer.zero_grad()
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()


        t_accuracy = accuracy(out.float().data, targets.float().data)
        losses.update(loss.data[0], images.size(0))
        accuracies.update(t_accuracy, images.size(0))

        t1 = time.time()

        if i % 16 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(i) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

    return losses.avg, accuracies.avg




def validate(val_set_loader, model, criterion):
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()
    for i, (images, targets) in enumerate(val_set_loader):
        # process image data
        images = Variable(images.cuda())
        targets = Variable(targets.cuda())

        t0 = time.time()
        out = model(images)
        loss = criterion(out, targets)
        t_accuracy = accuracy(out.float().data, targets.float().data)
        losses.update(loss.data[0], images.size(0))
        accuracies.update(t_accuracy, images.size(0))

        t1 = time.time()

        if i % 16 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('EVAL iter ' + repr(i) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
    # return average loss and accuracy
    return losses.avg, accuracies.avg


def runTraining():
    vgg_net = cnn.__dict__['vgg16_bn'](20)
    print('====== net =====')
    print(vgg_net)
    print('================')

    # net = torch.nn.DataParallel(vgg_net)
    net = vgg_net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()


    print('Loading Dataset...')

    train_dataset = VGGDetection(args.voc_root, train_sets, VGGAugmentation(
        ssd_dim, means), AnnotationTransform())
    train_data_loader = data.DataLoader(train_dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=vgg_detection_collate, pin_memory=True)

    val_dataset = VGGDetection(args.voc_root, val_sets, VGGValAugmentation(
        ssd_dim, means), AnnotationTransform())
    val_data_loader = data.DataLoader(val_dataset, batch_size, num_workers=args.num_workers,
                                        shuffle=True, collate_fn=vgg_detection_collate, pin_memory=True)

    best_accuracy = 0
    is_best = False
    # epoch iteration
    step_index = 0
    for iteration in range(args.start_iter, max_iter):
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)


        # load train data
        #images, targets = next(batch_iterator)
        # train
        t0 = time.time()
        t_loss_avg, t_accuracy_avg = train(train_data_loader, net, criterion, optimizer, iteration)
        t1 = time.time()
        print('Epoch Train Timer: %.4f sec.' % (t1 - t0))
        print(print('Training iter ' + repr(iteration) + ' || Loss: %.4f ||' % (t_loss_avg) + ' || ac: %.4f ||' % (t_accuracy_avg), end=' '))

        # validation

        t2 = time.time()
        v_loss_avg, v_accuracy_avg = validate(val_data_loader, net, criterion)
        t3 = time.time()
        print('Epoch Val Timer: %.4f sec.' % (t3 - t2))
        print(print('Val iter ' + repr(iteration) + ' || Loss: %.4f ||' % (v_loss_avg) + ' || ac: %.4f ||' % (
        v_accuracy_avg), end=' '))

        # check best
        is_best = v_accuracy_avg > best_accuracy
        best_accuracy = max(v_accuracy_avg, best_accuracy)

        if iteration % 50 == 0:
            print('Saving features, iter:', iteration)
            torch.save(net.features.state_dict(),
                       '/media/maxiaoyu/data/checkpoint/vgg/vgg16_voc0712_features' + repr(iteration) + '.pth')
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(),
                       '/media/maxiaoyu/data/checkpoint/vgg/vgg16_voc0712_' + repr(iteration) + '.pth')

    # used for eval
    torch.save(net.state_dict(), args.save_folder + '' + args.version + '.pth')


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
