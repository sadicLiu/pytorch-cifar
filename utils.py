'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import json

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms


def load_metric(metric_name):
    assert os.path.isfile(metric_name), 'Error: no metric file found!'
    with open(metric_name) as f:
        metric = json.load(f)

    train_losses = metric['train_losses']
    test_losses = metric['test_losses']
    train_acc = metric['train_acc']
    test_acc = metric['test_acc']

    return train_losses, test_losses, train_acc, test_acc


def save_metric(metric_name, train_losses, test_losses, train_acc, test_acc):

    metric = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    print('Saving metrics...')
    with open(metric_name, 'w') as f:
        json.dump(metric, f)


def resume_ckpt(net, ckpt_name):
    '''Load saved info from ckpt file'''
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(ckpt_name), 'Error: no checkpoint file found!'
    checkpoint = torch.load(ckpt_name)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net, best_acc, start_epoch


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# 获取命令行的宽高, 必须在命令行里运行才有效
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


# Epoch: 1
# ^Z===>..... 23/391 .....]  Step: 109ms | Tot: 2s457ms | Loss: 2.123 | Acc: 21.501% (633/2944)
# 每个batch调用一次
def progress_bar(current_batch, total_batch, msg=None):
    global last_time, begin_time
    if current_batch == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current_batch / total_batch)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()  # 当前batch结束的时间
    step_time = cur_time - last_time  # 两个batch之间间隔的时间
    last_time = cur_time
    total_time = cur_time - begin_time  # 从第1个batch到当前batch所用的时间

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Total: %s' % format_time(total_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 6):
        sys.stdout.write('\b')
    sys.stdout.write(' batch:%d/%d ' % (current_batch + 1, total_batch))

    if current_batch < total_batch - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


if __name__ == '__main__':
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/home/liuhy/res/deep-learning/数据集/cifar',
                                            train=True, download=False, transform=transform_train)
    mean, std = get_mean_and_std(trainset)
    print(mean)
    print(std)
