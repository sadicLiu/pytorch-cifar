'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim

from dataloader import get_data
from models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--num_epoch', default=150, type=int, help='Number of training epochs')
parser.add_argument('--save_flag', required=True, type=str, help='The flag appended to ckpt name')

args = parser.parse_args()

project_dir = os.path.dirname(os.path.abspath(__file__))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.empty_cache()
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
train_losses, test_losses, train_acc, test_acc = [], [], [], []  # save metric of each epoch

# Data
trainloader, testloader, classes = get_data()

# Model
print('==> Building model..')
# net = VGG('VGG16')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = DenseNet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)  # , device_ids=[0, 1])  # 多GPU计算
    cudnn.benchmark = True

optim_flag = 'sgd'
ckpt_name = project_dir + '/checkpoint/ckpt_' + args.save_flag + '_' + optim_flag + '.pth'
metric_name = project_dir + '/metric/metric_' + args.save_flag + '_' + optim_flag + '.json'
# ckpt_name = project_dir + '/checkpoint/ckpt_' + args.save_flag + '.pth'
# metric_name = project_dir + '/metric/metric_' + args.save_flag + '.json'
if args.resume:
    # Load checkpoint.
    net, best_acc, start_epoch = resume_ckpt(net, ckpt_name)
    train_losses, test_losses, train_acc, test_acc = load_metric(metric_name)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    global train_losses, train_acc

    # 累加每个batch的loss, 整个train()函数调用结束后, train_loss才代表当前epoch的loss
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每个batch跑完更新一次 progress bar
        progress_bar(batch_idx, len(trainloader), msg='Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # add metrics
    train_losses.append(train_loss)
    train_acc.append(100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()

    global test_losses, test_acc

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt_name)
        best_acc = acc

    # add metrics
    test_losses.append(test_loss)
    test_acc.append(acc)


for epoch in range(start_epoch, start_epoch + args.num_epoch):
    train(epoch)
    test(epoch)
    if not os.path.isdir(os.path.join(project_dir, 'metric')):
        os.mkdir(os.path.join(project_dir, 'metric'))
    save_metric(metric_name, train_losses, test_losses, train_acc, test_acc)
