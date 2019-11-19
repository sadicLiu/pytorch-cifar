"""Visualize metrics during model training and testing"""
from utils import load_metric
import matplotlib.pyplot as plt


def plot_loss(metric_name):
    save_name = metric_name.split('.json')[0] + '_loss'
    train_losses, test_losses, _, _ = load_metric(metric_name)
    assert len(train_losses) == len(test_losses), \
        "Number of train loss and test loss must be same"

    epoch = len(train_losses)
    x = range(1, epoch + 1)
    plt.clf()
    plt.plot(x, train_losses, 'r--', label='train')
    plt.plot(x, test_losses, 'b--', label='test')
    plt.plot(x, train_losses, 'ro-', x, test_losses, 'bo-')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_name)


def plot_acc(metric_name):
    save_name = metric_name.split('.json')[0] + '_acc'
    _, _, train_acc, test_acc = load_metric(metric_name)
    assert len(train_acc) == len(test_acc), \
        "Number of train acc and test acc must be same"

    epoch = len(train_acc)
    x = range(1, epoch + 1)
    plt.clf()
    plt.plot(x, train_acc, 'r--', label='train')
    plt.plot(x, test_acc, 'b--', label='test')
    plt.plot(x, train_acc, 'ro-', x, test_acc, 'bo-')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.legend()
    plt.savefig(save_name)


if __name__ == '__main__':
    metric_name = '/home/liuhy/res/deep-learning/03.分类/pytorch-cifar/metric/metric_DenseNet.json'
    plot_loss(metric_name)
    plot_acc(metric_name)
