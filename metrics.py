import copy

import torch
from collections import defaultdict
import numpy as np
from torch import nn

class AverageMeterVector(object):
    """Computes and stores the average and current value
       Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, N):
        self.reset(N)

    def reset(self, N):
        self.avg = np.zeros(N)
        self.sum = np.zeros(N)
        self.count = np.zeros(N)

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,), per_class = False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    if per_class:
        # per class accuracy, only top1
        num_classes = output.size(1)
        res_per_class = torch.zeros(num_classes)
        rec_num = torch.zeros(num_classes)
        for class_i in range(num_classes):
            correct_class = correct * (target.view(1, -1) == class_i).expand_as(pred)
            correct_k = correct_class[0].view(-1).float().sum(0)
            rec_num[class_i] = torch.sum(target == class_i)
            res_per_class[class_i] = (correct_k.mul_(100.0 / rec_num[class_i])) if rec_num[class_i]>0 else 0.0
        return res_per_class, rec_num
    else:
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(1. / batch_size))
#     return res


def cross_entropy_smooth(input, target, size_average=True, label_smoothing=0.1):
    y = torch.eye(10).cuda()
    lb_oh = y[target]

    target = lb_oh * (1 - label_smoothing) + 0.5 * label_smoothing

    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone


class SummaryWriterDummy:
    def __init__(self, logdir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
