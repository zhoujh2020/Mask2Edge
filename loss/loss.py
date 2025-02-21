import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms


def clip_by_value(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """

    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

class SegmentationLosses(object):
    def __init__(self, size_average=True, batch_average=True, cuda=False):
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'focal2':
            return self.FocalLoss2
        elif mode == 'attention':
            print('attention loss')
            return self.AttentionLoss
        else:
            raise NotImplementedError

    def FocalLoss(self, prediction, label, balanced_w=1.1):

        label = label.float()
        prediction = prediction.float()

        with torch.no_grad():
            num_pos = torch.sum(label == 1).float()
            num_neg = torch.sum(label == 0).float()
            alpha = num_neg / (num_pos + num_neg) * 1.0
            eps = 1e-14
            p_clip = torch.clamp(prediction, min=eps, max=1.0 - eps)

            mask = label * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
                   (1.0 - label) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
            mask[label == 2] = 0
            mask = mask.detach()

        # print('bce')
        temp = torch.nn.functional.binary_cross_entropy(prediction.float(), label.float(), weight=mask, reduction='none')
        cost = torch.sum(temp)

        return cost


if __name__ == "__main__":
    loss = SegmentationLosses()

    logit = torch.rand(8, 4, 7, 7)
    target = torch.rand(8, 7, 7)

   # a = a.view(8,196)

    print(loss.FocalLoss(logit, target, gamma=2, alpha=0.5).item())




