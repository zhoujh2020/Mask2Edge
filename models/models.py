#!/user/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from os.path import isfile

from models.utils.cofusion import CoFusion
from .mask2former.mask_former_head import MaskFormerHead
from .efficientnet.efficientnet_b7 import EfficientNet
from .unetPlusPlus import UnetPlusPlus
from .utils.DWConv import DepthWiseConv

class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        self.rate = rate
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = cfg
        self.rate = 4
        # self.resnext = resnext101_5out.ResNeXt101()
        self.efficientnet_b7 = EfficientNet.from_pretrained('efficientnet-b7')
        self.mask2former = MaskFormerHead()
        self.unetPlusPlus = UnetPlusPlus()
        self.conv0_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            DepthWiseConv(in_channel=64, out_channel=64),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(144, 128, kernel_size=1, stride=1, padding=0),
            DepthWiseConv(in_channel=128, out_channel=128),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(240, 256, kernel_size=1, stride=1, padding=0),
            DepthWiseConv(in_channel=256, out_channel=256),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv0_3 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0),
            DepthWiseConv(in_channel=512, out_channel=512),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv0_4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            DepthWiseConv(in_channel=1024, out_channel=1024),
            # nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(48, 128, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(80, 256, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(224, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(640, 1024, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU()
        # )
        self.conv6 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(100),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(100),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(100, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(100, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(100, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.msblock0 = MSBlock(100, self.rate)
        self.msblock1 = MSBlock(100, self.rate)
        self.msblock2 = MSBlock(256, self.rate)
        self.msblock3 = MSBlock(512, self.rate)
        self.msblock4 = MSBlock(1024, self.rate)
        
        self.score_dsn0 = nn.Conv2d(32, 1, (1, 1), stride=1)
        self.score_dsn1 = nn.Conv2d(32, 1, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(32, 1, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(32, 1, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(32, 1, (1, 1), stride=1)
        self.weight_deconv1 = make_bilinear_weights(4, 1).cuda()
        self.weight_deconv2 = make_bilinear_weights(8, 1).cuda()
        self.weight_deconv3 = make_bilinear_weights(16, 1).cuda()
        self.weight_deconv4 = make_bilinear_weights(32, 1).cuda()
        self.weight_deconv5 = make_bilinear_weights(64, 1).cuda()
        self.attention = CoFusion(5, 5)

    def init_weight(self):

        print("=> Initialization by Gaussian(0, 0.01)")

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.01)
                if not ly.bias is None: ly.bias.data.zero_()

        if self.cfg.pretrained:
            if not isfile(self.cfg.pretrained):
                print("No pretrained VGG16 model found at '{}'".format(self.cfg.pretrained))

            else:
                print("=> Initialize VGG16 backbone")

                state_dict = torch.load(self.cfg.pretrained, map_location=torch.device("cpu"))

                self_state_dict = self.state_dict()
                for k, v in self_state_dict.items():
                    if k in state_dict.keys():
                        self_state_dict.update({k: state_dict[k]})
                        print("*** Load {} ***".format(k))

                self.load_state_dict(self_state_dict)
                print("=> Pretrained Loaded")

    def load_checkpoint(self):
        if isfile(self.cfg.resume):
            print("=> Loading checkpoint '{}'".format(self.cfg.resume))
            checkpoint = torch.load(self.cfg.resume)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint '{}'"
                  .format(self.cfg.resume))

        else:
            print("=> No checkpoint found at '{}'".format(self.cfg.resume))

    def forward(self, x):
        # resnext_output = self.resnext(x)
        bs, _, h, w = x.shape
        efficient_output, ultra = self.efficientnet_b7.extract_endpoints(x)
        ultra_c = []
        for i in range(len(ultra)):
            if i % 2 != 0 and i < 4:
                ultra_c.append(torch.concat((ultra[i - 1], ultra[i]), dim=1))
            elif i >= 4:
                ultra_c.append(ultra[i])
        # print(len(ultra_c))
        x0 = self.conv0_0(efficient_output['reduction_1'])
        x1 = self.conv0_1(torch.concat((efficient_output['reduction_3'], ultra_c[0]), dim=1))
        x2 = self.conv0_2(torch.concat((efficient_output['reduction_4'], ultra_c[1]), dim=1))
        x3 = self.conv0_3(torch.concat((efficient_output['reduction_6'], ultra_c[2]), dim=1))
        x4 = self.conv0_4(torch.concat((efficient_output['reduction_8'], ultra_c[3]), dim=1))
        scores = [x0, x1, x2, x3, x4]
        scores = self.unetPlusPlus(scores)
        mask_input = scores
        mask_output = self.mask2former(mask_input)

        scores[0] = self.conv6(mask_output['pred_masks'])
        scores[1] = self.conv7(mask_output['pred_logits'][3])
        scores[2] = self.conv8(mask_output['pred_logits'][2])
        scores[3] = self.conv9(mask_output['pred_logits'][1])
        scores[4] = self.conv10(mask_output['pred_logits'][0])

        img_H, img_W = x.shape[2], x.shape[3]
        ## transpose and crop way
        scores[0] = self.msblock0(scores[0])
        scores[1] = self.msblock1(scores[1])
        scores[2] = self.msblock2(scores[2])
        scores[3] = self.msblock3(scores[3])
        scores[4] = self.msblock4(scores[4])
        scores[0] = self.score_dsn0(scores[0])
        scores[1] = self.score_dsn1(scores[1])
        scores[2] = self.score_dsn2(scores[2])
        scores[3] = self.score_dsn3(scores[3])
        scores[4] = self.score_dsn4(scores[4])
        upsample1 = torch.nn.functional.conv_transpose2d(scores[0], self.weight_deconv1, stride=2)
        upsample2 = torch.nn.functional.conv_transpose2d(scores[1], self.weight_deconv2, stride=4)
        upsample3 = torch.nn.functional.conv_transpose2d(scores[2], self.weight_deconv3, stride=8)
        upsample4 = torch.nn.functional.conv_transpose2d(scores[3], self.weight_deconv4, stride=16)
        upsample5 = torch.nn.functional.conv_transpose2d(scores[4], self.weight_deconv5, stride=32)
        ### center crop
        so1 = crop_bdcn(upsample1, img_H, img_W, 1, 1)
        so2 = crop_bdcn(upsample2, img_H, img_W, 2, 2)
        so3 = crop_bdcn(upsample3, img_H, img_W, 4, 4)
        so4 = crop_bdcn(upsample4, img_H, img_W, 8, 8)
        so5 = crop_bdcn(upsample5, img_H, img_W, 16, 16)
        results = [so1, so2, so3, so4, so5]
        fuse = self.attention(results)
        results.append(fuse)
        results = [torch.sigmoid(r) for r in results]
        return results

    def update(self, state_dict):
        del state_dict['fc.bias']
        del state_dict['fc.weight']
        return state_dict


# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop_bdcn(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert (h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
    return data


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)


