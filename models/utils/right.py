import torch
import math
import torch.nn as nn
from models.utils.neck import DeformLayer
from models.utils.Corr import CoordAtt
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Batchnorm1d(nn.Module):
    def __init__(self, c_num:int, eps:float = 1e-10):
        super(Batchnorm1d,self).__init__()
        self.gamma = nn.Parameter(torch.randn(1, c_num, 1))
        self.beta  = nn.Parameter(torch.zeros(1, c_num, 1))
        self.eps   = eps

    def forward(self, x):
        N, Q, K  = x.size()
        x        = x.transpose(0,1).flatten(1)
        mean        = x.mean(dim = 1, keepdim = True )
        std         = x.std (dim = 1, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(Q, N, K).transpose(0, 1)
        return x * self.gamma + self.beta
"""
class Channel_Norm(nn.Module):
    def __init__(self, in_channels:int, oup_channels:int):
        super().__init__()
        
        self.conv0 = nn.Conv1d(in_channels, in_channels, 9, stride=1, padding=4)
        self.conv1 = nn.Conv1d(in_channels, in_channels, 9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.bn    = Batchnorm1d(oup_channels)
        self.relu = nn.ReLU()
        # self.h_swish = h_swish()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
    
        x = x.transpose(1,2)
        x1 = self.relu(self.bn(self.conv0(x)))
        x1 = self.conv1(x1)
        w_gamma = self.bn.gamma.unsqueeze(3)
        w_gamma = self.relu(self.conv2(w_gamma))
        w_gamma = self.softmax(self.conv3(w_gamma))
        y = (x1 * w_gamma.squeeze(3)).sum(1)
        
        return y
"""        
class Channel_Norm2(nn.Module):
    def __init__(self, in_channels:int, oup_channels:int):
        super().__init__()
        
        self.conv0 = nn.Conv1d(in_channels, in_channels, 9, stride=1, padding=4)
        self.conv1 = nn.Conv1d(in_channels, in_channels, 9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        
        self.conv4 = nn.Conv1d(in_channels, in_channels, 9, stride=1, padding=4, groups=in_channels)
        self.conv6 = nn.Conv1d(in_channels, in_channels, 9, stride=1, padding=4, groups=in_channels)
        
        self.bn    = Batchnorm1d(oup_channels)
        self.relu = nn.ReLU()

        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.softmax1 = nn.Softmax(dim=-1)

    def forward(self,x,pos2,pos3):
    
        B, Ks, Q = x.size()
        x = x.transpose(1,2)
        x1 = self.relu(self.bn(self.conv0(x)))
        x1 = self.conv1(x1)
        w_gamma = self.bn.gamma.unsqueeze(3)
        w_gamma = self.relu(self.conv2(w_gamma))
        w_gamma = self.softmax(self.conv3(w_gamma))   # self.softmax
        w_ammag = 1 - w_gamma
        x2 = x1 * w_gamma.squeeze(3)
        x3 = x1 * w_ammag.squeeze(3)
        
        x2 = self.conv4(x2) + pos2 
        x3 = self.conv6(x3) + pos3 

        y_attn = x2 @ x3.transpose(1,2)
        y2 = self.softmax1(y_attn) @ x3
        y3 = self.softmax1(y_attn.transpose(1,2)) @ x2

        y3_std = y3.std(dim=2,keepdim=True)
        res_attn = self.softmax(y3_std * y3) 
        res = (y3 * res_attn).sum(1) 
        
        y2_std = y2.std(dim=2,keepdim=True)
        fin_attn = self.softmax(y2_std * y2)
        fin = (y2 * fin_attn).sum(1) 
        
        return  res + fin  # 
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = BasicConv(256, 256, kernel_size, stride=1, padding=(kernel_size-1) // 2, groups=1, bn=False, relu=False)
        
    def forward(self, x):
        x_out = self.spatial(x)
        scale = torch.softmax(x_out.flatten(2),dim=-1)
        return  scale
        
class GCBlock8(nn.Module):     # simplest
    def __init__(self, inplane, ouplane, kernel):
        super(GCBlock8, self).__init__()
        
        self.kernel = kernel
                                                                                                                  
                                                                                       
        self.weight_linear = nn.Linear(100, kernel * 2) 
        
        self.deweight_linear = nn.Linear(2 * kernel, 100)
        
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, value, hard_sigmoid_masks, pos2, pos3):
        Q, B, C = query.size()
        bs, c, h, w = value.size()
        
        conv1d = self.weight_linear(query.permute(1,2,0))

        former_conv1d, weight_conv1d = torch.split(conv1d, [9, 9], dim=-1)  # .view(B, C, 1, 3, 3)
        former_conv1d, weight_conv1d = former_conv1d.view(B, C, 1, 3, 3), weight_conv1d.view(B, C, 1, 3, 3)
        
        filt1 = former_conv1d
        
        value = value.unsqueeze(1)
        
        temp = []
        for i in range(B):
            out = F.conv2d(input=value[i], weight=filt1[i], groups=C, padding="same")   
            temp.append(out)
        y_temp = torch.cat(temp, dim=0)
        
        diff = torch.exp(-(value.squeeze(1) - y_temp) ** 2)

        filt2 = torch.abs(weight_conv1d).detach()
        
        diff = diff.unsqueeze(1)
        res_temp = []        
        for i in range(B):
            out = F.conv2d(input=diff[i], weight=filt2[i], groups=C, padding="same")
            res_temp.append(out)
        y_diff9 = torch.cat(res_temp, dim=0)

        res = []
        for i in range(B):
            out = F.conv2d(input= value[i] * diff[i], weight=weight_conv1d[i], groups=C, padding="same") 
            res.append(out)
           
        res_diff = []
        weights_d = weight_conv1d.sum(dim=[2, 3], keepdim=True)
        for i in range(B):
            out = F.conv2d(input= value[i] * diff[i], weight=weights_d[i], groups=C, padding="same")  
            res_diff.append(out)
        
        y_diff = (torch.cat(res, dim=0) - torch.cat(res_diff, dim=0))/(y_diff9 + 1e-10)
                        
        y = y_temp - y_diff
        
        weight1 = torch.cat((former_conv1d.flatten(2), weight_conv1d.flatten(2)), dim=-1)
        q = self.deweight_linear(weight1)
        q = q.permute(2,0,1) 
                  
        return q, y  
        
class GCBlock9(nn.Module):     # simplest
    def __init__(self, inplane, ouplane, kernel):
        super(GCBlock9, self).__init__()
        
        self.kernel = kernel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.Dynamic_propose = nn.Sequential(nn.Conv2d(inplane, 4 * inplane, 1, 1, 0),
                                             nn.BatchNorm2d(4 * inplane),
                                             nn.ReLU())
        
        self.Dynamic_weight_channel = nn.Sequential(
                                                     nn.Conv2d(4 * inplane, 3 * inplane, 1, 1, 0),
                                                     nn.Sigmoid()
                                                    )                             
                                           
        self.weight_linear = nn.Linear(100, 3 * kernel) 
        
        self.deweight_linear = nn.Linear(3 * kernel, 100)
        
                                 
    def forward(self, query, value, hard_sigmoid_masks):
        Q, B, C = query.size()
        bs, c, h, w = value.size()
        
        Dynamic_propose = self.Dynamic_propose(self.avg_pool(value))
        
        Dynamic_channel = self.Dynamic_weight_channel(Dynamic_propose)       
        Dynamic_former_channel, Dynamic_weight_channel, Dynamic_exp_channel = torch.split(Dynamic_channel, [256, 256, 256], dim=1)
        Dynamic_former_channel, Dynamic_weight_channel, Dynamic_exp_channel = Dynamic_former_channel.view(bs, c, 1, 1, 1), Dynamic_weight_channel.view(bs, c, 1, 1, 1), Dynamic_exp_channel.view(bs, c, 1, 1, 1) 

        former_conv1d, weight_conv1d, exp_conv1d = torch.split(self.weight_linear(query.permute(1,2,0)), [9, 9, 9], dim=-1)
        former_conv1d, weight_conv1d, exp_conv1d = former_conv1d.view(B, C, 1, 3, 3), weight_conv1d.view(B, C, 1, 3, 3), exp_conv1d.view(B, C, 1, 3, 3)
        former_conv1d, weight_conv1d, exp_conv1d = former_conv1d * Dynamic_former_channel, weight_conv1d * Dynamic_weight_channel, exp_conv1d * Dynamic_exp_channel
        
        filt1 = former_conv1d # (torch.ones(B, C, 1, 3, 3) / 9).float()
        # filt1.requires_grad = False
        # filt1 = filt1.cuda()
        
        value = value.unsqueeze(1)
        
        temp = []
        for i in range(B):
            out = F.conv2d(input=value[i], weight=filt1[i], groups=C, padding="same")   # the number of groups can be changed
            temp.append(out)
        y_temp = torch.cat(temp, dim=0)
        
        diff = torch.exp(-(value.squeeze(1) - y_temp) ** 2)
        diff1, diff2 = diff, diff # torch.split(diff, [128, 128], dim=1)
        value = value.squeeze(1)
        value1, value2 = value, value# torch.split(value, [128, 128], dim=1)
        weight1_conv1d, weight2_conv1d = weight_conv1d, exp_conv1d # torch.split(weight_conv1d, [128,128], dim=1)
        
        weight = torch.zeros_like(diff1).unsqueeze(2).float()
        weight = weight.repeat(1, 1, 9, 1, 1)
        
        weight[:, :, 0, 1:, 1:] = diff1[:, :, :h-1, :w-1]
        weight[:, :, 1, 1:, :] = diff1[:, :, :h-1, :]
        weight[:, :, 2, 1:, :w-1] = diff1[:, :, :h-1, 1:]        
        weight[:, :, 3, :, 1:] = diff1[:, :, :, :w-1]                
        weight[:, :, 4, :, :] = diff1[:, :, :, :]                
        weight[:, :, 5, :, :w-1] = diff1[:, :, :, 1:]        
        weight[:, :, 6, :h-1, 1:] = diff1[:, :, 1:, :w-1]        
        weight[:, :, 7, :h-1, :] = diff1[:, :, 1:, :]
        weight[:, :, 8, :h-1, :w-1] = diff1[:, :, 1:, 1:]
        
               
        Difference = torch.zeros_like(value1).unsqueeze(2).float()
        Difference = Difference.repeat(1, 1, 9, 1, 1)
        
        Difference[:, :, 0, 1:, 1:] = value1[:, :, :h-1, :w-1]
        Difference[:, :, 1, 1:, :] = value1[:, :, :h-1, :]
        Difference[:, :, 2, 1:, :w-1] = value1[:, :, :h-1, 1:]        
        Difference[:, :, 3, :, 1:] = value1[:, :, :, :w-1]                
        Difference[:, :, 4, :, :] = value1[:, :, :, :]      
        Difference[:, :, 5, :, :w-1] = value1[:, :, :, 1:]        
        Difference[:, :, 6, :h-1, 1:] = value1[:, :, 1:, :w-1]        
        Difference[:, :, 7, :h-1, :] = value1[:, :, 1:, :]
        Difference[:, :, 8, :h-1, :w-1] = value1[:, :, 1:, 1:]
        
        Difference_weight = torch.sigmoid(torch.abs(weight * Difference - (diff1 * value1).unsqueeze(2)))
        
        y_diff1 = (Difference_weight * Difference * weight1_conv1d.flatten(2).view(B, C, 9, 1, 1)).sum(2)
        
        filt2 = torch.abs(weight2_conv1d).detach()
        
        diff2 = diff2.unsqueeze(1)
        res_temp = []        
        for i in range(B):
            out = F.conv2d(input=diff2[i], weight=filt2[i], groups=C, padding="same")
            res_temp.append(out)
        y_diff9 = torch.cat(res_temp, dim=0)

        res = []
        for i in range(B):
            out = F.conv2d(input= value2[i] * diff2[i], weight=weight2_conv1d[i], groups=C, padding="same") 
            res.append(out)
           
        res_diff = []
        weights_d = weight2_conv1d.sum(dim=[2, 3], keepdim=True)
        for i in range(B):
            out = F.conv2d(input= value2[i] * diff2[i], weight=weights_d[i], groups=C, padding="same")  
            res_diff.append(out)
        
        y_diff2 = (torch.cat(res, dim=0) - torch.cat(res_diff, dim=0))/(y_diff9 + 1e-10)
        
        # y_diff = torch.cat((y_diff1, y_diff2), dim=1)
        
        y = y_temp - y_diff1 + y_temp - y_diff2
        
        # weight_conv1d = torch.cat((weight1_conv1d, weight2_conv1d), dim=1)
        q = self.deweight_linear(torch.cat((former_conv1d.flatten(2), weight1_conv1d.flatten(2), weight2_conv1d.flatten(2)), dim=-1)) 
        q = q.permute(2,0,1) 
                  
        return q, y  

class GCBlock10(nn.Module):     # simplest
    def __init__(self, inplane, ouplane, kernel):
        super(GCBlock10, self).__init__()
        
        self.kernel = kernel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.Dynamic_propose = nn.Sequential(nn.Conv2d(inplane, 4 * inplane, 1, 1, 0),
                                             nn.BatchNorm2d(4 * inplane),
                                             nn.ReLU())
        
        self.Dynamic_weight_channel = nn.Sequential(
                                                     nn.Conv2d(4 * inplane, 2 * inplane, 1, 1, 0),
                                                     nn.Sigmoid()
                                                    )                             
                                           
        self.weight_linear = nn.Linear(100, 2 * kernel) 
        
        self.deweight_linear = nn.Linear(2 * kernel, 100)
        
        self.feature_linear = nn.Sequential(
                                            nn.Conv2d(inplane, 4 * inplane, 1, 1, 0),
                                            nn.BatchNorm2d(4 * inplane),
                                            nn.ReLU(),
                                            nn.Conv2d(4 * inplane, inplane, 1, 1, 0)
                                            )
        
                                 
    def forward(self, query, value, hard_sigmoid_masks):
        Q, B, C = query.size()
        bs, c, h, w = value.size()
        
        Dynamic_propose = self.Dynamic_propose(self.avg_pool(value))
        
        Dynamic_channel = self.Dynamic_weight_channel(Dynamic_propose)       
        Dynamic_former_channel, Dynamic_weight_channel = torch.split(Dynamic_channel, [256, 256], dim=1)
        Dynamic_former_channel, Dynamic_weight_channel = Dynamic_former_channel.view(bs, c, 1, 1, 1), Dynamic_weight_channel.view(bs, c, 1, 1, 1)

        former_conv1d, weight_conv1d = torch.split(self.weight_linear(query.permute(1,2,0)), [9, 9], dim=-1)
        former_conv1d, weight_conv1d = former_conv1d.view(B, C, 1, 3, 3), weight_conv1d.view(B, C, 1, 3, 3)
        former_conv1d, weight_conv1d = former_conv1d * Dynamic_former_channel, weight_conv1d * Dynamic_weight_channel
        
        filt1 = former_conv1d # (torch.ones(B, C, 1, 3, 3) / 9).float()
        # filt1.requires_grad = False
        # filt1 = filt1.cuda()
        
        value = value.unsqueeze(1)
        
        temp = []
        for i in range(B):
            out = F.conv2d(input=value[i], weight=filt1[i], groups=C, padding="same")   # the number of groups can be changed
            temp.append(out)
        y_temp = torch.cat(temp, dim=0)
        
        diff = torch.exp(-(value.squeeze(1) - y_temp) ** 2)
        
        weight = torch.zeros_like(diff).unsqueeze(2).float()
        weight = weight.repeat(1, 1, 9, 1, 1)
        
        weight[:, :, 0, 1:, 1:] = diff[:, :, :h-1, :w-1]
        weight[:, :, 1, 1:, :] = diff[:, :, :h-1, :]
        weight[:, :, 2, 1:, :w-1] = diff[:, :, :h-1, 1:]        
        weight[:, :, 3, :, 1:] = diff[:, :, :, :w-1]                
        weight[:, :, 4, :, :] = diff[:, :, :, :]                
        weight[:, :, 5, :, :w-1] = diff[:, :, :, 1:]        
        weight[:, :, 6, :h-1, 1:] = diff[:, :, 1:, :w-1]        
        weight[:, :, 7, :h-1, :] = diff[:, :, 1:, :]
        weight[:, :, 8, :h-1, :w-1] = diff[:, :, 1:, 1:]
        
        value = value.squeeze(1)       
        Difference = torch.zeros_like(value).unsqueeze(2).float()
        Difference = Difference.repeat(1, 1, 9, 1, 1)
        
        Difference[:, :, 0, 1:, 1:] = value[:, :, :h-1, :w-1]
        Difference[:, :, 1, 1:, :] = value[:, :, :h-1, :]
        Difference[:, :, 2, 1:, :w-1] = value[:, :, :h-1, 1:]        
        Difference[:, :, 3, :, 1:] = value[:, :, :, :w-1]                
        Difference[:, :, 4, :, :] = value[:, :, :, :]      
        Difference[:, :, 5, :, :w-1] = value[:, :, :, 1:]        
        Difference[:, :, 6, :h-1, 1:] = value[:, :, 1:, :w-1]        
        Difference[:, :, 7, :h-1, :] = value[:, :, 1:, :]
        Difference[:, :, 8, :h-1, :w-1] = value[:, :, 1:, 1:]
        
        Difference_weight = torch.abs(weight * Difference - (diff * value).unsqueeze(2))
        
        y_diff = (torch.sigmoid(Difference_weight) * Difference * weight_conv1d.flatten(2).view(B, C, 9, 1, 1)).sum(2)  # - value.unsqueeze(2)
        
        # Final_weight = 1 + torch.tanh((Difference_weight.sum(2))/9)
                
        y = y_temp - y_diff # Final_weight * ()
        # y = self.feature_linear(y)
        
        q = self.deweight_linear(torch.cat((former_conv1d.flatten(2), weight_conv1d.flatten(2)), dim=-1)) 
        q = q.permute(2,0,1) 
                  
        return q, y                      
