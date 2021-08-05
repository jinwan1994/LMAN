import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class HSAModule(nn.Module):
    def __init__(self, channels ,kernel_size= 9):
        super(HSAModule, self).__init__()
        
        self.compress = nn.Conv2d(channels, channels//channels, 1 , padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.spatial = nn.Conv2d(channels//channels, channels//channels, kernel_size, padding=((kernel_size-1)//2))
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.compress(input)
        x = self.relu(x)
        x = self.spatial(x)
        x = self.sigmoid(x)
        return input * x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,groups=4):

        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(conv(n_feats, n_feats, 1, bias=bias))
        self.conv2 = nn.ModuleList([conv(n_feats//groups, n_feats//groups, 3, bias=bias) for _ in range(groups-1)])
        self.conv3 = nn.Sequential(conv(n_feats, n_feats, 1, bias=bias))
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(n_feats, 16)
        self.groups = groups

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        xs = torch.chunk(res, self.groups, 1)
        ys = [None] * self.groups
        ys[0] = xs[0]
        ys[1] = self.relu(self.conv2[0](xs[1] + ys[0]))
        ys[2] = self.relu(self.conv2[1](xs[2] + ys[1]))
        ys[3] = self.relu(self.conv2[2](xs[3] + ys[2]))
        res = self.conv3(torch.cat(ys, 1))
        res = self.se(res)
        res += x

        return res

class ResGroup(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResGroup, self).__init__()
        n_resblocks = 4
        self.resblocks = nn.ModuleList([ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)])
        self.hse = nn.Sequential(HSAModule(n_feats))
        self.conv1 = nn.Sequential(conv(n_feats*n_resblocks, n_feats, 1, bias=bias))
        self.n_resblocks = n_resblocks

    def forward(self, x):
        bs = [None] * self.n_resblocks
        for i in range(self.n_resblocks):
            if i == 0:
                bs[i] = self.resblocks[i](x)
            else:
                bs[i] = self.resblocks[i](bs[i-1])
        for i in range(self.n_resblocks):
            bs[i] = self.hse(bs[i])
        res = self.conv1(torch.cat(bs, 1))
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

