import torch
import torch.nn as nn
from tex.utils.functional import any_gt, mul, all_odd, map_, list_


class Block(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(Block, self).__init__()
        self.subnetwork = sub
        self.downsample = None
        if any_gt(stride, 1) or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion, (1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        r = x  # id(r) == id(x)
        if callable(self.downsample):
            r = self.downsample(r)
        x = self.net(x)
        if callable(self.subnetwork):
            x = self.subnetwork(x)
        return torch.relu(r + x)


class CoTAttention(nn.Module):

    def __init__(self, d_model, d_hidden, kernel_size, stride=(1, 1)):
        super(CoTAttention, self).__init__()
        assert all_odd(kernel_size)  # 卷积尺度必须为奇数
        padding = list_(map_(lambda x: (x - 1) // 2, kernel_size))
        self.kernel_size = kernel_size
        self.key_mapping = nn.Sequential(  # TODO: conv groups=4 ?
            nn.Conv2d(d_model, d_model, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.val_mapping = nn.Sequential(
            nn.Conv2d(d_model, d_model, (1, 1), bias=False),
            nn.BatchNorm2d(d_model)
        )
        self.atn_mapping = nn.Sequential(
            nn.Conv2d(2 * d_model, d_hidden, (1, 1), bias=False),
            nn.BatchNorm2d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_hidden, d_model * mul(kernel_size), (1, 1))
        )
        self.pool = None
        if any_gt(stride, 1):
            self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if callable(self.pool): x = self.pool(x)  # bs,c,h,w
        k_1 = self.key_mapping(x)  # bs,c,h,w
        val = self.val_mapping(x).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        atn = self.atn_mapping(torch.cat([k_1, x], dim=1))  # bs,c*alpha,h,w
        atn = atn.view(x.size(0), x.size(1), mul(self.kernel_size), x.size(2), x.size(3))
        atn = atn.mean(2, keepdim=False).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        k_2 = torch.softmax(atn, dim=-1) * val  # bs,c,h*w
        return k_1 + k_2.view(*x.size())  # bs,c,h,w


class CoTBottleNeck(Block):

    expansion = 4

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(CoTBottleNeck, self).__init__(in_planes, planes, stride, sub)
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, (1, 1)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            CoTAttention(planes, planes, (3, 3), stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (1, 1)),
            nn.BatchNorm2d(planes * self.expansion),
        )
