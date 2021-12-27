import torch
import torch.nn as nn
from typing import Type


class Block(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1), downsample=True, sub=None):
        super(Block, self).__init__()
        self.sub = sub
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion, (1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        # net采用pre-activation机制
        def _call(v, func=None): return func(v) if callable(func) else v
        return _call(self.net(x), self.sub) + _call(x, self.downsample)


class BasicBlock(Block):

    def __init__(self, in_planes, planes, stride=(1, 1), downsample=True, sub=None):
        super(BasicBlock, self).__init__(in_planes, planes, stride, downsample, sub)
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, planes, (3, 3), padding=(1, 1), stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (3, 3), padding=(1, 1)),
        )


class Bottleneck(Block):

    expansion = 4

    def __init__(self, in_planes, planes, stride=(1, 1), downsample=True, sub=None):
        super(Bottleneck, self).__init__(in_planes, planes, stride, downsample, sub)
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, planes, (1, 1)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (3, 3), padding=(1, 1), stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (1, 1)),
        )


def make_layer(layers: int, block: Type[Block], in_planes, planes, stride=(1, 1), sub=None):
    return nn.Sequential(
        block(
            in_planes, planes, stride=stride, downsample=True, sub=sub
        ),
        *[
            block(
                planes * block.expansion, planes, downsample=False, sub=sub
            ) for _ in range(1, layers)
        ]
    )


class CoTAttention(nn.Module):

    def __init__(self, d_model, d_hidden, kernel_size=(3, 3), padding=(1, 1), alpha=9):
        super(CoTAttention, self).__init__()
        self.alpha = alpha
        self.key_mapping = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size, padding=padding, bias=False),  # TODO: groups?
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
            nn.Conv2d(d_hidden, d_model * alpha, (1, 1))
        )

    def forward(self, x):
        k_1 = self.key_mapping(x)  # bs,c,h,w
        val = self.val_mapping(x).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        atn = self.atn_mapping(torch.cat([k_1, x], dim=1))  # bs,c*alpha,h,w
        atn = atn.view(x.size(0), x.size(1), self.alpha, x.size(2), x.size(3))
        atn = atn.mean(2, keepdim=False).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        k_2 = torch.softmax(atn, dim=-1) * val  # bs,c,h*w
        return k_1 + k_2.view(*x.size())  # bs,c,h,w


class CoTBlock(Block):

    expansion = 4

    def __init__(self, in_planes, planes, stride=(1, 1), downsample=True, sub=None):
        super(CoTBlock, self).__init__(in_planes, planes, stride, downsample, sub)
        if stride[0] * stride[1] > 1:
            self.net = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes, planes, (1, 1)),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.AvgPool2d((3, 3), stride, padding=(1, 1)),
                CoTAttention(planes, planes, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes * self.expansion, (1, 1)),
            )
        else:
            self.net = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes, planes, (1, 1)),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                CoTAttention(planes, planes, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes * self.expansion, (1, 1)),
            )


class MaskedBlock(nn.Module):

    def __init__(self, in_planes, planes, h, w):
        super(MaskedBlock, self).__init__()
        self.row_conv = nn.Conv2d(in_planes, planes, (1, w))
        self.col_conv = nn.Conv2d(in_planes, planes, (h, 1))

    def forward(self, x):
        row = torch.sigmoid(self.row_conv(x))
        col = torch.sigmoid(self.col_conv(x))
        return x + torch.matmul(row, col)


if __name__ == '__main__':
    # net = CoTBlock(64, 32, (1, 1), True)
    # i = torch.randn((10, 64, 56, 56))
    # print(net(i).size())
    net = MaskedBlock(3, 3, 6, 9)
    i = torch.randn((10, 3, 6, 9))
    print(net(i).size())
