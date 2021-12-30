import torch
import torch.nn as nn
from typing import Type


class Block(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(Block, self).__init__()
        assert stride[0] > 0 and stride[1] > 0
        self.sub_method = sub
        self.downsample = None
        if (stride[0] > 1 or stride[1] > 1) or (in_planes != planes * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion, (1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):  # pre-activation
        def _call(v, func=None): return func(v) if callable(func) else v
        return _call(
            self.net(x), self.sub_method) + _call(x, self.downsample)


class BasicBlock(Block):

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(BasicBlock, self).__init__(in_planes, planes, stride, sub)
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

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(Bottleneck, self).__init__(in_planes, planes, stride, sub)
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
        block(in_planes, planes, stride=stride, sub=sub),
        *[
            block(
                planes * block.expansion, planes, sub=sub
            ) for _ in range(1, layers)
        ]
    )


class ContextualAttention(nn.Module):

    def __init__(self, d_model, d_hidden,
            kernel_size, padding=(1, 1), alpha=9, dropout=0.1):
        super(ContextualAttention, self).__init__()
        self.alpha = alpha
        self.key_mapping = nn.Sequential(  # TODO: why groups
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
            nn.Conv2d(d_hidden, d_model * alpha, (1, 1))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k_1 = self.key_mapping(x)  # bs,c,h,w
        val = self.val_mapping(x).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        atn = self.atn_mapping(torch.cat([k_1, x], dim=1))  # bs,c*alpha,h,w
        atn = atn.view(x.size(0), x.size(1), self.alpha, x.size(2), x.size(3))
        atn = atn.mean(2, keepdim=False).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        k_2 = torch.softmax(atn, dim=-1) * val  # bs,c,h*w
        return self.dropout(k_1 + k_2.view(*x.size()))  # bs,c,h,w


class ContextualBlock(Block):

    expansion = 4

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(ContextualBlock, self).__init__(in_planes, planes, stride, sub)
        if stride[0] > 1 or stride[1] > 1:
            self.net = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes, planes, (1, 1)),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.AvgPool2d((3, 3), stride=stride, padding=(1, 1)),
                ContextualAttention(planes, planes, (3, 3), padding=(1, 1)),
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
                ContextualAttention(planes, planes, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes * self.expansion, (1, 1)),
            )


if __name__ == '__main__':
    net = ContextualBlock(64, 32, (2, 2))
    i = torch.randn((10, 64, 56, 56))
    print(net(i).size())
