import numpy as np
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
        def _call(v, func=None): return func(v) if func else v
        return _call(
            self.net(x), self.sub) + _call(x, self.downsample)


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


if __name__ == '__main__':
    net = make_layer(3, Bottleneck, 64, 64)
    i = torch.tensor(np.random.random((10, 64, 56, 56)), dtype=torch.float)
    print(net(i).size())
