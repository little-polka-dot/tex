import torch
import torch.nn as nn
from typing import Type, Union
from tex.utils.functional import optional_function, gt, mul, is_odd, map_, list_


class Block(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(Block, self).__init__()
        self.sub_method = sub
        self.downsample = None
        if gt(stride, 1) or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion, (1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):  # pre-activation
        return optional_function(self.sub_method, lambda i: i)(
            self.net(x)) + optional_function(self.downsample, lambda i: i)(x)


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


class BottleNeck(Block):

    expansion = 4

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(BottleNeck, self).__init__(in_planes, planes, stride, sub)
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


class BottleNeckX(Block):
    """ ResNeXt Block """

    expansion = 2

    def __init__(self, in_planes, planes, stride=(1, 1), groups=32, sub=None):
        super(BottleNeckX, self).__init__(in_planes, planes, stride, sub)
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, planes, (1, 1)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (3, 3), padding=(1, 1), stride=stride, groups=groups),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (1, 1)),
        )


class CoTAttention(nn.Module):

    def __init__(self, d_model, d_hidden, kernel_size, stride=(1, 1)):
        super(CoTAttention, self).__init__()
        assert is_odd(kernel_size)  # 卷积尺度必须为奇数
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
        if gt(stride, 1):
            self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = optional_function(self.pool, lambda i: i)(x)  # bs,c,h,w
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
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, planes, (1, 1)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            CoTAttention(planes, planes, (3, 3), stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (1, 1)),
        )


def make_layer(layers: int, block: Type[Block],
               in_planes, planes, stride=(1, 1), sub=None):
    return nn.Sequential(
        block(in_planes, planes, stride=stride, sub=sub),
        *[
            block(
                planes * block.expansion, planes, sub=sub
            ) for _ in range(1, layers)
        ]
    )


class ContextModeling(nn.Module):

    def __init__(self, channels):
        super(ContextModeling, self).__init__()
        self.score_conv = nn.Conv2d(channels, 1, (1, 1))

    def forward(self, x):  # x: [batch_size, c, h, w]
        score = self.score_conv(x).view(x.size(0), 1, -1)  # [batch_size, 1, h*w]
        score = torch.softmax(score, dim=-1).unsqueeze(-1)  # [batch_size, 1, h*w, 1]
        x = x.view(x.size(0), x.size(1), -1).unsqueeze(1)  # [batch_size, 1, c, h*w]
        return torch.matmul(x, score).transpose(1, 2)  # [batch_size, c, 1, 1]


class ContextTransformer(nn.Module):

    def __init__(self, channels, d_hidden):
        super(ContextTransformer, self).__init__()
        self.conv_1 = nn.Conv2d(channels, d_hidden, (1, 1))
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.conv_2 = nn.Conv2d(d_hidden, channels, (1, 1))

    def forward(self, x):
        return self.conv_2(  # [batch_size, channels, 1, 1]
            torch.relu(  # [batch_size, d_hidden, 1, 1]
                self.layer_norm(
                    self.conv_1(x).view(x.size(0), -1))  # [batch_size, d_hidden]
            ).unsqueeze(-1).unsqueeze(-1)
        )


class GlobalContextBlock(nn.Module):

    def __init__(self, channels, d_hidden):
        super(GlobalContextBlock, self).__init__()
        self.net = nn.Sequential(
            ContextModeling(channels), ContextTransformer(channels, d_hidden))

    def forward(self, x): return x + self.net(x)


class BackbonePreprocess(nn.Module):

    def __init__(self, d_input, d_model):
        super(BackbonePreprocess, self).__init__()
        # TODO: diff between 7x7 and 3x3
        self.net = nn.Sequential(
            nn.Conv2d(d_input, d_model, (7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(d_model, d_model, (1, 1), bias=False)
        )

    def forward(self, x): return self.net(x)


class BackboneEncoder(nn.Module):

    def __init__(self, d_input, d_model,
                 block: Union[Block, str], layers, d_layer=(64, 128, 256, 512)):
        super(BackboneEncoder, self).__init__()
        if isinstance(block, str): block = globals()[block]
        self.pre_layer = BackbonePreprocess(d_input, d_layer[0])
        self.layers = nn.ModuleList([
            make_layer(
                layers[0],
                block,
                d_layer[0],
                d_layer[0],
                stride=(1, 1),
                sub=GlobalContextBlock(
                    d_layer[0] * block.expansion, d_layer[0] * block.expansion)
            ),
            make_layer(
                layers[1],
                block,
                d_layer[0] * block.expansion,
                d_layer[1],
                stride=(2, 2),
                sub=GlobalContextBlock(
                    d_layer[1] * block.expansion, d_layer[1] * block.expansion)
            ),
            make_layer(
                layers[2],
                block,
                d_layer[1] * block.expansion,
                d_layer[2],
                stride=(2, 2),
                sub=GlobalContextBlock(
                    d_layer[2] * block.expansion, d_layer[2] * block.expansion)
            ),
            make_layer(
                layers[3],
                block,
                d_layer[2] * block.expansion,
                d_layer[3],
                stride=(2, 2),
                sub=GlobalContextBlock(
                    d_layer[3] * block.expansion, d_layer[3] * block.expansion)
            ),
        ])
        self.layers_map = nn.ModuleList([
            nn.Conv2d(
                d_layer[1] * block.expansion, d_model, (1, 1), bias=False),
            nn.Conv2d(
                d_layer[2] * block.expansion, d_model, (1, 1), bias=False),
            nn.Conv2d(
                d_layer[3] * block.expansion, d_model, (1, 1), bias=False)
        ])

    def forward(self, x):
        """
        将细粒度，中粒度，粗粒度三个特征图合并为一个向量
        输出： [batch_size, img_len(取决于输入图像的size), d_model]
        """
        x = self.pre_layer(x)
        output = None
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx > 0:
                vec = self.layers_map[idx - 1](x)
                vec = vec.view(vec.size(0), vec.size(1), -1)
                if output is None:
                    output = vec
                else:
                    output = torch.cat((output, vec), -1)
        return output.transpose(1, 2)


if __name__ == '__main__':
    net = BackboneEncoder(1, 256, 'BasicBlock', [3, 4, 6, 3])
    i = torch.randn((10, 1, 224, 224))
    print(net(i).size())
