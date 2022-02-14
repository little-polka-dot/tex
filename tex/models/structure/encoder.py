import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from tex.models.transformer import attention
from tex.models.backbone import resnet
from tex.models.backbone import gcnet


class ResEncoder(nn.Module):
    """ ResNet + FPN + GCNet """

    def __init__(self, d_input, d_model, block: Union[str, resnet.Block], n_layers, d_layers):
        super(ResEncoder, self).__init__()
        if isinstance(block, str): block = getattr(resnet, block)
        self.header = nn.Sequential(
            nn.Conv2d(d_input, d_layers[0], (7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(d_layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.vis_layers = nn.ModuleList()
        self.fpn_layers = nn.ModuleList()
        self.smt_layers = nn.ModuleList()
        for index, n_layer in enumerate(n_layers):
            if index == 0:
                self.vis_layers.append(
                    resnet.make_layer(
                        n_layer,
                        block,
                        d_layers[index],
                        d_layers[index],
                        stride=(1, 1),
                        sub=gcnet.gc(
                            d_layers[index] * block.expansion, d_layers[index] * block.expansion)
                    )
                )
            else:
                self.vis_layers.append(
                    resnet.make_layer(
                        n_layer,
                        block,
                        d_layers[index - 1] * block.expansion,
                        d_layers[index],
                        stride=(2, 2),
                        sub=gcnet.gc(
                            d_layers[index] * block.expansion, d_layers[index] * block.expansion)
                    )
                )
                self.smt_layers.append(
                    nn.Conv2d(d_model, d_model, (3, 3), stride=(1, 1), padding=(1, 1))
                )
            self.fpn_layers.append(
                nn.Conv2d(d_layers[index] * block.expansion, d_model, (1, 1))
            )

    def forward(self, x):
        vis = [self.header(x)]  # [c1, c2, c3, c4, c5]
        for layer in self.vis_layers: vis.append(layer(vis[-1]))
        fpn = []  # [m5, m4, m3, m2]
        for index, o in enumerate(reversed(vis[1:]), start=1):
            fpn.append(self.fpn_layers[-1 * index](o))
        lat = []  # [m5, m4, m3, m2]
        for o in fpn:
            if lat:
                size = (o.size(-2), o.size(-1))
                t = F.interpolate(
                    lat[-1], size=size, mode='bilinear', align_corners=True)
                lat.append(t + o)
            else:
                lat.append(o)
        smt = [lat[0]]  # [p5, p4, p3, p2]
        for index, layer in enumerate(self.smt_layers, start=1):
            smt.append(layer(lat[index]))
        smt = [o.view(o.size(0), o.size(1), -1) for o in smt]
        return torch.cat(smt, -1).transpose(-1, -2)


def ResEncoder50(d_input, d_model):
    return ResEncoder(d_input, d_model, 'BottleNeck', [3, 4, 6, 3], [64, 128, 256, 512])


class PosEncoder(nn.Module):

    def __init__(self, d_input, d_model, n_head, d_k, d_ffn, layers, dropout=0.1):
        super(PosEncoder, self).__init__()
        self.pos_mapping = nn.Sequential(
            nn.Linear(d_input, d_model, bias=False),
            nn.LayerNorm(d_model),
        )  # 该模型不具有平移不变性与尺度不变性
        self.encoders = nn.ModuleList([
            attention.EncodeLayer(
                d_model, n_head, d_k, d_ffn, dropout=dropout) for _ in range(layers)
        ])

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.pos_mask(x)
        x = self.pos_mapping(x)
        for layer in self.encoders:
            x = layer(x, mask)
        # [batch_size, len, dim]
        return x, mask  # mask to dec

    @staticmethod
    def pos_mask(x):
        # [batch_size, sql_len, d_input] -> [batch_size, 1, sql_len]
        return ((x > 0) | (x < 0)).any(-1).unsqueeze(-2)


if __name__ == '__main__':
    i = torch.randn((10, 2, 224, 224))
    m = ResEncoder50(2, 64)
    i = m(i)
    print(i.size())

    # import numpy as np
    # x = torch.tensor(np.array(range(1,10)).reshape((1,1,3,3)), dtype=torch.double)
    # y = F.interpolate(x, size=(5, 5), mode='bilinear', align_corners=True)
    # print(x)
    # print(y)
    # y = F.interpolate(x, size=(5, 5), mode='bilinear', align_corners=False)
    # print(y)
    # y = F.upsample(x, size=(5, 5), mode='bilinear')
    # print(y)
